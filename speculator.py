from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging
import time

import argparse

import fewlines.timer as ft
import fewlines.dashboard as fd
import fewlines.metrics as fm

import threading
import queue

from batch_speculator import BatchSpeculator
from linear_speculator import LinearSpeculator

default_max_tokens = 256
default_min_tokens = 8

class AsyncSpeculator:
    def __init__(self, speculator):
        self.speculator = speculator
        self.queue = queue.Queue()
        self.new_tokens = None
        self.debounce_lock = threading.Lock()
        self.gen_since_last = 0
        threading.Thread(target=self.gen_loop, daemon=True).start()

    def query(self, request):
        with self.debounce_lock:
            self.queue.put(request)
            self.queue.join()
            result = self.new_tokens
            self.new_tokens = None
            return result

    def gen_loop(self):
        while True:
            try:
                req = self.queue.get(timeout=1e-4)
            except:
                req = None

            if req is not None:
                with ft.Timer("request_handle_latency") as _:
                    logging.info(f'working on {req}')
                    min_tokens = req.get('min_tokens', default_min_tokens)
                    self.speculator.handle_query(req)
                    fm.add('already_computed', len(self.speculator.tokens) - len(req['tokens']))
                    while self.gen_since_last < min_tokens:
                        self.speculator.gen_next()
                        self.gen_since_last += 1
                    self.new_tokens = [t[len(req['tokens']) - 1:] for t in self.speculator.tokens]
                    logging.info(f'generated tokens: {self.speculator.tokens, self.new_tokens}')
                    self.gen_since_last = 0
                    self.queue.task_done()
            else:
                if self.gen_since_last < 16:
                    self.speculator.gen_next()
                    self.gen_since_last += 1
                else:
                    time.sleep(0.1)

class SpeculatorHTTPHandler(BaseHTTPRequestHandler):
    def __init__(self, speculator, *args, **kwargs):
        self.async_speculator = speculator
        super().__init__(*args, **kwargs)

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        req = json.loads(post_data.decode('utf-8'))
        res = {}
        if 'tokens' not in req or 'session_id' not in req or len(req['tokens']) == 0:
            logging.warn('requests must contain non-empty prompt and session_id') 
        else:
            # TODO: we send back one of the tokens from input. Need to fix that
            new_tokens = self.async_speculator.query(req)
            res['tokens'] = new_tokens
            logging.info(f'returning {res}')

        # TODO: send NOT success if request is not well-formed
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(res).encode())

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain; charset=utf-8')
        self.end_headers()
        self.wfile.writelines(f'{l}\n'.encode() for l in fd.dashboard({"charts": [('*latency', 'histogram')], "n_lines": 1}))
        self.wfile.writelines(f'{l}\n'.encode() for l in fm.histogram("tokens_to_process"))
        self.wfile.writelines(f'{l}\n'.encode() for l in fm.histogram("already_computed"))


class SpeculatorHTTPServer(HTTPServer):
    def __init__(self, server_address, RequestHandlerClass, speculator):
        self.speculator = speculator
        self.async_speculator = AsyncSpeculator(self.speculator)
        super().__init__(server_address, RequestHandlerClass)
    
    def finish_request(self, request, client_address):
        # Pass db_connection to the handler
        self.RequestHandlerClass(self.async_speculator, request, client_address, self)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(description="speculation service")
    parser.add_argument(
        "--addr",
        type=str,
        default="0.0.0.0",
        help="Address where to start http server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8808,
        help="Port where to start http server",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="mlx_model",
        help="The path to the model weights and tokenizer",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Different speculative samples to work on",
    )
    args = parser.parse_args()

    #speculator = BatchSpeculator(args.model_path, batch_size=args.batch_size)
    speculator = LinearSpeculator(args.model_path)
    
    httpd = SpeculatorHTTPServer((args.addr, args.port), SpeculatorHTTPHandler, speculator)
    logging.info(f"Speculator server started on http://{args.addr}:{args.port}")
    httpd.serve_forever()

