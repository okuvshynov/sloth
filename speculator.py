from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging

import argparse

from mistral7b.mistral_mlx_scratch import load_model
import mlx.core as mx

import fewlines.timer as ft
import fewlines.dashboard as fd
import fewlines.metrics as fm

default_max_tokens = 256
default_min_tokens = 8

def _longest_prefix(a, b):
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return i
    return min(len(a), len(b))

# this one is non-thread safe
# it would perform just two operations:
#  - update current (based on new input)
#  - perform exploration step
# all concurrency between background exploration and new inputs from the client 
# would be handled externally

class Speculator:
    def __init__(self, model_path):
        # here we'll initialize model and current search tree
        logging.info(f"loading model from {model_path}")
        self.model, self.tokenizer = load_model(model_path)
        self.session_id = None

        # need this for cache operations, might need to move somewhere
        self.num_layers = len(self.model.layers)

        # Cache is 6D: [layer, k|v, batch_index, head_index, position, data_index]
        self.cache = [None for _ in self.model.layers]

        # length of sequence for which we computed the cache
        self.cache_len = 0
        self.tokens = []


    def handle_query(self, request):
        # request must have:
        # session_id. If changed, all previous session data is gone
        # tokens -- either initial prompt or incremental update within the session
        # max_tokens -- how many more tokens to generate after the current incremental update.

        self.max_tokens = request.get('max_tokens', default_max_tokens)

        if self.session_id is None or self.session_id != request['session_id']:
            # starting new session
            self.session_id = request['session_id']
            logging.info(f"starting session {self.session_id}")
            self.tokens = request['tokens'][:]
            self.cache = [None for _ in self.model.layers]
            self.cache_len = 0
        else:
            # we need to compare self.prompt + self.generated to request['tokens']
            # and strip generated/cache if needed
            new_tokens = request['tokens'][:]
            
            match_len = _longest_prefix(self.tokens, new_tokens)

            # if everything matched. We can keep all generated + cache, 
            # even what we have generated after the new prompt. Otherwise:

            if match_len < len(new_tokens):

                # and the generated tokens we need to update to all the passed tokens
                self.tokens = new_tokens

                for i in range(len(self.cache)):
                    # either all caches are empty, or all are not
                    if self.cache[i] is None:
                        break
                    K = self.cache[i][0][:, :, :match_len, :]
                    V = self.cache[i][1][:, :, :match_len, :]
                    self.cache[i] = K, V
                # TODO: can this be off by 1?
                # we can keep the cache up to match_len. 
                # Or should it be match len - 1? no, we'll compute the same thing again anyway
                logging.info(f'updating cache len from {self.cache_len} to {match_len}')
                self.cache_len = match_len
                
    def gen_next(self):
        if self.session_id is None:
            return
        
        # need to find the input. It is a difference between populated to cache and current tokens
        tokens_to_process = self.tokens[self.cache_len:]
        logging.info(f"tokens to process: {tokens_to_process}")

        x = mx.array(tokens_to_process)[None]
        logits, local_cache = self.model(x, self.cache)
        y = mx.argmax(logits[:, -1, :]).item()
        self.tokens.append(y)

        for i, (local_K, local_V) in enumerate(local_cache):
            assert local_V.shape[2] == len(tokens_to_process)
            assert local_K.shape[2] == len(tokens_to_process)
            
            if self.cache[i] is not None:
                K, V = self.cache[i]
                K = mx.concatenate([K, local_K], axis=2)
                V = mx.concatenate([V, local_V], axis=2)
                self.cache[i] = K, V
            else:
                self.cache[i] = local_K, local_V
        self.cache_len += len(tokens_to_process)

class SpeculatorHTTPHandler(BaseHTTPRequestHandler):
    def __init__(self, speculator, *args, **kwargs):
        self.speculator: Speculator = speculator
        super().__init__(*args, **kwargs)

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        req = json.loads(post_data.decode('utf-8'))
        res = {}
        if 'tokens' not in req or 'session_id' not in req or len(req['tokens']) == 0:
            logging.warn('requests must contain non-empty prompt and session_id') 
        else:
            with ft.Timer('handle_query_latency') as _:
                self.speculator.handle_query(req)
            for i in range(8):
                with ft.Timer('gen_next_latency') as _:
                    self.speculator.gen_next()

            # TODO: we send back one of the tokens from input. Need to fix that
            
            new_tokens = self.speculator.tokens[len(req['tokens']) - 1:]
            logging.info(f'generated tokens: {self.speculator.tokens, new_tokens}')
            res['tokens'] = new_tokens

        # TODO: send NOT success if request is not well-formed
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(res).encode())

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain; charset=utf-8')
        self.end_headers()
        self.wfile.writelines(f'{l}\n'.encode() for l in fd.histograms('*latency*'))
        self.wfile.writelines(f'Tokens to process: {l}\n'.encode() for l in fm.histogram("tokens_to_process"))


class SpeculatorHTTPServer(HTTPServer):
    def __init__(self, server_address, RequestHandlerClass, speculator):
        self.speculator = speculator
        super().__init__(server_address, RequestHandlerClass)
    
    def finish_request(self, request, client_address):
        # Pass db_connection to the handler
        self.RequestHandlerClass(self.speculator, request, client_address, self)


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
    args = parser.parse_args()

    speculator = Speculator(args.model_path)
    
    httpd = SpeculatorHTTPServer((args.addr, args.port), SpeculatorHTTPHandler, speculator)
    logging.info(f"Speculator server started on http://{args.addr}:{args.port}")
    httpd.serve_forever()

