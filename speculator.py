import logging
import time

import argparse
import zmq

import fewlines.timer as ft

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
        with ft.Timer("query_latency") as _:
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
                    #logging.info(f'working on {req}')
                    min_tokens = req.get('min_tokens', default_min_tokens)
                    self.speculator.handle_query(req)
                    while self.gen_since_last < min_tokens:
                        self.speculator.gen_next()
                        self.gen_since_last += 1
                    self.new_tokens = [t[len(req['tokens']) - 1:] for t in self.speculator.tokens]
                    #logging.info(f'generated tokens: {self.speculator.tokens, self.new_tokens}')
                    self.gen_since_last = 0
                    self.queue.task_done()
            else:
                if self.gen_since_last < 16:
                    self.speculator.gen_next()
                    self.gen_since_last += 1
                else:
                    time.sleep(0.1)

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
    async_speculator = AsyncSpeculator(speculator)

    context = zmq.Context()

    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{args.addr}:{args.port}")
    while True:
        req = socket.recv_json()
        new_tokens = async_speculator.query(req)
        socket.send_json({'tokens': new_tokens})

