import requests
import argparse
import random
import logging

from models.mistral7b_mlx import load_model
from gen_speculative import gen_speculative

# pretends to just predict last token
def mock_speculation(curr):
    return [[curr[-1]]]

class SpeculatorClient:
    def __init__(self, addr, port, min_tokens):
        self.addr = addr
        self.port = port
        self.url = f'http://{addr}:{port}'

        self.session = requests.Session()
        self.session.get(self.url)
        self.session_id = random.randint(0, 1000000000000)
        self.min_tokens = min_tokens

    def send_request(self, curr, min_tokens=4):
        data = {'tokens': curr, 'session_id': self.session_id, 'min_tokens': self.min_tokens}
        
        response = self.session.post(self.url, json=data)
        received_data = response.json()
        
        return received_data['tokens']

# TODO: this should be also a service, which connects to the speculator
# this service would wait for queries with some typical API, queue them and run model
# we can also dump kv cache somewhere on disk?

def generate():
    parser = argparse.ArgumentParser(description="Mistral inference script")
    parser.add_argument(
        "--model-path",
        type=str,
        default="mlx_model",
        help="The path to the model weights and tokenizer",
    )
    parser.add_argument(
        "--temp",
        help="The sampling temperature.",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--addr",
        type=str,
        default="localhost",
        help="Address where to find speculator server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8808,
        help="Port where to find speculator server",
    )
    parser.add_argument(
        "--prompt",
        help="The message to be processed by the model",
        default="London is a capital of",
    )
    parser.add_argument(
        "--min-tokens",
        help="How many tokens to wait for from speculator model",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--max-tokens",
        help="How many tokens to generate",
        type=int,
        default=64,
    )

    args = parser.parse_args()
    model, tokenizer = load_model(args.model_path)

    client = SpeculatorClient(args.addr, args.port, min_tokens=args.min_tokens)
    prompt = tokenizer.encode(args.prompt)

    gen_speculative(model, tokenizer, prompt, client.send_request, max_tokens=args.max_tokens)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    generate()