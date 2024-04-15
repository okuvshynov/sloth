import requests
import argparse
import random
import logging

from mistral7b.mistral_mlx_scratch import load_model, speculative_loop

# pretends to just predict last token
def mock_speculation(curr):
    return [[curr[-1]]]

class SpeculatorClient:
    def __init__(self, addr, port):
        self.addr = addr
        self.port = port
        self.url = f'http://{addr}:{port}'

        self.session = requests.Session()
        self.session.get(self.url)
        self.session_id = random.randint(0, 1000000000000)

    def send_request(self, curr):
        data = {'tokens': curr, 'session_id': self.session_id}
        
        response = self.session.post(self.url, json=data)
        received_data = response.json()
        
        return [received_data['tokens']]

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
        "--speculator-addr",
        type=str,
        default="localhost",
        help="Address where to find speculator server.",
    )
    parser.add_argument(
        "--speculator-port",
        type=int,
        default=8808,
        help="Port where to find speculator server",
    )
    args = parser.parse_args()
    model, tokenizer = load_model(args.model_path)

    client = SpeculatorClient(args.speculator_addr, args.speculator_port)
    
    # London is a capital of
    prompt = [1, 4222, 349, 264, 5565, 302, 28705]

    speculative_loop(model, tokenizer, prompt, client.send_request, max_tokens=64)
    #speculative_loop(model, tokenizer, prompt, mock_speculation, max_tokens=64)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    generate()