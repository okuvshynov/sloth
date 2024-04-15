# let's test access to speculator first

import requests
import argparse
import random

from mistral7b.mistral_mlx_scratch import load_model, speculative_loop

def send_request(curr, session_id=None):
    addr = 'localhost'
    port = 8808

    url = f'http://{addr}:{port}'

    if session_id is None:
        session_id = random.randint(0, 1000000000000)
    
    data = {'tokens': curr, 'session_id': session_id}
    
    response = requests.post(url, json=data)
    received_data = response.json()
    
    print('Response from server:', received_data)
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
    args = parser.parse_args()
    model, tokenizer = load_model(args.model_path)

    prompt = [1, 4222, 349, 264, 5565, 302, 28705]

    speculative_loop(model, tokenizer, prompt, send_request)


if __name__ == '__main__':
    generate()