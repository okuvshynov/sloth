from models.mistral import load_model
import argparse
from collections import Counter

import mlx.core as mx

import fewlines.metrics as fm

def prob_sample(logits):
    samples = Counter(mx.random.categorical(logits).item() for _ in range(1000))
    print(samples)

def rank_of(index_a, logits):
    a = [(value, index) for index, value in enumerate(logits)]
    a.sort(reverse=True, key=lambda x: x[0])
    a = [original_index for _, original_index in a]
    return a.index(index_a)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mistral inference script")
    parser.add_argument(
        "--model_a",
        type=str,
        default="mlx_model",
        help="The path to the model weights and tokenizer",
    )
    parser.add_argument(
        "--model_b",
        type=str,
        default="mlx_model",
        help="The path to the model weights and tokenizer",
    )
    parser.add_argument(
        "--prompt",
        help="The message to be processed by the model",
        default="London is a capital of",
    )

    args = parser.parse_args()
    model_a, tokenizer = load_model(args.model_a)
    model_b, _ = load_model(args.model_b)

    curr = mx.array(tokenizer.encode(args.prompt))
    cache_a = None
    cache_b = None

    while True:
        x = curr[None]
        #print(x)
        logits, cache_a = model_a(x, cache_a)
        y = mx.argmax(logits[:, -1, :])
        #print(y)
        logits, cache_b = model_b(x, cache_b)
        rank = rank_of(y.item(), logits[0, -1, :].tolist())
        fm.add("rank", rank)
        prob_sample(logits=logits[0, -1, :])
        curr = mx.array([y.item()])
        for l in fm.histogram("rank", color='green', n_lines=3):
            print(l)