# might run for a while
from pathlib import Path
import logging
import sys

import mistral7b_base
import mistral7b_shared_cache

def baseline(prompts, model_path):
    tokenizer = mistral7b_base.Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer_base = mistral7b_base.Transformer.from_folder(Path(model_path), max_batch_size=1)
    res = []
    for p in prompts:
        s, _ = mistral7b_base.generate([p], transformer_base, tokenizer, max_tokens=1)
        res.extend(s)
    return res

def single_batch(prompts, model_path):
    tokenizer = mistral7b_base.Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = mistral7b_shared_cache.TransformerShared.from_folder(Path(model_path), max_batch_size=len(prompts))
    return mistral7b_shared_cache.gen_single_token(prompts, transformer, tokenizer)

def run_test(model_path: str):
    prompts = [
        "A",
        "A B",
        "A B C",
        "B C D",
        "A B C D",
        "B C D E",
        "C D E F G",
        "D E F G H",
        "Explain the theory of relativity in simple te",
        "Write a short story about a robot learning to",
        "Summarize the main plot of 'Pride and",
        "What are the differences between Python 2 and",
        "Describe the process of phot",
        "How does a blockchain wor",
        "List the steps to make a chocolate ca",
        "Translate 'Hello, how are you?' into Fre",
        "What is the capital of Canad",
        "Generate a poem about the se",
        "Explain the concept of machine le",
        "What are the main causes of climate",
        "Describe how to perform CP",
        "Write a dialogue between a teacher and a student who didn't do their homew",
        "What is quantum comput",
        "Explain the rules of che",
        "Summarize the key points of World Wa",
        "What is the significance of the Turi",
        "How does the human immune system wor",
        "Write an essay on the importan"
    ]

    v0 = baseline(prompts, model_path)
    logging.info(v0)

    v1 = single_batch(prompts, model_path)
    logging.info(v1)

    logging.info(v0 == v1)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    run_test(sys.argv[1])
