# Copyright © 2023 Apple Inc.
# From https://github.com/ml-explore/mlx-examples/tree/main/llms/mistral, MIT Licence

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from sentencepiece import SentencePieceProcessor

import fewlines.timer as ft
import fewlines.dashboard as fd

@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    rope_theta: float = 10000


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        self.rope = nn.RoPE(args.head_dim, traditional=True, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        
        # we'll need that for populating incremental cache updates
        new_values = values
        offset = 0

        if cache is not None:
            # here we read from shared cache which has B == 1
            key_cache, value_cache = cache
            key_cache = mx.tile(key_cache, [B, 1, 1, 1])
            value_cache = mx.tile(value_cache, [B, 1, 1, 1])
            
            offset = key_cache.shape[2]
            queries = self.rope(queries, offset=offset)
            new_keys = self.rope(keys, offset=offset)
            keys = mx.concatenate([key_cache, new_keys], axis=2)
            values = mx.concatenate([value_cache, new_values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
            new_keys = keys

        ext_mask = mx.zeros((L, offset + L), dtype=mask.dtype)
        ext_mask[:, -L:] = mask

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=ext_mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        # This is important - we return and store only newly computed keys/values
        # otherwise we'll spend too much memory (batch_size * entire_cache) 
        return self.wo(output), (new_keys, new_values)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> mx.array:
        r, cache = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        out = h + r
        return out, cache


class Mistral(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        assert self.vocab_size > 0
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.tok_embeddings(inputs)

        mask = None
        #if h.shape[1] > 1:
        mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
        mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        # no overwriting old cache. we'll merge after we know the max match
        new_cache = [None] * len(self.layers)

        for e, layer in enumerate(self.layers):
            h, new_cache[e] = layer(h, mask, cache[e])

        return self.output(self.norm(h)), new_cache

class Tokenizer:
    def __init__(self, model_path: str):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)
        self._sep = "▁"
        assert self._model.vocab_size() == self._model.get_piece_size()

    @property
    def eos_id(self) -> int:
        return self._model.eos_id()

    @property
    def pad_id(self) -> int:
        return self._model.pad_id()

    def encode(self, s: str) -> List[int]:
        return [self._model.bos_id(), *self._model.encode(s)]

    def decode(self, t: List[int]) -> str:
        out = self._model.decode(t)
        if t and self._model.id_to_piece(t[0])[0] == self._sep:
            return " " + out
        return out

def load_model(folder: str):
    model_path = Path(folder)
    tokenizer = Tokenizer(str(model_path / "tokenizer.model"))
    with open(model_path / "config.json", "r") as f:
        config = json.loads(f.read())
        config.pop("sliding_window", None)
        config.pop("model_type", None)
        quantization = config.pop("quantization", None)
        model_args = ModelArgs(**config)
    weights = mx.load(str(model_path / "weights.npz"))
    weights = tree_unflatten(list(weights.items()))
    model = Mistral(model_args)
    if quantization is not None:
        nn.QuantizedLinear.quantize_module(model, **quantization)
    model.update(weights)
    mx.eval(model.parameters())
    return model, tokenizer

def common_prefix_len(A, B):
    res = 0
    for a, b in zip(A, B):
        if a != b:
            break
        res += 1
    return res

def apply_tree(model: Mistral):
    prefix = [1, 330, 365, 334] # A B C
    suffixes_m = [
        [[384], [384, 333], [384, 413, 401], [384, 401, 413]],
        [[420], [420, 444], [420, 382], [420, 315]],
        [[315], [315, 315], [315, 375], [315, 375, 876]],
        [[475, 524], [475], [475, 393]],
    ]

    generated = []
    x = mx.array(prefix)[None]

    # Cache is 6D: [layer, k|v, batch_index, head_index, position, data_index]
    logits, cache = model(x)
    y = mx.argmax(logits[:, -1, :]).item()
    print(f'after prefix :{y}')
    generated.append(y)

    pad = tokenizer.pad_id
    for si, suffixes in enumerate(suffixes_m):
        lengths = [len(s) for s in suffixes]
        max_len = max(lengths)
        x = mx.array([s + [pad] * (max_len - len(s)) for s in suffixes])
        
        # here we pass shared cache with batch dim = 1
        # and get back partial cache for each new suffix in a batch
        logits, local_cache = model(x, cache)

        best = []
        best_i = -1
        for i, candidate in enumerate(suffixes):
            l = len(candidate)
            output = [mx.argmax(logits[i, j,:]).item() for j in range(len(candidate))]
            
            # TODO: are we computing 1 extra char here?

            # now we need to find the longest match between candidate and generated
            # Our output is offset by one + we need to add one non-matching token
            # For example, if the candidate was [A, B, C, D] the output would show 
            # 'what was generated after' [A], [A, B], etc. In case of perfect prediction
            # we should see something like [B, C, D, E] in the output. 
            # We add 1 as it is also 'correct' symbol produced by main model. 

            approved_len = common_prefix_len(output, candidate[1:]) + 1
            if approved_len > len(best):
                best = output[:approved_len]
                best_i = i

        print(f'next approved sequence: {best}')
        generated.extend(best)

        # Now we append the matched sequence to the global cache
        if best_i >= 0:
            new_len = len(best)

            # for each layer update the cache
            for i, (local_K, local_V) in enumerate(local_cache):
                K, V = cache[i]
                new_K = local_K[None, best_i, :, :new_len, :]
                new_V = local_V[None, best_i, :, :new_len, :]

                K = mx.concatenate([K, new_K], axis=2)
                V = mx.concatenate([V, new_V], axis=2)

                cache[i] = K, V
        
    print(tokenizer.decode(generated))

def speculative_loop(model: Mistral, tokenizer: Tokenizer, prefix, next_suffixes_fn, max_tokens=256):
    generated = []
    x = mx.array(prefix)[None]

    # Cache is 6D: [layer, k|v, batch_index, head_index, position, data_index]
    logits, cache = model(x)
    y = mx.argmax(logits[:, -1, :]).item()
    print(f'after prefix :{y}')
    generated.append(y)

    pad = tokenizer.pad_id

    started = time.time()
    while True:
        with ft.Timer("next_suffixes_fn_latency") as _:
            suffixes = next_suffixes_fn(prefix + generated)
        if len(suffixes) == 0:
            break
        lengths = [len(s) for s in suffixes]
        max_len = max(lengths)
        x = mx.array([s + [pad] * (max_len - len(s)) for s in suffixes])
        
        # here we pass shared cache with batch dim = 1
        # and get back partial cache for each new suffix in a batch
        with ft.Timer("model_eval_latency") as _:
            logits, local_cache = model(x, cache)

        best = []
        best_i = -1
        for i, candidate in enumerate(suffixes):
            output = [mx.argmax(logits[i, j,:]).item() for j in range(len(candidate))]
            
            # TODO: are we computing 1 extra char here?

            # now we need to find the longest match between candidate and generated
            # Our output is offset by one + we need to add one non-matching token
            # For example, if the candidate was [A, B, C, D] the output would show 
            # 'what was generated after' [A], [A, B], etc. In case of perfect prediction
            # we should see something like [B, C, D, E] in the output. 
            # We add 1 as it is also 'correct' symbol produced by main model. 

            approved_len = common_prefix_len(output, candidate[1:]) + 1
            if approved_len > len(best):
                best = output[:approved_len]
                best_i = i

        print(f'next approved sequence: {best}')
        generated.extend(best)

        if len(generated) >= max_tokens:
            break

        # Now we append the matched sequence to the global cache
        if best_i >= 0:
            with ft.Timer("cache_update_latency") as _:
                new_len = len(best)

                # for each layer update the cache
                for i, (local_K, local_V) in enumerate(local_cache):
                    K, V = cache[i]
                    new_K = local_K[None, best_i, :, :new_len, :]
                    new_V = local_V[None, best_i, :, :new_len, :]

                    K = mx.concatenate([K, new_K], axis=2)
                    V = mx.concatenate([V, new_V], axis=2)

                    cache[i] = K, V
        
    print(tokenizer.decode(generated))
    print(f"TPS: {len(generated) / (time.time() - started)}")
    for l in fd.dashboard({"charts" : [[("*latency", 'histogram')]], 'color': 'green', 'n_lines': 3}):
        print(l)


def test_correctness():
    prompt = [1, 4222, 349, 264, 5565, 302, 28705]

    # for mistral7b@q4
    expected = [28750, 28740, 303, 5445, 28723, 661, 349, 264, 2990, 302, 4638, 472, 28725, 5514, 28725, 304, 16863, 28723, 661, 349, 264, 2990, 302, 272, 3437, 28723, 13, 13, 20491, 349, 264, 2990, 302, 272, 2609, 28723, 661, 349, 264, 2990, 302, 3340, 28725, 9097, 28725, 304, 5679, 28723, 661, 349, 264, 2990, 302, 272, 2609, 28723, 13, 13, 20491, 349, 264, 2990, 302, 272, 2169, 28723, 661, 349, 264, 2990, 302, 272, 2169, 28723, 13, 13, 20491, 349, 264, 2990, 302, 272, 3437, 28723, 661, 349, 264, 2990, 302, 272, 3437, 28723, 13, 13, 20491, 349, 264, 2990, 302, 272]

    # for mistral7b@fp16
    expected = [28750, 28740, 303, 5445, 28723, 661, 349, 264, 2990, 302, 4638, 472, 28725, 5514, 28725, 304, 16863, 28723, 661, 349, 264, 2990, 302, 272, 3437, 28723, 661, 349, 264, 2990, 302, 272, 2169, 28723, 661, 349, 264, 2990, 302, 272, 2609, 28723, 661, 349, 264, 2990, 302, 272, 3437, 28723, 661, 349, 264, 2990, 302, 272, 2169, 28723, 661, 349, 264, 2990, 302, 272, 2609, 28723, 661, 349, 264, 2990, 302, 272, 3437, 28723, 661, 349, 264, 2990, 302, 272, 2169, 28723, 661, 349, 264, 2990, 302, 272, 2609, 28723, 661, 349, 264, 2990, 302, 272, 3437, 28723, 661, 349]

    complete = prompt + expected

    def mock_speculate(curr):
        curr = curr[:-1]
        seq_len = len(curr)

        assert curr == complete[:seq_len]

        next_tokens = complete[seq_len:]

        if len(next_tokens) == 0:
            return []

        # now we need to randomly pick some tokens which would/would not match
        p_correct = 0.8
        p_stop = 0.1
        res = []

        samples = mx.random.randint(low=1, high=2).item()

        # how many candidates do we produce?
        for _ in range(samples):
            speculation = []
            for i, tok in enumerate(next_tokens):
                speculation.append(tok if mx.random.uniform() < p_correct or i == 0 else tok + 1)
                if mx.random.uniform() < p_stop:
                    break

        res.append(speculation)

        return res

    model, tokenizer = load_model(args.model_path)

    speculative_loop(model, tokenizer, prompt, mock_speculate)


if __name__ == "__main__":
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
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")

    args = parser.parse_args()

    mx.random.seed(args.seed)

    test_correctness()

    print("[INFO] Loading model from disk.")
    model, tokenizer = load_model(args.model_path)

    apply_tree(model)