# modified version of one_file_ref.py by mistral-ai, Apache Licence 2.0


# this implementation focuses on inference for the speculative tree-like decoding
# - We do have kv-cache but just 1 of them, rather than batch_size
# - After each tree evaluation we pick the longest matching subsequence
# - (and we'll need to update cache for that)
# - cache is for the shared prefix

from pathlib import Path
from sentencepiece import SentencePieceProcessor
from torch import nn
from typing import Optional, Tuple, List
import json
import torch
import logging
from mistral7b_conf import ModelArgs
import os
import sys

def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=2)
    values = torch.repeat_interleave(values, repeats=repeats, dim=2)
    return keys, values

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    assert freqs_cos.dim() == 3
    freqs_cos = freqs_cos.unsqueeze(2)
    freqs_sin = freqs_sin.unsqueeze(2)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.dbg = False

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads
        
        self.repeats = self.n_heads // self.n_kv_heads
        self.sliding_window = self.args.sliding_window

        self.scale = self.args.head_dim**-0.5

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * args.head_dim,
            bias=False
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            bias=False
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            bias=False
        )
        self.wo = nn.Linear(
            args.n_heads * args.head_dim,
            args.dim,
            bias=False
        )

        # This is shared cache for all suffixes.
        # A cache for already 'confirmed' sequence
        self.cache_k = torch.empty(
            (
                1,
                args.sliding_window,
                self.n_kv_heads,
                self.args.head_dim,
            ), dtype=torch.float16
        ).to('mps')
        self.cache_v = torch.empty(
            (
                1,
                args.sliding_window,
                self.n_kv_heads,
                self.args.head_dim,
            ), dtype=torch.float16
        ).to('mps')        

        # this is local cache.
        # after each iteration once we decided how much/if at all we reuse
        # we merge data from selected cache with longest match to the shared cache
        self.local_cache_k = torch.empty(
            (
                args.max_batch_size,
                args.max_speculative_seq,
                self.n_kv_heads,
                self.args.head_dim,
            ), dtype=torch.float16
        ).to('mps')
        self.local_cache_v = torch.empty(
            (
                args.max_batch_size,
                args.max_speculative_seq,
                self.n_kv_heads,
                self.args.head_dim,
            ), dtype=torch.float16
        ).to('mps')        


    # there needs to be another method for prefilling main cache
    def forward(
            self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, positions: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        
        bsz, seqlen, _ = x.shape
        assert seqlen <= self.args.max_speculative_seq

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.args.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.args.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.args.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # here we need to: 
        #  1. Fill in local_caches
        #  2. populate key/value from global cache

        self.local_cache_k[:bsz, :seqlen] = xk
        self.local_cache_v[:bsz, :seqlen] = xv

        # now we need to merge data from global cache and new xk/xv
        # global cache would have shape [1, sliding_window, n_kv_heads, head_dim] == [1, 4096, 8, 128]
        # xk, xv would have shape [bsz, seqlen, n_kv_heads, head_dim] == [?, <, 8, 128]

        # key, value should have shape
        # [bsz, position[-1] + seqlen, n_kv_heads, head_dim] and then we repeat it over dim=2

        if self.dbg:
            print(positions)

        # TODO: just pass single pos here?
        prev_pos = positions[0][0].item()
        key = torch.cat((self.cache_k[:, :prev_pos, ...].expand(bsz, -1, -1, -1), xk), dim=1)
        if self.dbg:
            print(key)
        value = torch.cat((self.cache_v[:, :prev_pos, ...].expand(bsz, -1, -1, -1), xv), dim=1)

        key, value = repeat_kv(key, value, self.repeats)

        query = xq.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        # scores : [bsz, n_heads, seqlen | 1, seqlen]
        scores = torch.matmul(query, key.transpose(2, 3)) * self.scale

        if self.dbg:
            print(scores.shape, mask.shape)
        
        if mask is not None:
            scores[:, :, :, -seqlen:] += mask[:, None, :, :]
        
        scores = scores.float()
        scores = nn.functional.softmax(scores, dim=-1).type_as(query)
        
        output = torch.matmul(scores, value)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        # TODO: are we sure about this 
        output = torch.nan_to_num(output)
        
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(
            args.dim,
            args.hidden_dim,
            bias=False
        )
        self.w2 = nn.Linear(
            args.hidden_dim,
            args.dim,
            bias=False
        )
        self.w3 = nn.Linear(
            args.dim,
            args.hidden_dim,
            bias=False
        )

    def forward(self, x) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args
    
    def forward_all(
            self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, positions: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin, positions, mask)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


class TransformerShared(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        assert self.vocab_size > 0

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.layers = torch.nn.ModuleList(
            [TransformerBlock(args=args) for _ in range(args.n_layers)]
        )
        #self.layers[0].attention.dbg = True

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.output = nn.Linear(
            args.dim,
            args.vocab_size,
            bias=False
        )

        freqs_cos, freqs_sin = precompute_freqs_cis(self.args.head_dim, 128_000)
        freqs_sin = freqs_sin.to("mps")
        freqs_cos = freqs_cos.to("mps")
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        prompt_lens
    ):
        h = self.tok_embeddings(input_ids)

        # this became 3d tensor in contrast with more typical 2d. 
        # we can look at it later nad make it back to 2d.
        freqs_cos = self.freqs_cos[positions]
        freqs_sin = self.freqs_sin[positions]

        mask: Optional[torch.Tensor] = None
        if input_ids.shape[1] > 1:
            bsz, n = input_ids.shape

            mask = torch.full((bsz, n, n), fill_value=0.0, dtype=input_ids.dtype, device=input_ids.device)

            for i in range(bsz):
                length = prompt_lens[i]
                tmp_mask = torch.tril(torch.ones((length, length), dtype=input_ids.dtype, device=input_ids.device), diagonal=0)
                tmp_mask = torch.triu(tmp_mask, diagonal=-self.args.sliding_window)
                mask[i, :length, :length] = tmp_mask[:length, :length]

            mask = torch.log(mask)

        for layer in self.layers:
            h = layer.forward_all(h, freqs_cos, freqs_sin, positions, mask)

        res = self.output(self.norm(h))

        return res.float()

    @staticmethod
    def from_folder(folder: Path, max_batch_size: int = 1, device="mps", dtype=torch.float16):
        with open(folder / 'params.json', 'r') as f:
            model_args = ModelArgs(**json.loads(f.read()))
        model_args.max_batch_size = max_batch_size
        model_args.max_speculative_seq = 8
        logging.info('creating model instance')
        model = TransformerShared(model_args).to(device=device, dtype=dtype)
        logging.info('loading checkpoint')
        filename = os.path.join(folder, 'consolidated.00.pth')
        loaded = torch.load(filename, mmap=True)
        logging.info('loading state dictionary')
        model.load_state_dict(loaded, strict=False)

        return model


class Tokenizer:
    def __init__(self, model_path: str):
        assert Path(model_path).exists(), model_path
        self._model = SentencePieceProcessor(model_file=model_path)
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
        return self._model.decode(t)
    
def gen_next_token_batch(input_tokens, prompt_lens, model: TransformerShared, tokenizer: Tokenizer):
    input_mask = input_tokens != tokenizer.pad_id

    all_positions = torch.arange(0, input_tokens.shape[1]).repeat(input_tokens.shape[0], 1).to('mps')
    all_positions = all_positions * input_mask
    logits = model.forward(input_tokens, all_positions, prompt_lens)

    logprobs = nn.functional.log_softmax(logits, dim=-1)

    min_len = min(prompt_lens)
    for i, l in enumerate(prompt_lens):
        for li in range(min_len - 1, l):
            tok = torch.argmax(logprobs[i, li,:], dim=-1)
            print(f'{i} {l} {li} {tok}')
        print()

    return [torch.argmax(logprobs[i, l - 1,:], dim=-1) for i, l in enumerate(prompt_lens)]

@torch.no_grad()
def gen_single_token(prompts: List[str], model: TransformerShared, tokenizer: Tokenizer):
    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]
    prompt_lens = [len(x) for x in encoded_prompts]
    max_prompt_len = max(prompt_lens)

    input_tokens = torch.full((len(prompts), max_prompt_len), tokenizer.pad_id, dtype=torch.long, device="mps")
    for i, encoded in enumerate(encoded_prompts):
        input_tokens[i, :len(encoded)] = torch.tensor(encoded).to(input_tokens)
    
    next_tokens = gen_next_token_batch(input_tokens, prompt_lens, model, tokenizer)

    res = []
    for i, x in enumerate(encoded_prompts):
        v = tokenizer.decode(x + [next_tokens[i].item()])
        res.append(v)
    return res

def pick_longest_match(prefix, suffixes, model_path):
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))

    # TODO: we should keep kv cache for the 'prefix' part
    prompts = [prefix + s for s in suffixes]

    prompt_lens = [len(x) for x in prompts]
    max_prompt_len = max(prompt_lens)

    input_tokens = torch.full((len(prompts), max_prompt_len), tokenizer.pad_id, dtype=torch.long, device="mps")
    for i, encoded in enumerate(prompts):
        input_tokens[i, :len(encoded)] = torch.tensor(encoded).to(input_tokens)

    print(f'input tokens = {input_tokens}')

    model = TransformerShared.from_folder(Path(model_path), max_batch_size=len(suffixes))
    input_mask = input_tokens != tokenizer.pad_id

    all_positions = torch.arange(0, input_tokens.shape[1]).repeat(input_tokens.shape[0], 1).to('mps')
    all_positions = all_positions * input_mask
    logits = model.forward(input_tokens, all_positions, prompt_lens)
    logprobs = nn.functional.log_softmax(logits, dim=-1)

    best = []
    best_i = -1
    for i, candidate in enumerate(suffixes):
        l = len(candidate)
        curr = []
        for j in range(l + 1):
            li = len(prefix) + j - 1
            tok = torch.argmax(logprobs[i, li,:], dim=-1)
            
            curr.append(tok)
            if (j < l and tok != candidate[j]):
                # first non-matching token    
                break
        #print(curr)
        if len(curr) > len(best):
            best = curr
            best_i = i

    # TODO: and write to cache somewhere here?
    return best

def apply_tree(model_path):
    prefix = [1, 330, 365, 334] # A B C
    suffixes_m = [
        [[384], [384, 333], [384, 413, 401], [384, 401, 413]],
        [[420], [420, 444], [420, 382], [420, 315]],
        [[315], [315, 315], [315, 375], [315, 375, 876]]
    ]
    max_batch_size = max(len(s) for s in suffixes_m)
    print(f'max_batch_size = {max_batch_size}')

    model = TransformerShared.from_folder(Path(model_path), max_batch_size=max_batch_size)
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))

    # do in two steps:
    # 1. populate prefix
    # 2. evaluate suffixes


    input_tokens = torch.full((1, len(prefix)), tokenizer.pad_id, dtype=torch.long, device="mps")
    input_tokens[0, :] = torch.tensor(prefix).to(input_tokens)
    print(f'input_tokens: {input_tokens}')
    all_positions = torch.arange(0, input_tokens.shape[1]).repeat(input_tokens.shape[0], 1).to('mps')
    logits = model.forward(input_tokens, all_positions, [len(prefix)])
    logprobs = nn.functional.log_softmax(logits, dim=-1)
    next_token = torch.argmax(logprobs[:, -1,:], dim=-1)

    print(f'next_token {next_token}')
    #print(model.layers[0].attention.local_cache_v)
    #print(model.layers[0].attention.local_cache_k)
    #print(model.layers[0].attention.cache_k.shape)

    seq_len = len(prefix)
    curr_pos = seq_len
    for layer in model.layers:
        layer.attention.cache_k[0, :seq_len] = layer.attention.local_cache_k[0, :seq_len]
        layer.attention.cache_v[0, :seq_len] = layer.attention.local_cache_v[0, :seq_len]

    for si, suffixes in enumerate(suffixes_m):
        # now evaluate some options
        prompt_lens = [len(x) for x in suffixes]
        max_prompt_len = max(prompt_lens)

        input_tokens = torch.full((len(suffixes), max_prompt_len), tokenizer.pad_id, dtype=torch.long, device="mps")
        for i, encoded in enumerate(suffixes):
            input_tokens[i, :len(encoded)] = torch.tensor(encoded).to(input_tokens)

        print(f'input_tokens: {input_tokens}')

        input_mask = input_tokens != tokenizer.pad_id

        all_positions = torch.arange(curr_pos, curr_pos + input_tokens.shape[1]).repeat(input_tokens.shape[0], 1).to('mps')
        all_positions = all_positions * input_mask

        print(f'all_positions {all_positions}')

        logits = model.forward(input_tokens, all_positions, prompt_lens)
        logprobs = nn.functional.log_softmax(logits, dim=-1)
        #print(logprobs.shape)

        #print(torch.argmax(logprobs, dim=2))

        best = []
        best_i = -1
        for i, candidate in enumerate(suffixes):
            l = len(candidate)
            curr = []
            for j in range(l):
                tok = torch.argmax(logprobs[i, j,:], dim=-1)
                
                curr.append(tok)
                if (j + 1 < l and tok != candidate[j + 1]):
                    # first non-matching token    
                    break
            #print(curr)
            if len(curr) > len(best):
                best = curr
                best_i = i

        print(best)
        # TODO: and write to cache somewhere here?
        seq_len = len(best)
        print(seq_len)
        curr_pos += seq_len

        for layer in model.layers:
            layer.attention.cache_k[0, curr_pos:curr_pos+seq_len] = layer.attention.local_cache_k[best_i, :seq_len]
            layer.attention.cache_v[0, curr_pos:curr_pos+seq_len] = layer.attention.local_cache_v[best_i, :seq_len]
    return best

if __name__ == '__main__':
    #print(pick_longest_match([1, 330, 365, 334], [[], [123], [384, 413, 401], [384, 413, 401, 420], [384, 401, 413]], sys.argv[1]))
    #print(apply_tree([1, 330, 365, 334], [[384], [384, 333], [384, 413, 401], [384, 401, 413]], sys.argv[1]))
    apply_tree(sys.argv[1])
