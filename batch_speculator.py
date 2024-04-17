import logging

from models.mistral7b_mlx import load_model

import fewlines.timer as ft
import mlx.core as mx

from pprint import pprint

default_max_tokens = 256
default_min_tokens = 8

def _longest_prefix(a, b):
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return i
    return min(len(a), len(b))

class BatchSpeculator:
    def __init__(self, model_path, batch_size=8, temp=1.0):
        # here we'll initialize model and current search tree
        logging.info(f"loading model from {model_path}")
        self.model, self.tokenizer = load_model(model_path)
        self.pad = self.tokenizer.pad_id
        self.session_id = None
        self.batch_size = batch_size
        self.temp = temp

        # need this for cache operations, might need to move somewhere
        self.num_layers = len(self.model.layers)

        # Cache is 6D: [layer, k|v, batch_index, head_index, position, data_index]
        self.shared_cache = [None for _ in self.model.layers]
        self.local_cache = [None for _ in self.model.layers]

        # length of sequence for which we computed the cache
        self.cache_len = 0

        # TODO: make this tensors/arrays, not python lists
        # no sharing of the tokens themselves for simplicity
        # tokens are different within batch
        self.tokens = [[] for _ in range(self.batch_size)]

    def handle_query(self, request):
        # request must have:
        # session_id. If changed, all previous session data is gone
        # tokens -- either initial prompt or incremental update within the session
        # max_tokens -- how many more tokens to generate after the current incremental update.
        with ft.Timer("handle_query_latency") as _:
            self.max_tokens = request.get('max_tokens', default_max_tokens)

            if self.session_id is None or self.session_id != request['session_id']:
                # starting new session
                self.session_id = request['session_id']
                logging.info(f"starting session {self.session_id}")

                # each sample within a batch would be the same at the start
                self.tokens = [request['tokens'][:] for _ in range(self.batch_size)]
                
                # clearing shared cache (batch size = 1)
                self.shared_cache = [None for _ in self.model.layers]
                self.cache_len = 0

                # also clear local cache
                self.local_cache = [None for _ in self.model.layers]
            else:
                # we need to compare self.prompt + self.generated to request['tokens']
                # and strip generated/cache if needed
                new_tokens = request['tokens'][:]
                
                # here we might get different matches for different sequences in batch
                # We need to find longest of them and update others accordingly - both the sequence
                # and the kv cache. What do we do with partial match? Copy from the best?
                # Cut at some point? This means we'll get different lengths of the sequences to 
                # evaluate.
                match_lens = [_longest_prefix(t, new_tokens) for t in self.tokens]
                max_match_idx = mx.argmax(mx.array(match_lens)).item()

                #pprint(new_tokens)
                #pprint(self.tokens)
                
                # TODO: for simplicity we just reset all local cache here.
                # Can probably optimize later. Local cache will be write-only in a way
                # We get local cache from the item with max match, append it to shared cache and 
                # reset all local cache
                local_cache_match_len = min(match_lens[max_match_idx], len(new_tokens) - 1) - self.cache_len
                if local_cache_match_len > 0:
                    for i in range(len(self.shared_cache)):
                        if self.local_cache[i] is None:
                            break
                        local_K, local_V = self.local_cache[i]
                        # shape is [batch_size, head, position, data]
                        # now need to select matching subset from the local cache from correct matching element in batch
                        local_K = local_K[None, max_match_idx, :, :local_cache_match_len, :]
                        local_V = local_V[None, max_match_idx, :, :local_cache_match_len, :]
                        
                        # either all caches are empty, or all are not
                        if self.shared_cache[i] is not None:
                            K, V = self.shared_cache[i]
                            K = mx.concatenate([K, local_K], axis=2)
                            V = mx.concatenate([V, local_V], axis=2)
                            self.shared_cache[i] = K, V
                        else:
                            self.shared_cache[i] = local_K, local_V

                    self.cache_len += local_cache_match_len

                for bi in range(self.batch_size):
                    if match_lens[bi] < len(new_tokens):
                        self.tokens[bi] = new_tokens[:]

                #print("AFTER CACHE CONSOLIDATE")
                #pprint(self.tokens)
                
    def gen_next(self):
        if self.session_id is None:
            return
        with ft.Timer("gen_next_latency") as _:
            with ft.Timer("inference_latency") as _:
                # need to find the input. It is a difference between 
                # processed/populated to cache and current tokens
                tokens_to_process = [t[self.cache_len:] for t in self.tokens]

                lengths = [len(s) for s in tokens_to_process]
                max_len = max(lengths)
                x = mx.array([s + [self.pad] * (max_len - len(s)) for s in tokens_to_process])
                #x = mx.array(tokens_to_process)
                #print('INPUT', x)
                
                with ft.Timer("model_eval_latency") as _:
                    # We can modify model interface to accept both shared and local cache.
                    # This way we won't need to have batch_size copies of main cache for entire model,
                    # just for a single layer. 
                    # Interface could be self.model(x, shared_cache, local_cache)
                    # with shared cache having 1 as batch dimension. local cache will be updated and returned.
                    logits, self.local_cache = self.model(x, self.shared_cache)
                    #print(self.local_cache[0][0].shape)

                for i in range(len(self.tokens)):
                    y = mx.random.categorical(logits[i, lengths[i] - 1, :] * (1.0 / self.temp)).item()
                    self.tokens[i].append(y)
                    
                # need to consolidate caches? just pass current local cache?

if __name__ == '__main__':
    import sys
    bs = BatchSpeculator(sys.argv[1], batch_size=4)
    r1 = {
        'tokens': [1, 4222, 349, 264, 5565, 302, 272],
        'session_id': 0 
    }
    r2 = {
        'tokens': [1, 4222, 349, 264, 5565, 302, 272, 2969],
        'session_id': 0
    }
    r3 = {
        'tokens': [1, 4222, 349, 264, 5565, 302, 272, 2969, 11508, 304, 272],
        'session_id': 0
    }
    bs.handle_query(r1)
    bs.gen_next()
    bs.handle_query(r2)
    bs.gen_next()
    bs.gen_next()
    bs.gen_next()
    bs.gen_next()
    bs.gen_next()
    bs.gen_next()
    bs.handle_query(r3)
