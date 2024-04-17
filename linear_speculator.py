import logging

from models.mistral7b_mlx import load_model
import mlx.core as mx

import fewlines.timer as ft

default_max_tokens = 256

def _longest_prefix(a, b):
    for i, (ca, cb) in enumerate(zip(a, b)):
        if ca != cb:
            return i
    return min(len(a), len(b))

# this one is not-thread safe
# it would perform just two operations:
#  - update current (based on new input)
#  - perform exploration step
# all concurrency between background exploration and new inputs from the client 
# would be handled externally
class LinearSpeculator:
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
        self.tokens = [[]]

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
                self.tokens = [request['tokens'][:]]
                self.cache = [None for _ in self.model.layers]
                self.cache_len = 0
            else:
                # we need to compare self.prompt + self.generated to request['tokens']
                # and strip generated/cache if needed
                new_tokens = request['tokens'][:]
                
                match_len = _longest_prefix(self.tokens[0], new_tokens)

                # if everything matched. We can keep all generated + cache, 
                # even what we have generated after the new prompt. Otherwise:

                if match_len < len(new_tokens):

                    # and the generated tokens we need to update to all the passed tokens
                    self.tokens[0] = new_tokens

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
                    self.cache_len = match_len
                
    def gen_next(self):
        if self.session_id is None:
            return
        with ft.Timer("gen_next_latency") as _:
            with ft.Timer("inference_latency") as _:
                # need to find the input. It is a difference between 
                # processed/populated to cache and current tokens
                tokens_to_process = self.tokens[0][self.cache_len:]

                x = mx.array(tokens_to_process)[None]
                with ft.Timer("model_eval_latency") as _:
                    logits, local_cache = self.model(x, self.cache)
                    y = mx.argmax(logits[:, -1, :]).item()
                self.tokens[0].append(y)

            with ft.Timer("cache_update_latency") as _:
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