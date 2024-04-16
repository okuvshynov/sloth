import logging
import time

import mlx.core as mx

import fewlines.timer as ft
import fewlines.dashboard as fd

def common_prefix_len(A, B):
    res = 0
    for a, b in zip(A, B):
        if a != b:
            break
        res += 1
    return res

def gen_speculative(model, tokenizer, prefix, next_suffixes_fn, max_tokens=256):
    generated = []
    x = mx.array(prefix)[None]

    # Cache is 6D: [layer, k|v, batch_index, head_index, position, data_index]
    logits, cache = model(x)
    y = mx.argmax(logits[:, -1, :]).item()
    logging.info(f'after prefix :{y}')
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
            mx.eval(logits)

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

        logging.info(f'next approved sequence: {best}')
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
        
    logging.info(tokenizer.decode(generated))
    logging.info(f"TPS: {len(generated) / (time.time() - started)}")
    for l in fd.dashboard({"charts" : [[("*latency", 'histogram')]], 'color': 'green', 'n_lines': 3}):
        logging.info(l)