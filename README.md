# sloth

1. [done] Caching. Make sure it works on M7. Make more tests.
2. What should we use as simpler model? Maybe same mistral 7 quantized to 4bit.
2. Correctness tests on M7, M7x8
3. Performance study - use the typical qna workload - inputs of varying sizes - from 50 tokens to 5000 tokens, output typically ~500.
4. What should be the small model? - quantized same model? 4x smaller? 8x smaller?
5. Can we run Mixtral 8x22B on 192Gb M2 Ultra? At 8 bit maybe. At full 16 bit?


Relevant links.

1. https://github.com/ml-explore/mlx-examples/tree/main/llms/mixtral
2. https://github.com/FasterDecoding/Medusa?tab=readme-ov-file
3. https://arxiv.org/pdf/2401.18079.pdf
4. https://huggingface.co/Vezora/Mistral-22B-v0.1

