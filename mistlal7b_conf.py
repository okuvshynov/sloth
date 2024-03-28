from dataclasses import dataclass
import torch

@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    sliding_window: int
    norm_eps: float
    vocab_size: int
    max_batch_size: int = 0
    compute_dtype: torch.dtype = torch.float16
    frozen_dtype: torch.dtype = torch.float16
    served_model_path: str = '' # relative path by default
    cached_data_path: str = ''  # relative path by default
    init_frozen: bool = True