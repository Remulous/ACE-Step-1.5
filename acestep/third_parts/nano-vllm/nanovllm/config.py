import os
from dataclasses import dataclass
from transformers import AutoConfig


def select_kv_cache_blocks(available_blocks: int, max_kv_cache_blocks: int) -> int:
    """Select the initialization-time KV pool size without resizing live storage."""
    if max_kv_cache_blocks < 0:
        raise ValueError("max_kv_cache_blocks must be zero (automatic) or positive")
    if max_kv_cache_blocks == 0:
        return max(1, available_blocks)
    if available_blocks < 1:
        raise RuntimeError("Insufficient GPU memory for even one KV cache block")
    return min(available_blocks, max_kv_cache_blocks)


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    max_kvcache_blocks: int = 0

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if self.max_kvcache_blocks < 0:
            raise ValueError("max_kvcache_blocks must be zero (automatic) or positive")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
