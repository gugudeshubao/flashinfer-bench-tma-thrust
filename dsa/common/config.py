from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DeepSeekDSAConfig:
    """DeepSeek V3.2 DSA defaults taken from the public 671B config."""

    dim: int = 7168
    n_heads: int = 128
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def softmax_scale(self) -> float:
        return 1.0 / math.sqrt(self.qk_head_dim)

    @property
    def index_scale(self) -> float:
        return 1.0 / math.sqrt(self.index_head_dim)
