"""
Qwen3 Model Configuration.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Qwen3Config:
    """Configuration class for Qwen3 model."""
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_theta: float = 1000000.0
    rope_scaling: Optional[dict] = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151643
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "Qwen3Config":
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_json(cls, path: str) -> "Qwen3Config":
        import json
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
