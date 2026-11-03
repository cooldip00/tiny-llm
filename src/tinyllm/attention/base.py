"""The swappable attention contract and the registry (docs/ARCHITECTURE.md).

Every variant (mha, gqa, mla, gdn) subclasses `Attention`, registers itself under
a name, and is picked per layer through `ModelConfig.layer_types`. The rest of
the model never needs to know which one it got.
"""

from torch import Tensor, nn

from tinyllm.config import ModelConfig

ATTENTION: dict[str, type["Attention"]] = {}


def register(name: str):
    """Class decorator: @register("mha") adds the class to ATTENTION."""
    def deco(cls: type["Attention"]) -> type["Attention"]:
        if name in ATTENTION:
            raise ValueError(f"attention variant {name!r} already registered")
        ATTENTION[name] = cls
        return cls
    return deco


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor], cache=None) -> Tensor:
        """x: (B, T, d_model) -> (B, T, d_model).

        rope: (cos, sin) for the T positions in x.
        cache: per-layer state for incremental decoding. None during training.
        """
        raise NotImplementedError

    def cache_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """Decode-time state this layer adds per generated token (bf16 = 2 bytes)."""
        raise NotImplementedError


def build_attention(cfg: ModelConfig, layer_idx: int) -> Attention:
    name = cfg.layer_types[layer_idx]
    if name not in ATTENTION:
        raise KeyError(f"unknown attention {name!r}; registered: {sorted(ATTENTION)}")
    return ATTENTION[name](cfg, layer_idx)
