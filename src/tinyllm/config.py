"""Model configuration and named presets (docs/ARCHITECTURE.md)."""

from dataclasses import asdict, dataclass, field, replace


@dataclass(frozen=True)
class ModelConfig:
    # Sizes
    vocab_size: int = 8192
    max_seq_len: int = 512
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int | None = None   # None = n_heads (plain MHA); fewer = GQA (Phase 4.9a)
    ffn_hidden: int = 1408
    # Numerics
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    init_std: float = 0.02
    tie_embeddings: bool = True
    # Attention variant per layer, e.g. ("gdn", "gdn", "gdn", "mha") * 2 for a hybrid.
    # None = all "mha".
    layer_types: tuple[str, ...] | None = field(default=None)

    def __post_init__(self) -> None:
        # frozen dataclass: fill defaults through object.__setattr__
        if self.n_kv_heads is None:
            object.__setattr__(self, "n_kv_heads", self.n_heads)
        if self.layer_types is None:
            object.__setattr__(self, "layer_types", ("mha",) * self.n_layers)
        else:
            object.__setattr__(self, "layer_types", tuple(self.layer_types))

        if self.d_model % self.n_heads:
            raise ValueError(f"d_model {self.d_model} not divisible by n_heads {self.n_heads}")
        if self.n_heads % self.n_kv_heads:
            raise ValueError(f"n_heads {self.n_heads} not divisible by n_kv_heads {self.n_kv_heads}")
        if self.head_dim % 2:
            raise ValueError(f"head_dim {self.head_dim} must be even for RoPE")
        if len(self.layer_types) != self.n_layers:
            raise ValueError(f"{len(self.layer_types)} layer_types for {self.n_layers} layers")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def estimate_params(self) -> int:
        """Parameter count for softmax-attention layers (mha / gqa).

        embeddings V·d (tied with the LM head)
        + per layer: q,o 2·d² + k,v 2·d·kv_dim + MLP 3·d·ffn + 2 RMSNorms 2·d
        + final RMSNorm d
        With n_kv_heads == n_heads this is the ARCHITECTURE.md formula
        V·d + L·(4·d² + 3·d·ffn) + norms.
        """
        d, kv_dim = self.d_model, self.n_kv_heads * self.head_dim
        per_layer = 2 * d * d + 2 * d * kv_dim + 3 * d * self.ffn_hidden + 2 * d
        emb = self.vocab_size * d * (1 if self.tie_embeddings else 2)
        return emb + self.n_layers * per_layer + d

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**d)


PRESETS: dict[str, ModelConfig] = {
    "smoke": ModelConfig(d_model=384, n_layers=6, n_heads=6, ffn_hidden=1024),
    "main": ModelConfig(d_model=512, n_layers=8, n_heads=8, ffn_hidden=1408),
}


def get_config(name: str, **overrides) -> ModelConfig:
    """A preset with optional overrides, e.g. get_config("main", n_kv_heads=2)."""
    cfg = PRESETS[name]
    if overrides:
        if "n_layers" in overrides and "layer_types" not in overrides:
            overrides["layer_types"] = None  # re-derive for the new depth
        cfg = replace(cfg, **overrides)
    return cfg
