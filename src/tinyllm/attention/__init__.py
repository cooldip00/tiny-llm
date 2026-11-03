"""Attention variants. Importing this package registers every built-in variant."""

from tinyllm.attention.base import ATTENTION, Attention, build_attention, register
from tinyllm.attention import mha  # noqa: F401  (registers "mha")

__all__ = ["ATTENTION", "Attention", "build_attention", "register"]
