"""Attention slot used by DiT blocks.

The kernels themselves live in ``kandinsky.runtime.kernels``.
"""

from kandinsky.runtime.kernels.attention_engine import SelfAttentionEngine, _sdpa, resolve_attention_engine

__all__ = ("SelfAttentionEngine", "_sdpa", "resolve_attention_engine")
