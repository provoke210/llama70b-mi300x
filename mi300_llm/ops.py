"""Backend dispatch: every op runs either the reference or the Triton kernel.

``set_backend("triton")`` switches all ops at once; ``override`` swaps a single
op, which is how per-kernel speedups are measured in isolation.
"""

from __future__ import annotations

from contextlib import contextmanager

from . import ref_ops

_OPS = (
    "fused_add_rmsnorm", "rmsnorm", "apply_rope", "fp8_linear", "fp8_gemm", "paged_decode_attention",
    "silu_mul", "rmsnorm_quant", "fused_add_rmsnorm_quant", "silu_mul_quant",
)
_backend = "ref"
_overrides: dict[str, str] = {}


def _triton_impl(name: str):
    from . import kernels  # imported lazily: Triton is not available everywhere

    return getattr(kernels, name)


def set_backend(name: str) -> None:
    global _backend
    assert name in ("ref", "triton"), name
    _backend = name


def get_backend() -> str:
    return _backend


@contextmanager
def override(**ops: str):
    """with ops.override(paged_decode_attention="triton"): ..."""
    old = dict(_overrides)
    _overrides.update(ops)
    try:
        yield
    finally:
        _overrides.clear()
        _overrides.update(old)


def _get(name: str):
    backend = _overrides.get(name, _backend)
    return _triton_impl(name) if backend == "triton" else getattr(ref_ops, name)


def fused_add_rmsnorm(*a, **k):
    return _get("fused_add_rmsnorm")(*a, **k)


def rmsnorm(*a, **k):
    return _get("rmsnorm")(*a, **k)


def apply_rope(*a, **k):
    return _get("apply_rope")(*a, **k)


def fp8_linear(*a, **k):
    return _get("fp8_linear")(*a, **k)


def paged_decode_attention(*a, **k):
    return _get("paged_decode_attention")(*a, **k)


def silu_mul(*a, **k):
    return _get("silu_mul")(*a, **k)


def fp8_gemm(*a, **k):
    return _get("fp8_gemm")(*a, **k)


def rmsnorm_quant(*a, **k):
    return _get("rmsnorm_quant")(*a, **k)


def fused_add_rmsnorm_quant(*a, **k):
    return _get("fused_add_rmsnorm_quant")(*a, **k)


def silu_mul_quant(*a, **k):
    return _get("silu_mul_quant")(*a, **k)


# Prefill attention stays on the reference path for now; decode is the focus.
prefill_attention = ref_ops.prefill_attention
