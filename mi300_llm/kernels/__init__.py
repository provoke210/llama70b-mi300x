import os

from .decode_attention import paged_decode_attention

if os.environ.get("MI300_ATTN") == "v1":  # A/B against the first decode-attention kernel
    from .decode_attention_v1 import paged_decode_attention  # noqa: F811
from .fp8_gemm import fp8_gemm, fp8_linear
from .fused_quant import fused_add_rmsnorm_quant, quantize_per_token, rmsnorm_quant, silu_mul_quant
from .norm_rope import apply_rope, fused_add_rmsnorm, rmsnorm, silu_mul
from .prefill_attention import prefill_attention

__all__ = [
    "apply_rope",
    "fp8_gemm",
    "fp8_linear",
    "fused_add_rmsnorm",
    "fused_add_rmsnorm_quant",
    "paged_decode_attention",
    "prefill_attention",
    "quantize_per_token",
    "rmsnorm",
    "rmsnorm_quant",
    "silu_mul",
    "silu_mul_quant",
]
