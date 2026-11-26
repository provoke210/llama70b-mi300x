"""FP8 formats and quantization helpers.

Checkpoints (and NVIDIA GPUs) use OCP ``float8_e4m3fn``: exponent bias 7, max 448.
MI300X (gfx942) computes in ``float8_e4m3fnuz``: exponent bias 8, max 240, and a
single NaN encoding (0x80) instead of negative zero.

The same bit pattern therefore means half the value in fnuz. Converting a
checkpoint is a reinterpretation plus doubling the scale -- no requantization,
so it is lossless except for the -0 encoding, which becomes +0.
"""

from __future__ import annotations

import torch

E4M3FN = torch.float8_e4m3fn
E4M3FNUZ = torch.float8_e4m3fnuz


def is_gfx942() -> bool:
    """True on MI300-series (gfx94x) GPUs, which need the fnuz format."""
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    arch = torch.cuda.get_device_properties(0).gcnArchName
    return arch.startswith("gfx94")


def compute_fp8_dtype() -> torch.dtype:
    return E4M3FNUZ if is_gfx942() else E4M3FN


def fp8_max(dtype: torch.dtype) -> float:
    return float(torch.finfo(dtype).max)


def e4m3fn_to_e4m3fnuz(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reinterpret e4m3fn weights as e4m3fnuz and double the scale to compensate."""
    assert weight.dtype == E4M3FN
    bits = weight.view(torch.int8).clone()
    # 0x80 is -0.0 in e4m3fn but NaN in e4m3fnuz.
    bits[bits == -128] = 0
    return bits.view(E4M3FNUZ), scale * 2.0


def quantize_per_token(
    x: torch.Tensor, dtype: torch.dtype, eps: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic symmetric per-row quantization. Returns (x_fp8, scale[M, 1] fp32)."""
    x32 = x.float()
    amax = x32.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    # Divide by a tensor, not a Python float: on ROCm GPUs, tensor / scalar
    # is not correctly rounded (it is off by 1 ulp for some inputs, which
    # flips ~0.08% of FP8 codes at rounding midpoints). tensor / tensor is.
    scale = amax / torch.full_like(amax, fp8_max(dtype))
    q = (x32 / scale).clamp(-fp8_max(dtype), fp8_max(dtype)).to(dtype)
    return q, scale


def quantize_per_channel(
    w: torch.Tensor, dtype: torch.dtype, eps: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    """Static symmetric per-output-channel quantization of a [N, K] weight."""
    return quantize_per_token(w, dtype, eps)
