import sys; sys.path.insert(0, "/shared-docker/llama70b-mi300x")
import torch
from mi300_llm import kernels, ref_ops
from mi300_llm.fp8 import compute_fp8_dtype, quantize_per_channel, quantize_per_token
torch.manual_seed(0)
dt = compute_fp8_dtype(); print("fp8 dtype", dt)
M, N, K = 40, 8192, 28672
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
wq, ws = quantize_per_channel(torch.randn(N, K, device="cuda") / K**0.5, dt); ws = ws.reshape(-1)
# 1) activation quantization: triton vs torch
tq, ts = kernels.quantize_per_token(x, dt)
rq, rs = quantize_per_token(x, dt)
print("scale max rel diff", ((ts - rs).abs() / rs).max().item())
neq = (tq.view(torch.uint8) != rq.view(torch.uint8))
print("quantized codes differing:", neq.sum().item(), "of", neq.numel())
if neq.any():
    i = neq.nonzero()[:5]
    for a, b in i.tolist():
        v = (x[a, b].float() / rs[a, 0]).item()
        print(f"  x/scale={v:.6f} triton={tq[a,b].float().item()} torch={rq[a,b].float().item()}")
# 2) GEMM alone on identical quantized inputs
exact = (rq.float() @ wq.float().t()) * rs * ws.view(1, -1)
g = kernels.fp8_gemm(rq, rs, wq, ws, torch.float32)
print("gemm fp32-out max abs diff vs fp32 ref", (g - exact).abs().max().item(), " |y| max", exact.abs().max().item())
gb = kernels.fp8_gemm(rq, rs, wq, ws, torch.bfloat16)
print("gemm bf16-out max abs diff", (gb.float() - exact.to(torch.bfloat16).float()).abs().max().item())
