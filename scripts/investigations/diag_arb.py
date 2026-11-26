import sys; sys.path.insert(0, "/shared-docker/llama70b-mi300x")
import torch
from mi300_llm import kernels
from mi300_llm.fp8 import quantize_per_token
torch.manual_seed(0)
dt = torch.float8_e4m3fnuz
x = torch.randn(40, 28672, device="cuda", dtype=torch.bfloat16)
tq, ts = kernels.quantize_per_token(x, dt)
rq, rs = quantize_per_token(x, dt)
amax = x.float().abs().amax(-1, keepdim=True)
# Correctly rounded fp32 scale: compute in fp64, round once to fp32.
exact_s = (amax.double() / 240.0).float()
cpu_s = (amax.cpu() / 240.0)
print("scale  triton==exact:", (ts == exact_s).all().item(), " torch_gpu==exact:", (rs == exact_s).all().item(), " torch_cpu==exact:", (cpu_s.cuda() == exact_s).all().item())
# Correctly rounded quantization from the exact scale, in fp64 then one fp8 rounding.
exact_q = (x.double() / exact_s.double()).clamp(-240, 240).to(dt)   # fp64 -> fp8 direct
d = lambda a, b: (a.view(torch.uint8) != b.view(torch.uint8)).sum().item()
print("codes  triton vs exact:", d(tq, exact_q), "  torch_gpu vs exact:", d(rq, exact_q))
# Where torch's fp32 x/scale lands
q32 = x.float() / rs
q32_exact = (x.double() / rs.double()).float()
print("torch gpu fp32 division correctly rounded?", (q32 == q32_exact).all().item(), " mismatches:", (q32 != q32_exact).sum().item())
