import sys; sys.path.insert(0, "/shared-docker/llama70b-mi300x")
import torch
from mi300_llm import kernels
from mi300_llm.fp8 import compute_fp8_dtype
dt, bf = compute_fp8_dtype(), torch.bfloat16
torch.manual_seed(0)
M = 4096
x, r = torch.randn(M, 8192, device="cuda", dtype=bf) * 3, torch.randn(M, 8192, device="cuda", dtype=bf) * 3
w = torch.randn(8192, device="cuda", dtype=bf)
q, s, res = kernels.fused_add_rmsnorm_quant(x, r, w, 1e-5, dt)
y, _ = kernels.fused_add_rmsnorm(x, r, w, 1e-5)
qu, su = kernels.quantize_per_token(y, dt)
bad = (q.view(torch.uint8) != qu.view(torch.uint8)).nonzero().tolist()
h = res.double()
rstd64 = torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5)
for a, b in bad:
    y64 = h[a, b] * rstd64[a, 0] * w[a*0 + b].double()
    print(f"row {a} col {b}: unfused bf16 y={y[a,b].item()!r}  y64={y64.item():.9f}  "
          f"y/scale: unfused {y[a,b].double().item()/su[a,0].item():.6f}  "
          f"codes fused={q[a,b].float().item()} unfused={qu[a,b].float().item()}")
    # Which bf16 value did the fused kernel quantize? Invert its code.
    print(f"      fused code*scale={q[a,b].float().item()*s[a,0].item():.6f}, unfused code*scale={qu[a,b].float().item()*su[a,0].item():.6f}")
    # neighbours of y in bf16
    yb = y[a, b]
    up = torch.nextafter(yb.float(), torch.tensor(1e9, device='cuda')).to(bf)
    print(f"      bf16 neighbours: {yb.item()} -> y/scale={yb.double().item()/su[a,0].item():.6f}")
