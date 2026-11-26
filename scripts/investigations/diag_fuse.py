import sys; sys.path.insert(0, "/shared-docker/llama70b-mi300x")
import torch
from mi300_llm import kernels
from mi300_llm.fp8 import compute_fp8_dtype
dt, bf = compute_fp8_dtype(), torch.bfloat16
torch.manual_seed(0)
def cmp(name, a, b):
    (qa, sa), (qb, sb) = a, b
    srows = (sa != sb).view(-1)
    codes = (qa.view(torch.uint8) != qb.view(torch.uint8))
    print(f"{name:28s} rows with different scale: {srows.sum().item():5d}/{srows.numel()}  "
          f"rows with any different code: {codes.any(-1).sum().item():5d}  codes: {codes.sum().item()}")
M = 16384
x, r = torch.randn(M, 8192, device="cuda", dtype=bf) * 3, torch.randn(M, 8192, device="cuda", dtype=bf) * 3
w = torch.randn(8192, device="cuda", dtype=bf)
q, s, res = kernels.fused_add_rmsnorm_quant(x, r, w, 1e-5, dt)
y, res2 = kernels.fused_add_rmsnorm(x, r, w, 1e-5)
print("residual equal:", torch.equal(res, res2))
cmp("add_rmsnorm_quant", (q, s), kernels.quantize_per_token(y, dt))
gu = torch.randn(M, 2 * 28672, device="cuda", dtype=bf) * 2
cmp("silu_mul_quant", kernels.silu_mul_quant(gu, dt), kernels.quantize_per_token(kernels.silu_mul(gu), dt))
# isolate silu: compare the bf16 values each path quantizes
y_unf = kernels.silu_mul(gu)
g, u = gu.float().chunk(2, -1)
for name, yy in [("torch g*sigmoid(g)*u", (g * torch.sigmoid(g) * u).to(bf)), ("torch silu(g)*u", (torch.nn.functional.silu(g) * u).to(bf))]:
    print(f"  unfused triton silu vs {name}: {(yy != y_unf).sum().item()} of {yy.numel()} differ")
