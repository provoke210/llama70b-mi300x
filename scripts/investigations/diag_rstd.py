import torch, triton, triton.language as tl
@triton.jit
def k(x_ptr, out_ptr, rstd_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * N + cols).to(tl.float32)
    rstd = tl.rsqrt(tl.div_rn(tl.sum(x * x, axis=0), N.to(tl.float32)) + 1e-5)
    tl.store(rstd_ptr + row, rstd)
    tl.store(out_ptr + row * N + cols, (x * rstd).to(out_ptr.dtype.element_ty))  # only the out dtype differs
torch.manual_seed(0)
M, N = 16384, 8192
x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16) * 3
r = {}
for dt in (torch.bfloat16, torch.float8_e4m3fnuz, torch.float32):
    o = torch.empty(M, N, device="cuda", dtype=dt); rs = torch.empty(M, device="cuda")
    h = k[(M,)](x, o, rs, N, BLOCK=N, num_warps=8)
    r[dt] = rs
    lay = [l for l in h.asm["ttgir"].splitlines() if "#blocked" in l and "=" in l and "#ttg.blocked" in l][:2]
    print(dt, "layout:", [l.split("<")[1][:70] for l in lay])
print("rstd differs bf16-out vs fp8-out:", (r[torch.bfloat16] != r[torch.float8_e4m3fnuz]).sum().item(), "of", M,
      "| bf16 vs fp32-out:", (r[torch.bfloat16] != r[torch.float32]).sum().item())
