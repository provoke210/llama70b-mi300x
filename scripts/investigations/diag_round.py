import torch, triton, triton.language as tl
@triton.jit
def k(x_ptr, o_ptr, n, MODE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i, mask=i < n)
    if MODE == 0:
        y = x.to(o_ptr.dtype.element_ty)
    elif MODE == 1:
        y = x.to(o_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
    else:
        y = x.to(o_ptr.dtype.element_ty, fp_downcast_rounding="rtz")
    tl.store(o_ptr + i, y, mask=i < n)
dt = torch.float8_e4m3fnuz
vals = torch.tensor([12.5, 12.499999, 100.0, 99.999992, 1.5625, -12.5, 13.5, 0.3, 200.7], device="cuda")
x = torch.cat([vals, torch.randn(1 << 20, device="cuda") * 60]).clamp(-240, 240).contiguous()
ref = x.to(dt)
for mode, name in [(0, "default"), (1, "rtne"), (2, "rtz")]:
    o = torch.empty_like(x, dtype=dt)
    try:
        k[(triton.cdiv(x.numel(), 1024),)](x, o, x.numel(), MODE=mode, BLOCK=1024)
        d = (o.view(torch.uint8) != ref.view(torch.uint8)).sum().item()
        print(f"{name:8s} diff_vs_torch={d:7d}  first:", o[:9].float().tolist())
    except Exception as e:
        print(name, "ERROR", type(e).__name__, str(e)[:300])
print("torch          first:", ref[:9].float().tolist())
