import torch, triton, triton.language as tl
@triton.jit
def k(x_ptr, o_store_ptr, o_reg_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    x = tl.load(x_ptr + i, mask=m)
    tl.store(o_store_ptr + i, x.to(tl.bfloat16), mask=m)           # convert at store
    tl.store(o_reg_ptr + i, x.to(tl.bfloat16).to(tl.float32), mask=m)  # convert in registers
torch.manual_seed(0)
# exact bf16 midpoints: bf16 value + half ulp (set bit 15 of the fp32 pattern)
b = (torch.randn(1 << 22, device="cuda") * 3).to(torch.bfloat16).float()
mid = (b.view(torch.int32) | 0x8000).view(torch.float32)
rnd = torch.randn(1 << 22, device="cuda") * 3
for name, x in [("exact midpoints", mid), ("random fp32", rnd)]:
    st = torch.empty_like(x, dtype=torch.bfloat16); rg = torch.empty_like(x)
    k[(triton.cdiv(x.numel(), 1024),)](x, st, rg, x.numel(), BLOCK=1024)
    ref = x.to(torch.bfloat16)  # torch: round-to-nearest-even
    print(f"{name:16s} store-path != torch: {(st != ref).sum().item():8d}   register-path != torch: {(rg != ref.float()).sum().item():8d}   of {x.numel()}")
    if name == "exact midpoints":
        d = (rg != ref.float()).nonzero()[:3].flatten().tolist()
        for j in d:
            print(f"   x={x[j].item()!r} torch={ref[j].item()!r} store={st[j].item()!r} reg={rg[j].item()!r}")
