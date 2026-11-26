import sys; sys.path.insert(0, "/shared-docker/llama70b-mi300x")
import torch, triton
from mi300_llm import kernels
HQ, HKV, D, page = 64, 8, 128, 16
for b in (1, 8, 32):
    for ctx in (1024, 4096, 16384):
        nblk = ctx // page
        k = torch.randn(b * nblk, page, HKV, D, device="cuda", dtype=torch.bfloat16); v = torch.randn_like(k)
        tables = torch.arange(b * nblk, device="cuda", dtype=torch.int32).view(b, nblk)
        lens = torch.full((b,), ctx, device="cuda", dtype=torch.int32)
        q = torch.randn(b, HQ, D, device="cuda", dtype=torch.bfloat16)
        nbytes = 2 * b * ctx * HKV * D * 2
        res = []
        for bn in (32, 64, 128):
            for s in (1, 2, 4, 8, 16, 32, 64, 128):
                if s * bn > ctx: continue
                t = triton.testing.do_bench(lambda: kernels.paged_decode_attention(q, k, v, tables, lens, D**-0.5, num_splits=s, block_n=bn), warmup=10, rep=40)
                res.append((t, s, bn))
        cur = triton.testing.do_bench(lambda: kernels.paged_decode_attention(q, k, v, tables, lens, D**-0.5), warmup=10, rep=40)
        t, s, bn = min(res)
        print(f"B={b:<3} ctx={ctx:<6} current {cur*1e3:7.1f} us | best {t*1e3:7.1f} us splits={s:<3} block_n={bn:<4} programs={b*HKV*s:<5} {nbytes/(t*1e-3)/1e12:.2f} TB/s")
