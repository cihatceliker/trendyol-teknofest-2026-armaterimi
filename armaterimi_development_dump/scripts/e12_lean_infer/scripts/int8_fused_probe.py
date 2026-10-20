#!/usr/bin/env python
"""Can int8's 1.29x raw-GEMM win survive the quantisation it requires?

Raw `torch._int_mm` measured 1.29x over bf16 at our shapes on an idle A100. But a usable int8
linear is: row-wise absmax -> scale -> round/clamp -> cast int8 -> int_mm -> rescale the int32
accumulator by (activation scale x per-channel weight scale). In eager that is five or six full
passes over a 28160x768 tensor with fp32 intermediates LARGER than the bf16 input, and that
traffic is exactly what can eat the 1.29x. Fused into one kernel it plausibly survives.

So this measures four things at our real shapes:
  1. bf16                      -- the baseline we ship today
  2. raw _int_mm               -- the 1.29x ceiling, ignoring quantisation
  3. eager quant + mm + scale  -- the honest cost written naively
  4. torch.compile'd (3)       -- the one that decides it

Decision rule stated UP FRONT so the result cannot be rationalised afterwards:
  >= 1.15x on the compiled variant -> implement in LeanBert/LeanGTE and verify DEV F1
  <  1.15x                         -> int8 is genuinely dead at these shapes, and it is measured
"""
import os, sys, time
import torch

DEV = "cuda"
SHAPES = [("qkv", 28160, 768, 2304), ("attn_out", 28160, 768, 768),
          ("mlp_up", 28160, 768, 3072), ("mlp_down", 28160, 3072, 768)]
INFER_S, TOTAL_S, BASE_SPS = 20.5, 34.0, 2652     # champion, measured in-container

def bench(f, n=40, warm=10):
    for _ in range(warm): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1000

def q_linear(x, Wq, ws):
    s = x.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-8) / 127.0
    xq = (x.float() / s).round().clamp(-127, 127).to(torch.int8)
    return (torch._int_mm(xq, Wq).float() * s * ws).to(torch.bfloat16)

q_linear_c = torch.compile(q_linear, dynamic=False)

def main():
    free, total = torch.cuda.mem_get_info()
    print(f"GPU {torch.cuda.get_device_name(0)}  free {free/2**30:.1f}/{total/2**30:.1f} GiB")
    print(f"{'gemm':<10}{'bf16':>9}{'raw i8':>9}{'eager':>9}{'compiled':>10}"
          f"{'raw x':>8}{'eager x':>9}{'comp x':>8}")
    tots = dict(bf=0.0, raw=0.0, eag=0.0, cmp=0.0)
    for name, M, K, N in SHAPES:
        x  = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
        W  = torch.randn(K, N, device=DEV, dtype=torch.bfloat16)
        Wq = torch.randint(-127, 127, (K, N), device=DEV, dtype=torch.int8)
        ws = (torch.rand(N, device=DEV, dtype=torch.float32) + 0.5)
        xq = torch.randint(-127, 127, (M, K), device=DEV, dtype=torch.int8)
        t_bf  = bench(lambda: x @ W)
        t_raw = bench(lambda: torch._int_mm(xq, Wq))
        t_eag = bench(lambda: q_linear(x, Wq, ws))
        try:
            q_linear_c(x, Wq, ws); torch.cuda.synchronize()
            t_cmp = bench(lambda: q_linear_c(x, Wq, ws))
        except Exception as e:
            print(f"  compile failed on {name}: {type(e).__name__}: {str(e)[:70]}"); t_cmp = float("nan")
        # accuracy sanity: how far does int8 move the output?
        ref = (x.float() @ (Wq.float() * ws))
        got = q_linear(x, Wq, ws).float()
        rel = ((got - ref).norm() / ref.norm()).item()
        tots["bf"] += t_bf; tots["raw"] += t_raw; tots["eag"] += t_eag
        if t_cmp == t_cmp: tots["cmp"] += t_cmp
        print(f"  {name:<8}{t_bf:9.3f}{t_raw:9.3f}{t_eag:9.3f}{t_cmp:10.3f}"
              f"{t_bf/t_raw:8.2f}{t_bf/t_eag:9.2f}{t_bf/t_cmp:8.2f}   relerr {rel:.4f}")
    print(f"\n  layer GEMM total  bf16 {tots['bf']:.2f} ms | raw {tots['raw']:.2f} "
          f"({tots['bf']/tots['raw']:.2f}x) | eager {tots['eag']:.2f} ({tots['bf']/tots['eag']:.2f}x) "
          f"| compiled {tots['cmp']:.2f} ({tots['bf']/tots['cmp']:.2f}x)")
    sp = tots["bf"] / tots["cmp"] if tots["cmp"] else 0.0
    print(f"\n  end-to-end projection (champion: {INFER_S}s infer of a {TOTAL_S}s run, {BASE_SPS} SPS)")
    for share in (0.7, 0.8):
        inf = INFER_S * ((1 - share) + share / max(sp, 1e-9))
        tot = TOTAL_S - (INFER_S - inf)
        print(f"    GEMMs={share:.0%}: infer {inf:5.1f}s  total {tot:5.1f}s  "
              f"{90000/tot:6.0f} SPS  ({100*((90000/tot)/BASE_SPS-1):+.0f}%)")
    print(f"\n  VERDICT: compiled speedup {sp:.2f}x -> "
          f"{'IMPLEMENT (>=1.15x)' if sp >= 1.15 else 'DEAD at these shapes (<1.15x), measured not assumed'}")

if __name__ == "__main__":
    main()
