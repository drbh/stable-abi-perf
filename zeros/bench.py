# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "kernels",
#     "numpy",
#     "torch==2.11",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
# ///

"""torch::zeros (classic ABI) vs torch::stable::new_zeros (stable ABI), nothing else."""

import statistics, time
from pathlib import Path
import torch
from kernels import get_local_kernel

HERE = Path(__file__).parent

C = get_local_kernel((HERE / "zeros-classic/result").resolve(), trust_remote_code=True)
S = get_local_kernel((HERE / "zeros-stable/result").resolve(), trust_remote_code=True)


def us_per_call(f, x, n, iters, reps=15):
    for _ in range(100):
        f(x, n)
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        for _ in range(iters):
            f(x, n)
        torch.cuda.synchronize()
        out.append((time.perf_counter_ns() - t0) / iters / 1e3)
    return out


print(f"torch {torch.__version__}, {torch.cuda.get_device_name()}")
print("median us/call over 15 reps; per-alloc = (zeros(n) - noop) / n\n")
print(
    f"{'numel':>9} {'n':>4} | {'classic':>9} {'stable':>9} | {'cls/alloc':>9} {'stb/alloc':>9} {'delta/alloc':>11} {'%':>6}"
)
for numel in [64, 4096, 1 << 20]:
    x = torch.zeros(numel, device="cuda")
    for n in [1, 3, 32]:
        iters = max(200, 20000 // n) if numel < (1 << 20) else 2000 // n + 50
        res = {}
        # interleave classic/stable a few times
        for _ in range(3):
            for name, mod in [("classic", C), ("stable", S)]:
                res.setdefault(name + "_noop", []).extend(
                    us_per_call(mod.noop, x, n, iters, 5)
                )
                res.setdefault(name, []).extend(us_per_call(mod.zeros, x, n, iters, 5))
        m = {k: statistics.median(v) for k, v in res.items()}
        ca = (m["classic"] - m["classic_noop"]) / n
        sa = (m["stable"] - m["stable_noop"]) / n
        print(
            f"{numel:>9} {n:>4} | {m['classic']:9.3f} {m['stable']:9.3f} | {ca:9.3f} {sa:9.3f} {sa - ca:11.3f} {100 * (sa - ca) / ca:5.1f}%"
        )
    print(
        f"{'':>9} noop | {m['classic_noop']:9.3f} {m['stable_noop']:9.3f}   (op-call overhead, no alloc)"
    )
