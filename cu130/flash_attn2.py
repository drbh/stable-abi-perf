# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "kernels",
#     "numpy",
#     "torch==2.11",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu130" }
# ///

# Regular (v2) vs stable-ABI (v3) build of kernels-community/flash-attn2 (transformers loads v3),
# on the calls transformers makes with Llama-3.1-8B attention (32 q heads, 8 kv heads, head_dim 128), bf16:
#   - generate(): flash_attn_func for prefill, and for decode (q_len=1 against the full KV cache).
#   - continuous batching: flash_attn_varlen_func over packed requests.

import random
import statistics
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from kernels import get_local_kernel


def load(repo, rev, variant):
    # Download only this variant, so get_local_kernel can only pick it.
    path = snapshot_download(
        repo,
        repo_type="kernel",
        revision=rev,
        allow_patterns=[f"build/{variant}/*"],
        local_dir=Path(__file__).parent / (repo.split("/")[1] + "-builds") / variant,
    )
    return get_local_kernel(Path(path), trust_remote_code=True)


def time_loop(fn, iters):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def header():
    print(
        f"\ntorch {torch.__version__}, {torch.cuda.get_device_name()}  (wall time, median us/call)"
    )
    print(f"{'workload':>48} | {'regular':>9} {'stable':>9} {'delta':>18}")


def row(label, reg_fn, stb_fn, reps=21, target_s=0.02):
    # Back-to-back calls, regular and stable alternated in random order each repetition.
    for fn in (reg_fn, stb_fn):  # warm up ~0.2 s each
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 0.2:
            fn()
    iters = max(10, int(target_s / time_loop(reg_fn, 10) * 1e6))
    times = {reg_fn: [], stb_fn: []}
    for _ in range(reps):
        for fn in random.sample([reg_fn, stb_fn], 2):
            times[fn].append(time_loop(fn, iters))
    reg, stb = statistics.median(times[reg_fn]), statistics.median(times[stb_fn])
    print(
        f"{label:>48} | {reg:9.2f} {stb:9.2f} {stb - reg:+8.2f} ({100 * (stb - reg) / reg:+5.1f}%)"
    )


REGULAR = load(
    "kernels-community/flash-attn2", "v2", "torch211-cxx11-cu130-x86_64-linux"
)
STABLE = load(
    "kernels-community/flash-attn2", "v3", "torch-stable-abi210-cu130-x86_64-linux"
)
header()

torch.manual_seed(0)
HEADS, KV_HEADS, HEAD_DIM, DTYPE = 32, 8, 128, torch.bfloat16


def qkv(b, q_len, kv_len):
    return (
        torch.randn(b, q_len, HEADS, HEAD_DIM, device="cuda", dtype=DTYPE),
        torch.randn(b, kv_len, KV_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE),
        torch.randn(b, kv_len, KV_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE),
    )


def check_and_row(label, reg, stb):
    assert torch.equal(reg(), stb()), "outputs differ"
    row(label, reg, stb)


# generate(): prefill one prompt.
for s in [512, 2048, 8192]:
    q, k, v = qkv(1, s, s)
    check_and_row(
        f"flash_attn_func prefill bs=1 prompt={s}",
        lambda: REGULAR.flash_attn_func(q, k, v, causal=True),
        lambda: STABLE.flash_attn_func(q, k, v, causal=True),
    )

# generate(): decode step, one new token against the cache.
for b, ctx in [(1, 1024), (1, 4096), (8, 2048)]:
    q, k, v = qkv(b, 1, ctx)
    check_and_row(
        f"flash_attn_func decode bs={b} ctx={ctx}",
        lambda: REGULAR.flash_attn_func(q, k, v, causal=True),
        lambda: STABLE.flash_attn_func(q, k, v, causal=True),
    )

# Continuous batching: packed prefill of several prompts, and packed decode of many requests.
for label, q_lens, kv_lens in [
    ("varlen prefill 8 prompts x 512", [512] * 8, [512] * 8),
    ("varlen decode 32 reqs ctx~1k", [1] * 32, [1024 + 32 * i for i in range(32)]),
    ("varlen decode 128 reqs ctx~1k", [1] * 128, [768 + 4 * i for i in range(128)]),
]:
    cu_q = torch.tensor([0] + q_lens, device="cuda", dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    cu_k = torch.tensor([0] + kv_lens, device="cuda", dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    q = torch.randn(sum(q_lens), HEADS, HEAD_DIM, device="cuda", dtype=DTYPE)
    k = torch.randn(sum(kv_lens), KV_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE)
    v = torch.randn_like(k)
    args = (q, k, v, cu_q, cu_k, max(q_lens), max(kv_lens))
    check_and_row(
        f"flash_attn_{label}",
        lambda: REGULAR.flash_attn_varlen_func(*args, causal=True),
        lambda: STABLE.flash_attn_varlen_func(*args, causal=True),
    )
