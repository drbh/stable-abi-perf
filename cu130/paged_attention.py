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

# Regular (v1) vs stable-ABI (v2) build of kernels-community/paged-attention, on LLM decode
# shapes: Llama-3.1-8B attention (32 q heads, 8 kv heads, head_dim 128), bf16, block_size 16.
# v1/v2 picked with vLLM's heuristic (partition 512).

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
    "kernels-community/paged-attention", "v1", "torch211-cxx11-cu130-x86_64-linux"
)
STABLE = load(
    "kernels-community/paged-attention", "v2", "torch-stable-abi210-cu130-x86_64-linux"
)
header()

torch.manual_seed(0)
HEADS, KV_HEADS, HEAD_DIM, BLOCK, DTYPE = 32, 8, 128, 16, torch.bfloat16
X = 16 // torch.tensor([], dtype=DTYPE).element_size()
SCALE, PARTITION = HEAD_DIM**-0.5, 512
one = torch.tensor(1.0, device="cuda")  # k_scale / v_scale


def kv_cache(num_blocks, zero=False):
    make = torch.zeros if zero else torch.randn
    return (
        make(num_blocks, KV_HEADS, HEAD_DIM // X, BLOCK, X, device="cuda", dtype=DTYPE),
        make(num_blocks, KV_HEADS, HEAD_DIM, BLOCK, device="cuda", dtype=DTYPE),
    )


# Decode: batch of sequences, one query token each.
for batch, ctx in [(1, 1024), (1, 4096), (32, 2048), (128, 1024)]:
    blocks_per_seq = ctx // BLOCK
    k_cache, v_cache = kv_cache(batch * blocks_per_seq)
    q = torch.randn(batch, HEADS, HEAD_DIM, device="cuda", dtype=DTYPE)
    tables = torch.randperm(
        batch * blocks_per_seq, device="cuda", dtype=torch.int32
    ).view(batch, -1)
    lens = torch.full((batch,), ctx, device="cuda", dtype=torch.int32)
    parts = (ctx + PARTITION - 1) // PARTITION
    use_v1 = parts == 1 or batch * HEADS > 512
    out = {k: torch.empty_like(q) for k in ("r", "s")}
    if not use_v1:
        exp_sums = torch.empty(batch, HEADS, parts, device="cuda", dtype=torch.float32)
        max_logits = torch.empty_like(exp_sums)
        tmp = torch.empty(batch, HEADS, parts, HEAD_DIM, device="cuda", dtype=DTYPE)

    def attn(m, o):
        if use_v1:
            return lambda: m.paged_attention_v1(
                o,
                q,
                k_cache,
                v_cache,
                KV_HEADS,
                SCALE,
                tables,
                lens,
                BLOCK,
                ctx,
                None,
                "auto",
                one,
                one,
            )
        return lambda: m.paged_attention_v2(
            o,
            exp_sums,
            max_logits,
            tmp,
            q,
            k_cache,
            v_cache,
            KV_HEADS,
            SCALE,
            tables,
            lens,
            BLOCK,
            ctx,
            None,
            "auto",
            one,
            one,
        )

    reg, stb = attn(REGULAR, out["r"]), attn(STABLE, out["s"])
    reg()
    stb()
    assert torch.equal(out["r"], out["s"]), "outputs differ"
    row(
        f"paged_attention_{'v1' if use_v1 else 'v2'} decode bs={batch} ctx={ctx}",
        reg,
        stb,
    )

# Writing new K/V into the cache: decode (1 token per sequence) and a 2048-token prefill.
for num_tokens, what in [
    (1, "decode bs=1"),
    (32, "decode bs=32"),
    (128, "decode bs=128"),
    (2048, "prefill 2048 tok"),
]:
    key = torch.randn(num_tokens, KV_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE)
    value = torch.randn_like(key)
    num_blocks = 4 * (num_tokens + BLOCK - 1) // BLOCK
    slots = torch.randperm(num_blocks * BLOCK, device="cuda")[:num_tokens]
    rc, sc = kv_cache(num_blocks, zero=True), kv_cache(num_blocks, zero=True)
    reg = lambda: REGULAR.reshape_and_cache(key, value, *rc, slots, "auto", one, one)
    stb = lambda: STABLE.reshape_and_cache(key, value, *sc, slots, "auto", one, one)
    reg()
    stb()
    assert all(torch.equal(x, y) for x, y in zip(rc, sc)), "outputs differ"
    row(f"reshape_and_cache {what}", reg, stb)
