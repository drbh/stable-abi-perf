# Torch stable ABI vs regular builds: kernel performance

These scripts compare the wall time of the regular (torch-ABI) and stable-ABI builds of three
[kernels-community](https://huggingface.co/kernels-community) kernels. Each comparison runs both
builds in one process, on the same torch runtime, with workload shapes taken from real usage.

## Layout

```
cu128/            one uv script per kernel, torch 2.11 cu128 (cu126 kernel builds; e.g. RTX 4090)
cu130/            the same scripts for torch 2.11 cu130 (cu130 kernel builds; needed for B200 / sm100)
zeros/            minimal isolated repro: torch::zeros vs torch::stable::new_zeros
results/          raw outputs from the runs below
```

## Run

```
uv run cu128/flash_attn2.py      # or cu128/paged_attention.py, cu128/cv_utils.py
uv run cu130/flash_attn2.py      # on Blackwell (B200); needs a CUDA 13 driver
```

Each script:

1. Downloads only the two variants it compares, into `<kernel>-builds/` next to the script.
2. Loads them with `kernels.get_local_kernel`.
3. Asserts that both builds produce bit-identical outputs.
4. For every workload, times back-to-back calls from both builds. The two builds alternate in
   random order, with 21 repetitions, and the script prints the median µs per call.

| Script | Regular build | Stable build | Workloads |
|---|---|---|---|
| `flash_attn2.py` | `v2` `torch211-cxx11-<cu>` | `v3` `torch-stable-abi210-<cu>` (the version transformers loads) | Llama-3.1-8B attention (32 q / 8 kv heads, head dim 128, bf16), called the way transformers does: `flash_attn_func` prefill and decode (`generate()`), `flash_attn_varlen_func` (continuous batching) |
| `paged_attention.py` | `v1` `torch211-cxx11-<cu>` | `v2` `torch-stable-abi210-<cu>` | Llama-3.1-8B decode, bf16, block size 16, v1/v2 chosen by vLLM's heuristic; `reshape_and_cache` |
| `cv_utils.py` | `v1` `torch211-cxx11-<cu>` | `v1` `torch-stable-abi210-<cu>` | SAM3 video post-processing in transformers: `cc_2d` on `[N,1,288,288]` masks, `generic_nms` on a mask-IoU matrix |

## Results (wall time, µs/call; Δ = stable − regular, range over 2 runs)

- RTX 4090: torch 2.11.0+cu128, cu126 builds. The GPU was shared with other jobs; deltas are
  paired, but absolute times can drift between processes.
- B200: torch 2.11.0+cu130, cu130 builds, idle GPU.

| Kernel | Workload | 4090 regular | 4090 Δ | B200 regular | B200 Δ |
|---|---|---|---|---|---|
| flash-attn2 | `flash_attn_func` prefill bs=1, 512 tok | 30.1 | +0.06 (+0.2%) | 32.3 | **+4.1 to +5.1 (+13–16%)** |
| | `flash_attn_func` prefill bs=1, 2048 tok | 262 | ≈0 | 146 | ≈0 |
| | `flash_attn_func` prefill bs=1, 8192 tok | 3481 | ≈0 | 1543 | ≈0 |
| | `flash_attn_func` decode bs=1, ctx 1024 | 29.2 | **+7.1 to +9.3 (+25–32%)** | 41.1 | **+10.2 to +11.6 (+25–30%)** |
| | `flash_attn_func` decode bs=1, ctx 4096 | 29.3 | **+7.0 to +9.1 (+24–31%)** | 41.1 | **+10.2 to +11.8 (+25–30%)** |
| | `flash_attn_func` decode bs=8, ctx 2048 | 35.5 | +0.3 to +1.9 (+1–6%) | 41.1 | **+10.4 to +11.3 (+25–29%)** |
| | varlen prefill, 8 prompts × 512 | 144 | ≈0 | 105 | ≈0 |
| | varlen decode, 32 / 128 reqs, ctx ~1k | 262 / 641 | ≈0 | 121 / 227 | ≈0 |
| paged-attention | `paged_attention_v1/v2` decode, bs 1–128 | 14–579 | ≈0 | 21–304 | ≈0 |
| | `reshape_and_cache` decode bs=1 | 2.7 | **+0.45 to +0.53 (+17–19%)** | 5.2–8.8 | **+0.5 to +0.9 (+10%)** |
| | `reshape_and_cache` decode bs=32 | 2.7 | **+0.47 to +0.50 (+18–19%)** | 4.3 | **+0.5 to +0.6 (+9–14%)** |
| | `reshape_and_cache` decode bs=128 | 4.1 | ≈0 | 4.3 | **+0.55 to +0.65 (+10–15%)** |
| | `reshape_and_cache` prefill, 2048 tok | 40.9 | ≈0 | 16.4 | ≈0 |
| cv-utils | `cc_2d` counts, 1–64 objects × 288² | 44–376 | ≈0 | 67–282 | ≈0 |
| | `generic_nms` mask-IoU, 8–200 dets | 49–62 | noise (±1–4%, sign flips) | 76–98 | noise (−1% to +2%) |

A regular-vs-regular control gives |Δ| ≤ 0.09 µs. Full output is in `results/`.

### Interpretation

- **The overhead is a fixed CPU cost per call; the GPU work is identical.** Both builds launch the
  same CUDA kernels with the same grid and block sizes, the Python wrappers are byte-identical, and
  the outputs are bit-identical. The stable op costs extra CPU time on every call because its
  allocations, view ops and the op call itself go through the boxed `torch_call_dispatcher` path:
  - flash-attn2: about 5–7 µs (4090), more on the B200 host
  - cv-utils `cc_2d`: about 3 µs
  - paged-attention: about 0.5 µs
- **It only shows in wall time when CPU time sets the pace.** When the GPU work for a call takes
  longer than issuing it, the extra CPU cost is hidden, which is why most rows are ≈0. Eager
  small-batch decode with `flash_attn_func` is CPU-bound, so it regresses about 25–30%, and a
  faster GPU (B200) exposes more cases. Running the step under CUDA graphs should remove this
  overhead; this has not been measured here.
- **Mechanism, isolated (`zeros/`).** On torch 2.11, `torch::stable::new_zeros` costs about
  0.55 µs more per allocation than `torch::zeros`.

### Caveats

- The regular and stable builds come from different source commits. The regular builds are from
  `8a730d9`; the stable ones are from the stable-ABI port commits: cv-utils `255292e`,
  paged-attention `4b8a7b6`, flash-attn2 `2bae138`. This compares the builds as shipped.
- The cv-utils synthetic masks are noisier than real SAM3 masks, and the NMS IoU matrix suppresses
  most detections. Both calls are GPU-bound, so this does not change the result.

## `zeros/`: minimal mechanism repro

There are two kernels with the same two ops. `zeros(x, n)` performs `n` allocations shaped like
`x`, and `noop(x, n)` does nothing. One kernel is built with the regular ABI (`torch::zeros`), the
other with the stable ABI (`torch::stable::new_zeros`). The benchmark reports cost per allocation
as `(zeros - noop) / n`.

```
cd zeros
(cd zeros-classic && nix run github:huggingface/kernels/f6bd904093ee680b77dbcee2b9c9292bb5f2a722#kernel-builder -- build --variant torch211-cu128-x86_64-linux)
(cd zeros-stable  && nix run github:huggingface/kernels/f6bd904093ee680b77dbcee2b9c9292bb5f2a722#kernel-builder -- build --variant torch-stable-abi211-cu128-x86_64-linux)
uv run bench.py
```

kernel-builder requires the kernel to be inside a git repository with at least one commit.

This benchmark runs the regular-ABI kernel and then the stable-ABI kernel in sequence, instead of
alternating them, so use it to understand the mechanism rather than as a precise measurement.
