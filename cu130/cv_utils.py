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

# Regular vs stable-ABI build of kernels-community/cv-utils (v1), on the workload transformers
# uses it for: SAM3 video post-processing (models/sam3_video/modeling_sam3_video.py).
#   - cc_2d(mask, get_counts=True) on low-res masks [num_objects, 1, 288, 288] uint8, twice per frame
#     (fill holes + remove sprinkles).
#   - generic_nms(mask_iou, probs, 0.1, use_iou_matrix=True) on the detector's valid queries (<= 200).

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


REGULAR = load("kernels-community/cv-utils", "v1", "torch211-cxx11-cu130-x86_64-linux")
STABLE = load(
    "kernels-community/cv-utils", "v1", "torch-stable-abi210-cu130-x86_64-linux"
)
header()

torch.manual_seed(0)
LOW_RES = 288  # Sam3VideoConfig.low_res_mask_size


def blob_masks(n, size=LOW_RES):
    # Object-like masks: smooth noise thresholded into a few blobs, with small holes/specks.
    x = torch.rand(n, 1, size // 16, size // 16, device="cuda")
    x = torch.nn.functional.interpolate(x, size=(size, size), mode="bicubic")
    x += 0.15 * torch.rand_like(x)
    return (x > 0.6).to(torch.uint8).contiguous()


for num_objects in [1, 4, 16, 64]:
    m = blob_masks(num_objects)
    a, b = REGULAR.cc_2d(m, True), STABLE.cc_2d(m, True)
    assert len(a) == len(b)
    assert all(torch.equal(x, y) for x, y in zip(a, b)), "outputs differ"
    row(
        f"cc_2d counts {num_objects} obj x 288x288",
        lambda: REGULAR.cc_2d(m, True),
        lambda: STABLE.cc_2d(m, True),
    )

for num_valid in [8, 32, 200]:  # valid detections out of num_queries=200
    masks = blob_masks(num_valid).flatten(1).float()
    inter = masks @ masks.T
    area = masks.sum(1)
    ious = (inter / (area[:, None] + area[None, :] - inter).clamp(min=1)).contiguous()
    probs = torch.rand(num_valid, device="cuda")
    a, b = (
        REGULAR.generic_nms(ious, probs, 0.1, True),
        STABLE.generic_nms(ious, probs, 0.1, True),
    )
    assert torch.equal(a, b), "outputs differ"
    row(
        f"generic_nms mask-iou {num_valid} dets",
        lambda: REGULAR.generic_nms(ious, probs, 0.1, True),
        lambda: STABLE.generic_nms(ious, probs, 0.1, True),
    )
