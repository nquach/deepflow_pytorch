"""Time DeepFlow against OpenCV.

    python benchmarks/bench_deepflow.py --sizes 480p 720p 1080p --batch 1 4

Reports the median seconds per frame pair after warm-up. CUDA timings
synchronize before reading the clock. The first call for each shape (graph
capture, compilation) is excluded.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import torch

import deepflow

SIZES = {"240p": (240, 320), "480p": (480, 640), "720p": (720, 1280), "1080p": (1080, 1920)}


def _pair(height: int, width: int, batch: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    frames0, frames1 = [], []
    for _ in range(batch):
        phase = rng.rand() * 6.0
        base = 110 + 50 * np.sin(xx / 17.0 + phase) * np.cos(yy / 13.0) + rng.rand(height, width) * 30
        frames0.append(base.astype(np.uint8))
        frames1.append(np.roll(np.roll(base, 2, axis=0), -3, axis=1).astype(np.uint8))
    return np.stack(frames0), np.stack(frames1)


def _time(fn, repeats: int, sync: bool) -> float:
    samples = []
    for _ in range(repeats):
        if sync:
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", nargs="+", default=["480p", "720p", "1080p"], choices=sorted(SIZES))
    parser.add_argument("--batch", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-opencv", action="store_true")
    parser.add_argument("--no-compile", action="store_true", help="skip the torch.compile variant")
    args = parser.parse_args()

    device = torch.device(args.device)
    on_cuda = device.type == "cuda"
    variants: dict[str, dict] = {"eager": dict(cuda_graph=False)}
    if on_cuda:
        variants["cuda_graph"] = dict(cuda_graph=True)
        if not args.no_compile:
            variants["compile+cuda_graph"] = dict(cuda_graph=True, compile=True)
    elif not args.no_compile:
        variants["compile"] = dict(compile=True)

    cv2 = None
    if not args.no_opencv:
        try:
            import cv2
        except ImportError:
            print("opencv-contrib-python not installed; skipping OpenCV")

    print(f"device={device}  repeats={args.repeats}  (seconds per frame pair, median)")
    header = f"{'size':>6} {'N':>3} " + " ".join(f"{name:>20}" for name in (["opencv"] if cv2 else []) + list(variants))
    print(header)
    for size in args.sizes:
        height, width = SIZES[size]
        for batch in args.batch:
            frames0, frames1 = _pair(height, width, batch)
            row = [f"{size:>6}", f"{batch:>3}"]
            if cv2 is not None:
                solver = cv2.optflow.createOptFlow_DeepFlow()

                def run_cv() -> None:
                    for a, b in zip(frames0, frames1):
                        solver.calc(a, b, None)

                row.append(f"{_time(run_cv, args.repeats, False) / batch:>20.3f}")
            t0 = torch.from_numpy(frames0)[:, None].to(device)
            t1 = torch.from_numpy(frames1)[:, None].to(device)
            for options in variants.values():
                model = deepflow.DeepFlow(device=device, **options)
                model.calc(t0, t1)  # capture / compile / cache warm-up
                seconds = _time(lambda: model.calc(t0, t1), args.repeats, on_cuda)
                row.append(f"{seconds / batch:>20.3f}")
                del model
                if on_cuda:
                    torch.cuda.empty_cache()
            print(" ".join(row), flush=True)


if __name__ == "__main__":
    main()
