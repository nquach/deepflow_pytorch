# deepflow-pytorch

Pure PyTorch optical flow on CPU, NVIDIA CUDA, Apple Silicon (MPS), and Google TPU (through `torch_xla`, `DeepFlow` only). There are no custom CUDA or Metal kernels.

Two entry points share the pyramid and the variational solver:

- `DeepFlow` matches OpenCV's `cv2.optflow.createOptFlow_DeepFlow`.
- `DeepFlowICCV` follows Weinzaepfel et al., ICCV 2013: the same coarse-to-fine solver, plus DeepMatching correspondences and a matching energy.

## OpenCV-compatible DeepFlow

OpenCV's DeepFlow does **not** compute DeepMatching correspondences. It blurs the two grayscale frames, builds a coarse-to-fine pyramid, and at every level runs Brox variational refinement (color constancy, gradient constancy, robust smoothness, fixed-point iterations, red-black SOR). Switching from OpenCV means the same inputs and, within numerical tolerance, the same flow.

## Install

```bash
pip install -e .
```

Runtime dependencies are PyTorch and NumPy. Comparing against OpenCV in the test suite also needs `opencv-contrib-python` and `pytest`:

```bash
pip install -e ".[test]"
pytest
```

## Usage

```python
import cv2
import deepflow

gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# OpenCV
flow_cv = cv2.optflow.createOptFlow_DeepFlow().calc(gray0, gray1, None)

# This package. Same grayscale inputs, same (H, W, 2) float32 flow.
flow = deepflow.DeepFlow().calc(gray0, gray1)
```

Channel 0 is horizontal displacement in pixels (x to the right). Channel 1 is vertical (y down). Any preallocated flow argument is ignored, as in OpenCV.

```python
import torch
import deepflow

model = deepflow.DeepFlow(device="mps")  # or "cuda"
# (N, 1, H, W) uint8 or float -> (N, 2, H, W) float32 on that device
flow = model.calc(batch0, batch1)
```

Torch layouts:

| Input | Output |
| --- | --- |
| `(H, W)`, `(1, H, W)` | `(2, H, W)` |
| `(N, H, W)`, `(N, 1, H, W)` | `(N, 2, H, W)` |

NumPy `(H, W)` or `(H, W, 1)` returns NumPy `(H, W, 2)`. With `device` set, NumPy inputs are computed on that device and copied back. Torch inputs with no `device` stay on the tensor's device. A batch must share one height and width.

Color images raise. OpenCV rejects anything but one channel, and BGR versus RGB weights are not guessed here.

## Parameters

Defaults are OpenCV's. `zeta` and `epsilon` are not DeepFlow fields; they are the variational solver's defaults and are exposed here so they can be inspected.

| Field | Default | Role |
| --- | --- | --- |
| `sigma` | 0.6 | Gaussian blur before the pyramid. Kernel size is `floor(3*sigma)*2+1`. |
| `min_size` | 25 | Stop the pyramid when the next level would be this small on either side. |
| `downscale_factor` | 0.95 | Pyramid scale. Must be in `(0, 1)`. |
| `fixed_point_iterations` | 5 | Inner iterations per level. |
| `sor_iterations` | 25 | Red-black SOR iterations per fixed-point step. |
| `alpha` | 1.0 | Smoothness weight, as stored by OpenCV. |
| `delta` | 0.5 | Color constancy weight, as stored by OpenCV. |
| `gamma` | 5.0 | Gradient constancy weight, as stored by OpenCV. |
| `omega` | 1.6 | SOR relaxation. |
| `max_layers` | 200 | Cap on extra pyramid levels. |
| `zeta` | 0.1 | Derivative normalization inside the solver. |
| `epsilon` | 0.001 | Robust penalty inside the solver. |

OpenCV multiplies these before calling the solver. This package does the same thing internally:

- smoothness weight used by the solver: `4 * alpha`
- color weight: `delta / 3`
- gradient weight: `gamma / 3`

`uint8` frames are converted with `float(pixel)`, so values stay on `0..255`. They are not divided by 255. `zeta` and `epsilon` are absolute, so a float image scaled to `[0, 1]` is a different problem and will not match OpenCV on the original 8-bit frames.

The pyramid is built by blurring once, then repeatedly resizing with bilinear interpolation. Flow upsampled between levels is multiplied by `1 / downscale_factor`, not by the exact ratio of the rounded sizes. That is what OpenCV does.

At the default scale of 0.95 a 1080p pair has on the order of 70 levels, and each level runs 5×25 SOR iterations. The implementation is a small number of dense tensor ops per iteration so it can use the GPU, but that iteration count is part of matching OpenCV.

## Performance

That iteration count means roughly half a million small kernels per frame. Most levels are tiny, so in plain eager mode a GPU spends its time on launch overhead rather than arithmetic. `DeepFlow` and `DeepFlowICCV` both take two constructor options that address that:

| Option | Default | Effect |
| --- | --- | --- |
| `cuda_graph` | `None` (on for CUDA) | The first `calc` for each input shape and setting records the whole pyramid as one CUDA graph; later calls copy the frames in and replay it. Results are identical to eager mode. Each recorded shape keeps its own GPU memory, and the first call costs two to three eager runs (warm-up, capture, replay). Ignored on CPU, MPS, and TPU. |
| `compile` | `False` | Runs each red/black SOR iteration through `torch.compile(dynamic=True)`, which fuses it into a few kernels. Works with `cuda_graph`. The first call compiles (tens of seconds), and fused kernels may round differently from OpenCV by about one ulp. |

```python
model = deepflow.DeepFlow(device="cuda", compile=True)  # cuda_graph is on by default
for frame0, frame1 in pairs:          # same resolution: captured once, replayed after
    flow = model.calc(frame0, frame1)
```

For `DeepFlowICCV` the CUDA graph also covers DeepMatching, so one replay runs the whole pipeline. Changing a field after the first call records a new graph rather than replaying a stale one. With `compile=True`, expect ICCV results to drift further from eager mode than DeepFlow's do: the matching term can amplify one-ulp rounding changes from fused kernels.

Batching several pairs into one `(N, 1, H, W)` call is the other lever: the coarse levels are too small to fill a GPU with a single pair. Time your setup with:

```bash
python benchmarks/bench_deepflow.py --sizes 480p 720p 1080p --batch 1 4
```

## TPU (Colab)

`DeepFlow` runs on Google TPUs through `torch_xla`. On Colab, choose a TPU runtime (`torch_xla` is preinstalled there), then install this package with `pip install -e .`. Elsewhere, install `torch_xla[tpu]` at the version matching your `torch`.

```python
import torch_xla.runtime as xr
import deepflow

xr.initialize_cache("/content/xla_cache")      # optional: keep XLA compiles across runs; call first
model = deepflow.DeepFlow(device="tpu", compile=True)   # "xla" works too
for frame0, frame1 in pairs:
    flow = model.calc(frame0, frame1)
```

- XLA records operations and compiles one executable per tensor shape. `DeepFlow` cuts the recording once per pyramid level, so a resolution needs about one compiled graph per level (about 70 at 1080p).
- Without `compile`, Python re-records every operation on every call, even when the compiled graph is cached. That costs seconds per large frame.
- `compile=True` uses `torch.compile(backend="openxla")` on each fixed-point step, so later frames skip re-recording. Use it for video. The first frame at each resolution (and batch size) is slow while those graphs are traced and compiled, potentially minutes at 1080p. Later frames reuse them. The persistent cache above saves the XLA compile part across sessions; the tracing still happens once per process.
- Results match the CPU within a small tolerance, not bit for bit: TPU square root and division round differently.
- `cuda_graph` is ignored on TPU. `DeepFlowICCV` does not support TPU yet. DeepMatching depends on convolution precision (TPUs default to bfloat16 passes for float32 matmuls) and on `scatter_reduce` tie-breaking, which have not been validated there.
- Benchmark with `python benchmarks/bench_deepflow.py --device tpu --sizes 480p 720p --batch 1 4`, and run `pytest tests/test_tpu.py` to check the setup.

## ICCV 2013 DeepFlow

`DeepFlowICCV` takes the same fields and defaults as `DeepFlow`, and `calc` accepts the same grayscale layouts. It does **not** apply OpenCV's `4 * alpha`, `delta / 3`, and `gamma / 3` rescaling. The values you pass are the weights in the paper's energy.

```python
flow = deepflow.DeepFlow().calc(gray0, gray1)
flow_paper = deepflow.DeepFlowICCV().calc(gray0, gray1)
```

Extra fields, at the published defaults:

| Field | Default | Role |
| --- | --- | --- |
| `beta` | 300 | Matching weight at the coarsest pyramid level. |
| `beta_exponent` | 0.6 | `beta_k = beta * (k / k_max) ** beta_exponent`. `k = 0` is the finest level, so fine levels are downweighted. A one-level pyramid uses `beta` unchanged. |
| `sigma_m` | 50 | Spread of the match confidence. |
| `match_power` | 1.4 | Correlation rectification during aggregation. |
| `descriptor_smooth` | 1 | Smoothing radius inside the 8-bin pixel descriptor. |
| `sigmoid_slope` | 0.2 | Sigmoid slope on each orientation. |
| `descriptor_regularizer` | 0.3 | Constant ninth descriptor bin. |
| `match_max_side` | 256 | Longest side of the pair before matching, keeping aspect ratio. This is the resolution the ICCV experiments used (they resized to about 256×128). `None` matches at full resolution. |

Matches are computed once, then moved onto each pyramid level. Displacements are resized and multiplied by the true size ratio of that level to the matching resolution. The mask is nearest-neighbor. The confidence is bilinear and then multiplied by the resized mask.

The Sintel operating point from the extended paper is `delta=0`, `gamma=0.8`, `sigma=0.5`, `beta=300`, `beta_exponent=0.6`. It is not the default, because the shared fields stay aligned with `DeepFlow`.

```python
sintel = deepflow.DeepFlowICCV(delta=0.0, gamma=0.8, sigma=0.5, beta=300.0, beta_exponent=0.6)
flow = sintel.calc(gray0, gray1)
```

DeepMatching here is the exact-but-chunked 2D procedure from the extended paper (8-bin descriptors, correlation pyramid, backtracking from the top level, reciprocal 4×4 merge). The fast k-means approximation and the scale/rotation search are not included. Bit-exact agreement with the authors' binary is not a goal; that program adds heuristics the papers do not specify.
