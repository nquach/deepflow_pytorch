"""Single-scale refinement against cv2.VariationalRefinement."""

import numpy as np
import pytest
import torch

cv2 = pytest.importorskip("cv2")

from deepflow.variational import variational_refine


def _shifted_texture(height: int, width: int, shift_x: int, shift_y: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(0)
    image = rng.randint(0, 256, size=(height, width), dtype=np.uint8).astype(np.float32)
    other = np.roll(np.roll(image, shift_y, axis=0), shift_x, axis=1)
    return image, other


def _opencv_refine(i0, i1, flow, **params):
    solver = cv2.VariationalRefinement_create()
    solver.setAlpha(float(params["alpha"]))
    solver.setDelta(float(params["delta"]))
    solver.setGamma(float(params["gamma"]))
    solver.setOmega(float(params["omega"]))
    solver.setFixedPointIterations(int(params["fixed_point_iterations"]))
    solver.setSorIterations(int(params["sor_iterations"]))
    solver.setEpsilon(float(params["epsilon"]))
    out = np.ascontiguousarray(flow, dtype=np.float32).copy()
    solver.calc(i0, i1, out)
    return out


def _ours(i0, i1, flow, **params):
    def nchw(image):
        return torch.from_numpy(np.ascontiguousarray(image)).view(1, 1, *image.shape)

    flow_t = torch.from_numpy(np.ascontiguousarray(flow)).permute(2, 0, 1)[None]
    u, v = variational_refine(nchw(i0), nchw(i1), flow_t[:, :1], flow_t[:, 1:], **params)
    return torch.cat([u, v], dim=1)[0].permute(1, 2, 0).numpy()


def _assert_close(actual, expected, mean_tol):
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    mean = float(diff.mean())
    maximum = float(diff.max())
    assert mean <= mean_tol, f"mean abs {mean}, max abs {maximum}"


DEEPFLOW_SCALED = dict(
    alpha=4.0,
    delta=0.5 / 3.0,
    gamma=5.0 / 3.0,
    omega=1.6,
    fixed_point_iterations=5,
    sor_iterations=25,
    epsilon=0.001,
)


def test_zero_initial_flow_even_size():
    i0, i1 = _shifted_texture(36, 40, 2, 1)
    init = np.zeros((36, 40, 2), np.float32)
    expected = _opencv_refine(i0, i1, init, **DEEPFLOW_SCALED)
    actual = _ours(i0, i1, init, zeta=0.1, **DEEPFLOW_SCALED)
    _assert_close(actual, expected, 1e-4)


def test_nonzero_initial_flow_odd_width():
    i0, i1 = _shifted_texture(29, 31, -1, 2)
    init = np.zeros((29, 31, 2), np.float32)
    init[..., 0] = 0.4
    init[..., 1] = -0.25
    expected = _opencv_refine(i0, i1, init, **DEEPFLOW_SCALED)
    actual = _ours(i0, i1, init, zeta=0.1, **DEEPFLOW_SCALED)
    _assert_close(actual, expected, 1e-4)
