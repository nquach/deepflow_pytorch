"""Optional torch_xla (TPU) hooks. ``torch_xla`` is imported only when an XLA device is used."""

from __future__ import annotations

from typing import Any

import torch

_INSTALL_HINT = (
    "TPU support needs torch_xla. On Colab, pick a TPU runtime (torch_xla is "
    "preinstalled there); elsewhere run `pip install torch_xla[tpu]` with the "
    "version matching your torch."
)


def is_xla(device: torch.device | None) -> bool:
    return device is not None and device.type == "xla"


def _xla_model() -> Any:
    try:
        import torch_xla.core.xla_model as xm
    except ImportError as error:
        raise ImportError(_INSTALL_HINT) from error
    return xm


def xla_device() -> torch.device:
    """The default XLA device (the local TPU core)."""
    xm = _xla_model()
    import torch_xla

    if hasattr(torch_xla, "device"):
        return torch.device(torch_xla.device())
    return torch.device(xm.xla_device())


def sync() -> None:
    """Cut the lazy graph here: compile (or reuse) and launch everything recorded so far."""
    xm = _xla_model()
    import torch_xla

    if hasattr(torch_xla, "sync"):
        torch_xla.sync()
    else:
        xm.mark_step()


def wait() -> None:
    """Block until queued TPU work has finished (for timing)."""
    _xla_model().wait_device_ops()
