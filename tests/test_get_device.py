"""Device selection: CUDA, Apple MPS, CPU."""

from __future__ import annotations

import pytest
import torch

from utils import get_device


def test_get_device_auto_prefers_cuda_then_mps(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda idx: "fake-cuda")
    assert get_device("auto").type == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("utils._mps_is_available", lambda: True)
    assert get_device("auto").type == "mps"

    monkeypatch.setattr("utils._mps_is_available", lambda: False)
    assert get_device("auto").type == "cpu"


def test_get_device_explicit_mps_falls_back_without_backend(monkeypatch):
    monkeypatch.setattr("utils._mps_is_available", lambda: False)
    assert get_device("mps").type == "cpu"


def test_get_device_cpu_passthrough():
    assert get_device("cpu").type == "cpu"


@pytest.mark.skipif(
    not (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    ),
    reason="MPS not available",
)
def test_optional_mps_tensor_roundtrip():
    device = get_device("mps")
    assert device.type == "mps"
    x = torch.ones(2, device=device)
    y = (x + 1).cpu()
    assert y.tolist() == [2.0, 2.0]
