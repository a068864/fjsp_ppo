"""Shared utilities for FJSP PPO: seeding, devices, actions, checkpoints, logging."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch


PathLike = Union[str, Path]


def unflatten_action(action_idx: int, n_operations: int) -> Tuple[int, int]:
    """Decode a flat discrete action index into ``(machine_id, operation_id)``.

    Args:
        action_idx: Flat action index.
        n_operations: Total number of operations in the instance.

    Returns:
        Tuple ``(machine_id, operation_id)``.
    """
    if n_operations <= 0:
        raise ValueError(f"n_operations must be positive, got {n_operations}")
    if action_idx < 0:
        raise ValueError(f"action_idx must be non-negative, got {action_idx}")
    machine_id = int(action_idx // n_operations)
    operation_id = int(action_idx % n_operations)
    return machine_id, operation_id


def worker_seed(base_seed: int, rank: int) -> int:
    """Derive a deterministic per-worker seed for vectorized environments.

    Args:
        base_seed: Global experiment seed.
        rank: Worker / subprocess rank (``0 .. n_envs-1``).

    Returns:
        Unique seed for the given worker.
    """
    if rank < 0:
        raise ValueError(f"rank must be non-negative, got {rank}")
    return int(base_seed) + int(rank) * 1000


def _mps_is_available() -> bool:
    """Return True when Apple Metal (MPS) backend can run tensors."""
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps is None:
        return False
    try:
        return bool(mps.is_available())
    except Exception:
        return False


def get_device(device: str = "auto") -> torch.device:
    """Select a torch device: CUDA, then Apple MPS, then CPU.

    Args:
        device: ``"auto"``, ``"cpu"``, ``"mps"``, ``"cuda"``, or a specific
            device string such as ``"cuda:0"``.

    Returns:
        Resolved ``torch.device``.
    """
    log = logging.getLogger(__name__)
    if device == "auto":
        if torch.cuda.is_available():
            resolved = torch.device("cuda")
            log.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
            return resolved
        if _mps_is_available():
            log.info("Using Apple Metal (MPS) device")
            return torch.device("mps")
        log.info("CUDA/MPS unavailable; falling back to CPU")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        log.warning(
            "Requested device %s but CUDA is unavailable; falling back to CPU",
            device,
        )
        return torch.device("cpu")
    if resolved.type == "mps" and not _mps_is_available():
        log.warning(
            "Requested device %s but MPS is unavailable; falling back to CPU",
            device,
        )
        return torch.device("cpu")
    return resolved


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch, CUDA/MPS, and Gymnasium for reproducibility.

    Args:
        seed: Global random seed.
        deterministic: If True, enable deterministic PyTorch algorithms where
            possible. Some CUDA/MPS ops may still warn or fall back.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if _mps_is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass

    try:
        import gymnasium as gym

        gym.utils.seeding.np_random(seed)
    except Exception:
        # Gymnasium seeding helpers vary by version; env-level seeds still apply.
        pass

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only avoids hard crashes on ops without deterministic kernels.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Older PyTorch versions may not support warn_only.
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                logging.getLogger(__name__).warning(
                    "Could not enable torch deterministic algorithms"
                )
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def ensure_dir(path: PathLike) -> Path:
    """Create a directory (and parents) if it does not exist.

    Args:
        path: Directory path.

    Returns:
        Resolved ``Path`` of the directory.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def checkpoint_exists(path: PathLike) -> bool:
    """Return True if a checkpoint file exists at ``path``."""
    return Path(path).is_file()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create or retrieve a module logger with a standard stream handler.

    Args:
        name: Logger name (typically ``__name__``).
        level: Logging level.

    Returns:
        Configured ``logging.Logger``.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def configure_root_logging(level: int = logging.INFO) -> None:
    """Configure root logging once for training/evaluation entry points."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        root.setLevel(level)


def safe_mean(values: np.ndarray, default: float = 0.0) -> float:
    """Compute mean of an array, returning ``default`` when empty."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float(default)
    return float(np.mean(arr))
