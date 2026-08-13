"""Rollout baseline: FPS, process RSS, and optional GPU memory / util."""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional

import torch

from config import TrainConfig
from train import build_ppo
from training.make_env import make_vec_env
from utils import get_device, get_logger, set_global_seed

logger = get_logger(__name__)


@dataclass
class BaselineMetrics:
    """Wall-clock training-step baseline for demo vs full-scale gates."""

    fps: float
    rss_mb: float
    cuda_alloc_mb: float
    gpu_util_pct: Optional[float]
    n_env_steps: int
    n_envs: int
    n_machines: int
    n_jobs: int
    avg_ops: int
    device: str

    def format_summary(self) -> str:
        gpu = "n/a" if self.gpu_util_pct is None else f"{self.gpu_util_pct:.1f}%"
        return "\n".join(
            [
                f"Instance           : {self.n_machines}x{self.n_jobs}x{self.avg_ops}",
                f"Device / n_envs    : {self.device} / {self.n_envs}",
                f"Env-steps          : {self.n_env_steps}",
                f"FPS (env-steps/s)  : {self.fps:.2f}",
                f"RSS                : {self.rss_mb:.1f} MB",
                f"CUDA allocated     : {self.cuda_alloc_mb:.1f} MB",
                f"GPU util           : {gpu}",
            ]
        )


def process_rss_bytes() -> int:
    """Current process working-set / max RSS in bytes."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.WorkingSetSize)

    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(rss)
    return int(rss) * 1024


def gpu_util_pct() -> Optional[float]:
    """Snapshot GPU utilization, or None when nvidia-smi is unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
        return float(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def measure_training_baseline(
    cfg: TrainConfig,
    *,
    n_env_steps: int = 64,
) -> BaselineMetrics:
    """Time policy.predict + env.step for ``n_env_steps`` vectorized steps."""
    if n_env_steps <= 0:
        raise ValueError(f"n_env_steps must be positive, got {n_env_steps}")

    set_global_seed(cfg.seed, deterministic=cfg.deterministic_torch)
    device = get_device(cfg.device)
    env = make_vec_env(
        cfg,
        n_envs=cfg.n_envs,
        use_subprocess=False,
        for_eval=True,
    )
    try:
        model = build_ppo(cfg, env)
        obs = env.reset()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        start = time.perf_counter()
        for _ in range(int(n_env_steps)):
            actions, _states = model.predict(obs, deterministic=True)
            obs, _rewards, _dones, _infos = env.step(actions)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = max(time.perf_counter() - start, 1e-8)

        total_steps = int(n_env_steps) * int(cfg.n_envs)
        cuda_mb = 0.0
        if device.type == "cuda":
            cuda_mb = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)

        metrics = BaselineMetrics(
            fps=float(total_steps) / elapsed,
            rss_mb=process_rss_bytes() / (1024.0 * 1024.0),
            cuda_alloc_mb=cuda_mb,
            gpu_util_pct=gpu_util_pct() if device.type == "cuda" else None,
            n_env_steps=int(n_env_steps),
            n_envs=int(cfg.n_envs),
            n_machines=int(cfg.env.n_machines),
            n_jobs=int(cfg.env.n_jobs),
            avg_ops=int(cfg.env.avg_operations_per_job),
            device=str(device),
        )
        logger.info("Baseline:\n%s", metrics.format_summary())
        return metrics
    finally:
        env.close()


__all__ = [
    "BaselineMetrics",
    "gpu_util_pct",
    "measure_training_baseline",
    "process_rss_bytes",
]
