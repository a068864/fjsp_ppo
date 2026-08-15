"""Atomic metadata-backed checkpoint helpers (stdlib only)."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from utils import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]

CHECKPOINT_META_SUFFIX = ".meta.json"
BEST_SCORE_NAME = "best_score.json"


def material_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Subset of config that must match to load a checkpoint into a trainer."""
    cfg = dict(config)
    ppo = dict(cfg.get("ppo") or {})
    ppo_keys = (
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "clip_range",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "target_kl",
    )
    return {
        "env": dict(cfg.get("env") or {}),
        "model": dict(cfg.get("model") or {}),
        "n_envs": cfg.get("n_envs"),
        "ppo": {key: ppo[key] for key in ppo_keys if key in ppo},
    }


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """Stable SHA-256 fingerprint of architecture / batch-geometry fields."""
    payload = json.dumps(
        material_config(config),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: PathLike) -> str:
    """SHA-256 of a file (used to bind sidecar metadata to a checkpoint zip)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def meta_path_for(checkpoint_path: PathLike) -> Path:
    """Sidecar metadata path for a checkpoint zip."""
    path = Path(checkpoint_path)
    return path.with_name(path.name + CHECKPOINT_META_SUFFIX)


def best_score_path(checkpoint_dir: PathLike) -> Path:
    """Path to persisted best evaluation score."""
    return Path(checkpoint_dir) / BEST_SCORE_NAME


def atomic_replace(src: PathLike, dst: PathLike) -> None:
    """Atomically replace ``dst`` with ``src`` (same-directory rename)."""
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src_path, dst_path)


def atomic_write_bytes(path: PathLike, data: bytes) -> None:
    """Write bytes via a temp file in the destination directory, then replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        atomic_replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: PathLike, payload: Mapping[str, Any]) -> None:
    """Atomically write a JSON document."""
    data = json.dumps(dict(payload), indent=2, sort_keys=True).encode("utf-8")
    atomic_write_bytes(path, data)


def write_checkpoint_metadata(
    checkpoint_path: PathLike,
    *,
    config: Mapping[str, Any],
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write sidecar metadata next to a checkpoint zip."""
    path = Path(checkpoint_path)
    meta: Dict[str, Any] = {
        "checkpoint": path.name,
        "config_fingerprint": config_fingerprint(config),
        "config": dict(config),
    }
    if extra:
        meta.update(dict(extra))
    if path.is_file():
        meta["zip_sha256"] = file_sha256(path)
    out = meta_path_for(path)
    atomic_write_json(out, meta)
    return out


def read_checkpoint_metadata(checkpoint_path: PathLike) -> Dict[str, Any]:
    """Load sidecar metadata for a checkpoint."""
    path = meta_path_for(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint metadata: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_best_score(checkpoint_dir: PathLike) -> Optional[float]:
    """Return the persisted best score, or None if absent."""
    payload = load_best_score_record(checkpoint_dir)
    if payload is None or "best_score" not in payload:
        return None
    return float(payload["best_score"])


def load_best_score_record(checkpoint_dir: PathLike) -> Optional[Dict[str, Any]]:
    """Return the full best-score JSON payload, or None if absent."""
    path = best_score_path(checkpoint_dir)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return dict(json.load(handle))


def save_best_score(
    checkpoint_dir: PathLike,
    best_score: float,
    *,
    metric: str = "mean_reward",
) -> Path:
    """Persist the best evaluation score atomically."""
    path = best_score_path(checkpoint_dir)
    payload: Dict[str, Any] = {
        "best_metric": str(metric),
        "best_score": float(best_score),
    }
    atomic_write_json(path, payload)
    return path


def assert_config_compatible(
    checkpoint_path: PathLike,
    config: Mapping[str, Any],
    *,
    require_metadata: bool = True,
) -> None:
    """Reject resume when material config fingerprints diverge."""
    meta_file = meta_path_for(checkpoint_path)
    if not meta_file.is_file():
        if require_metadata:
            raise ValueError(
                f"Cannot resume from {checkpoint_path}: missing metadata sidecar "
                f"{meta_file.name}. Start fresh or provide a trusted checkpoint with metadata."
            )
        logger.warning("Checkpoint metadata missing for %s; skipping fingerprint check", checkpoint_path)
        return
    meta = read_checkpoint_metadata(checkpoint_path)
    expected = config_fingerprint(config)
    actual = str(meta.get("config_fingerprint", ""))
    if actual != expected:
        raise ValueError(
            f"Cannot resume from {checkpoint_path}: config fingerprint mismatch "
            f"(checkpoint={actual[:12]}..., current={expected[:12]}...)."
        )
    zip_path = Path(checkpoint_path)
    if not zip_path.is_file():
        raise ValueError(f"Cannot resume from {checkpoint_path}: missing zip")
    expected_zip = file_sha256(zip_path)
    actual_zip = str(meta.get("zip_sha256", ""))
    if actual_zip != expected_zip:
        raise ValueError(
            f"Cannot resume from {checkpoint_path}: zip hash mismatch "
            "(file was replaced or metadata is stale)."
        )


def atomic_save_sb3(model: Any, path: PathLike) -> None:
    """Save an SB3 zip via a temp file in the destination directory, then replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.zip")
    try:
        model.save(str(tmp))
        atomic_replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def save_checkpoint(
    model: Any,
    path: PathLike,
    *,
    config: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    """Atomically save an SB3 zip, then write sidecar metadata when config is given."""
    atomic_save_sb3(model, path)
    if config is not None:
        write_checkpoint_metadata(path, config=config, extra=extra)


__all__ = [
    "BEST_SCORE_NAME",
    "CHECKPOINT_META_SUFFIX",
    "assert_config_compatible",
    "atomic_replace",
    "atomic_save_sb3",
    "atomic_write_bytes",
    "atomic_write_json",
    "best_score_path",
    "config_fingerprint",
    "file_sha256",
    "load_best_score",
    "load_best_score_record",
    "material_config",
    "meta_path_for",
    "read_checkpoint_metadata",
    "save_best_score",
    "save_checkpoint",
    "write_checkpoint_metadata",
]
