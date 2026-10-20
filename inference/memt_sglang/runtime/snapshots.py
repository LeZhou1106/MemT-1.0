"""Memory-only safetensors v1; atomic file replacement in a configured directory."""

import os
import re
import tempfile
from pathlib import Path

import torch

from memt_sglang.protocol import SNAPSHOT_FORMAT
from memt_sglang.runtime.config import MAX_SNAPSHOT_NAME_LENGTH


def snapshot_path(directory, name):
    if (
        not directory
        or not isinstance(name, str)
        or not re.fullmatch(
            rf"[A-Za-z0-9_.-]{{1,{MAX_SNAPSHOT_NAME_LENGTH}}}\.safetensors", name
        )
        or name.startswith(".")
    ):
        raise ValueError(
            "Snapshot requires a configured directory and a simple .safetensors name"
        )
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    if target.is_symlink():
        raise ValueError("Snapshot target must not be a symlink")
    return target


def save_memory(pool, slot, path, *, identity, overwrite=False, has_memory=True):
    from safetensors.torch import save_file

    metadata = {"format": SNAPSHOT_FORMAT, "identity": identity}
    tensors = (
        {
            f"memory.{layer}": data[slot].detach().cpu().contiguous()
            for layer, data in pool.storage.items()
        }
        if has_memory
        else {}
    )
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".memt-state-")
    os.close(descriptor)
    try:
        save_file(tensors, temporary, metadata=metadata)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # An automatically allocated key must never replace an existing file.
            os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"file": path.name, "format": metadata["format"], "has_memory": has_memory}


def load_memory(pool, slot, path, *, identity):
    from safetensors import SafetensorError, safe_open

    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            expected = {
                "format": SNAPSHOT_FORMAT,
                "identity": identity,
            }
            if handle.metadata() != expected:
                raise ValueError("Snapshot model/encoding/format identity mismatch")
            names = {f"memory.{layer}": layer for layer in pool.storage}
            keys = set(handle.keys())
            if keys and keys != names.keys():
                raise ValueError("Snapshot memory layer set mismatch")
            tensors = (
                {layer: handle.get_tensor(name) for name, layer in names.items()}
                if keys
                else {}
            )
    except SafetensorError as error:
        raise ValueError("Invalid memory snapshot") from error
    for layer, tensor in tensors.items():
        if (
            tensor.dtype != torch.float32
            or tensor.shape != pool.storage[layer].shape[1:]
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(
                "Snapshot requires finite FP32 tensors with exact layer shapes"
            )
    for layer, storage in pool.storage.items():
        if tensors:
            storage[slot].copy_(tensors[layer])
        else:
            storage[slot].zero_()
    return bool(tensors)


def save_snapshot(
    pool, slot, snapshot_id, *, directory, identity, overwrite=False, has_memory=True
):
    """Persist published memory with one shared snapshot ID."""
    path = snapshot_path(directory, snapshot_id)
    result = save_memory(
        pool,
        slot,
        path,
        identity=identity,
        overwrite=overwrite,
        has_memory=has_memory,
    )
    return {"snapshot_id": snapshot_id, **result}


def load_snapshot(pool, slot, snapshot_id, *, directory, identity):
    """Resolve a storage key and validate before changing the candidate state."""
    path = snapshot_path(directory, snapshot_id)
    has_memory = load_memory(pool, slot, path, identity=identity)
    return {
        "snapshot_id": snapshot_id,
        "file": snapshot_id,
        "format": SNAPSHOT_FORMAT,
        "has_memory": has_memory,
    }
