# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-host resource exchange for the PAP local transport."""

from __future__ import annotations

import base64
import mmap
import os
import pickle
import socket

import torch


def _local_hostname() -> str:
    """Return a best-effort stable hostname for same-machine detection."""

    candidate = os.environ.get("PAP_LOCAL_FAST_HOSTNAME")
    if candidate:
        return candidate
    try:
        return socket.gethostname() or "unknown"
    except Exception:
        return "unknown"


def _doorbell_path(actor_id: str) -> str:
    base = os.environ.get("PAP_LOCAL_FAST_DOORBELL_DIR", "/dev/shm")
    safe_actor = "".join(c if c.isalnum() or c in "-_" else "_" for c in actor_id)
    return os.path.join(base, f"pap_local_fast_{safe_actor}.db")


def _open_or_create_doorbell(path: str, size: int) -> tuple[int, mmap.mmap]:
    """Open and size a local-fast doorbell file."""

    flags = os.O_RDWR | os.O_CREAT
    fd = os.open(path, flags, 0o600)
    try:
        os.ftruncate(fd, size)
    except OSError:
        os.close(fd)
        raise
    mm = mmap.mmap(
        fd,
        size,
        flags=mmap.MAP_SHARED,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    return fd, mm


def _pack_cuda_ipc_handle(tensor: torch.Tensor) -> str:
    """Serialize a CUDA tensor reduction for the peer process."""

    from torch.multiprocessing.reductions import reduce_tensor

    rebuild, args = reduce_tensor(tensor)
    return base64.b64encode(pickle.dumps((rebuild, args))).decode("ascii")


def _unpack_cuda_ipc_handle(blob: str) -> torch.Tensor:
    """Rebuild a CUDA tensor from a serialized IPC handle blob."""

    rebuild, args = pickle.loads(base64.b64decode(blob.encode("ascii")))
    return rebuild(*args)


def _ensure_peer_access(local_device: torch.device, peer_device: torch.device) -> bool:
    """Enable local-to-peer CUDA access when supported."""

    if local_device.index == peer_device.index:
        return True
    if local_device.index is None or peer_device.index is None:
        return False
    try:
        can_access = torch.cuda.can_device_access_peer(
            local_device.index,
            peer_device.index,
        )
    except Exception:
        can_access = False
    if not can_access:
        return False
    try:
        torch.cuda.device(local_device.index)
        torch._C._cuda_enable_peer_access(peer_device.index)
    except RuntimeError as exc:
        if "peer access is already enabled" not in str(exc).lower():
            raise
    return True
