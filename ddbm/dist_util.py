"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

SETUP_RETRY_COUNT = 3


# def setup_dist(devices=None):
#     """
#     Setup a distributed process group.
#     Args:
#         devices: List of device indices to use. If None, uses all available devices.
#     """
#     if dist.is_initialized():
#         return
    
#     if devices is None:
#         devices = list(range(th.cuda.device_count()))
    
#     # Initialize environment variables
#     os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
#     tmp = str(_find_free_port())
#     print(tmp)
#     os.environ.setdefault("MASTER_PORT", tmp)
    
#     # These should be set by the launcher (torchrun or similar)
#     rank = int(os.environ.get("RANK", 0))
#     world_size = int(os.environ.get("WORLD_SIZE", len(devices)))
    
#     backend = "gloo" if not th.cuda.is_available() else "nccl"

#     if th.cuda.is_available():
#         device_idx = devices[rank % len(devices)]
#         th.cuda.set_device(device_idx)
    
#     print(f"Initializing process group: rank={rank}, world_size={world_size}")
#     dist.init_process_group(
#         backend=backend,
#         init_method="env://",
#         rank=rank,
#         world_size=world_size,
#         timeout=th.timedelta(seconds=60)  # Add timeout to prevent hanging
#     )
#     print(f"Successfully initialized process group: rank={rank}, world_size={world_size}")

# def dev():
#     """
#     Get the device to use for torch.distributed.
#     """
#     if th.cuda.is_available():
#         return th.device(f"cuda:{th.cuda.current_device()}")
#     return th.device("cpu")


def setup_dist():
    if dist.is_initialized():
        return

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if th.cuda.is_available():
        th.cuda.set_device(local_rank)

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(_find_free_port())

    dist.init_process_group(
        backend="nccl" if th.cuda.is_available() else "gloo",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

def dev():
    return th.device(f"cuda:{os.environ['LOCAL_RANK']}" if th.cuda.is_available() else "cpu")

def load_state_dict(path, **kwargs):
    rank = dist.get_rank()
    chunk_size = 2**30  # 1GB

    if rank == 0:
        with open(path, "rb") as f:
            data = f.read()
        num_chunks = (len(data) + chunk_size - 1) // chunk_size
        num_chunks_tensor = th.tensor(num_chunks, dtype=th.int)
        dist.broadcast(num_chunks_tensor, 0)
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk = data[start:end]
            chunk_tensor = th.ByteTensor(bytearray(chunk))
            dist.broadcast(chunk_tensor, 0)
    else:
        num_chunks_tensor = th.tensor(0, dtype=th.int)
        dist.broadcast(num_chunks_tensor, 0)
        num_chunks = num_chunks_tensor.item()
        data = bytearray()
        
        for _ in range(num_chunks):
            chunk_tensor = th.ByteTensor(chunk_size)
            dist.broadcast(chunk_tensor, 0)
            data += bytes(chunk_tensor.numpy())

    return th.load(io.BytesIO(data), **kwargs)

def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# def load_state_dict(path, **kwargs):
#     """
#     Load a PyTorch file synchronously across processes.
#     """
#     if dist.is_initialized():
#         return _distributed_load(path, **kwargs)
#     else:
#         return _local_load(path, **kwargs)

def _distributed_load(path, **kwargs):
    chunk_size = 2**30  # 1GB chunks
    rank = dist.get_rank()
    
    if rank == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = (len(data) + chunk_size - 1) // chunk_size
        num_chunks_tensor = th.tensor([num_chunks], dtype=th.long)
        dist.broadcast(num_chunks_tensor, 0)
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk = data[start:end]
            chunk_len = th.tensor([len(chunk)], dtype=th.long)
            dist.broadcast(chunk_len, 0)
            chunk_tensor = th.ByteTensor(bytearray(chunk))
            dist.broadcast(chunk_tensor, 0)
    else:
        num_chunks_tensor = th.tensor([0], dtype=th.long)
        dist.broadcast(num_chunks_tensor, 0)
        num_chunks = num_chunks_tensor.item()
        data = bytearray()
        
        for _ in range(num_chunks):
            chunk_len = th.tensor([0], dtype=th.long)
            dist.broadcast(chunk_len, 0)
            chunk_tensor = th.ByteTensor(chunk_len.item())
            dist.broadcast(chunk_tensor, 0)
            chunk = bytes(chunk_tensor.numpy().tobytes()[:chunk_len.item()])
            data += chunk
    
    return th.load(io.BytesIO(data), **kwargs)

def _local_load(path, **kwargs):
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)

def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)

# def _find_free_port():
#     try:
#         s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         s.bind(("", 0))
#         s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#         return s.getsockname()[1]
#     finally:
#         s.close()