"""
Given the 256 register tokens per image and 6 dims for each, apply FSQ quantization
and convert each 6-D register code to a single discrete token id in [0..V-1].

FSQ (Finite Scalar Quantization) turns each 6-D code into a single integer ID 
using mixed-radix encoding with per-dimension levels [8, 8, 8, 5, 5, 5] → 64,000 possible IDs.

Input NPZ is expected to contain an array with the 6-D codes, either shaped
- [N, 256, 6] or
- [N*256, 6] or
- [N, 256*6] (flattened per-image; e.g., 1536 cols)

Output NPZ will contain:
- token_ids: [N, 256] int32
- levels: the FSQ levels used
- labels: (optional) copied from input if present
"""
from __future__ import annotations
import os
import sys
import argparse
import math
from typing import List, Tuple

import numpy as np
import torch

# Ensure local FlexTok package is importable when running from repo root
# _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# _FLEXTOK_ROOT = os.path.join(_THIS_DIR, "external", "flextok")
# if os.path.isdir(_FLEXTOK_ROOT) and _FLEXTOK_ROOT not in sys.path:
#     sys.path.insert(0, _FLEXTOK_ROOT)

try:
    from flextok.regularizers.quantize_fsq import FSQ
except Exception as e:
    raise ImportError(
        f"Failed to import FSQ from flextok. Ensure external/flextok is available. Error: {e}"
    )


# Defaults (can be overridden via CLI)
path_tokens_npz = "/BS/latent-diffusion/work/ml-flextok/latents_d18_d18_val.npz"
path_output_npz = "/BS/data_mani_compress/work/thesis/thesis/data/datasets/imagnet_register_tokens/imagnet_val_register_tokens.npz"
DEFAULT_LEVELS = [8, 8, 8, 5, 5, 5]  # V=64_000 for 6-D FSQ used in FlexTok


def _pick_data_key(npz) -> str:
    """Pick the key that holds the 6D codes. Prefer 'data', else first non-'labels'."""
    files = list(npz.files)
    if "data" in files:
        return "data"
    for k in files:
        if k != "labels":
            return k
    if not files:
        raise ValueError("Input NPZ has no arrays")
    return files[0]


def _ensure_shape(arr: np.ndarray, seq_len: int = 256) -> Tuple[np.ndarray, int, int]:
    """Return array shaped [N, L, 6]; infer N if needed. Raises if incompatible."""
    if arr.ndim == 3 and arr.shape[-1] == 6:
        N, L, D = arr.shape
        if L != seq_len:
            raise ValueError(f"Expected sequence length {seq_len}, got {L}")
        return arr, N, L
    if arr.ndim == 2 and arr.shape[-1] == 6:
        total = arr.shape[0]
        if total % seq_len != 0:
            raise ValueError(
                f"Cannot infer N: total rows {total} not divisible by {seq_len}"
            )
        N = total // seq_len
        return arr.reshape(N, seq_len, 6), N, seq_len
    # NEW: handle flattened per-image layout [N, 6*seq_len]
    if arr.ndim == 2 and arr.shape[1] == 6 * seq_len:
        N = arr.shape[0]
        return arr.reshape(N, seq_len, 6), N, seq_len
    raise ValueError(f"Unsupported array shape {arr.shape}; expected [N,256,6] or [N*256,6] or [N,256*6]")


def fsq_codes_to_ids(codes_np: np.ndarray, levels: List[int], chunk_size: int = 8192, device: str = "cpu") -> np.ndarray:
    """Convert 6-D codes in [-1,1] to integer FSQ ids using FSQ's indices mapping.

    Args:
        codes_np: [N, L, 6] float array
        levels: list of radix levels (len=6)
        chunk_size: process N in chunks to reduce memory
        device: 'cpu' or 'cuda' (CPU is fine and safer for big arrays)
    Returns:
        ids_np: [N, L] int32
    """
    N, L, D = codes_np.shape
    assert D == len(levels) == 6

    fsq = FSQ(
        latents_read_key="_",
        quants_write_key="_",
        tokens_write_key="_",
        levels=levels,
    )
    fsq = fsq.to(device)

    ids_out = np.empty((N, L), dtype=np.int32)

    # Precompute half for snapping to grid
    half = (torch.tensor(levels, dtype=torch.float32, device=device) // 2)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = torch.from_numpy(codes_np[start:end]).to(torch.float32).to(device)  # [B,L,6]
        # Clamp and snap to FSQ grid for robustness
        chunk = torch.clamp(chunk, -1.0, 1.0)
        chunk_q = torch.round(chunk * half) / half
        # FSQ codes->indices over last dim
        ids = fsq.codes_to_indices(chunk_q)  # [B,L]
        ids_out[start:end] = ids.detach().cpu().numpy().astype(np.int32)
        del chunk, chunk_q, ids
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return ids_out


def main():
    parser = argparse.ArgumentParser(description="Apply FSQ to 6-D register codes -> integer token IDs")
    parser.add_argument("--input", default=path_tokens_npz, help="Path to NPZ with 6-D codes (default: %(default)s)")
    parser.add_argument("--output", default=path_output_npz, help="Path to save NPZ with token_ids (default: %(default)s)")
    parser.add_argument("--levels", default=",".join(map(str, DEFAULT_LEVELS)), help="Comma-separated FSQ levels, default 8,8,8,5,5,5")
    parser.add_argument("--seq_len", type=int, default=256, help="Registers per image (default: 256)")
    parser.add_argument("--chunk", type=int, default=8192, help="Chunk size over N to limit memory (default: 8192)")
    parser.add_argument("--device", default="cpu", choices=["cpu","cuda"], help="Torch device (default: cpu)")
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x]
    if len(levels) != 6:
        raise ValueError(f"Expected 6 levels, got {levels}")

    print(f"Loading NPZ from: {args.input}")
    npz = np.load(args.input)
    data_key = _pick_data_key(npz)
    arr = npz[data_key]
    print(f"Using key '{data_key}', shape={arr.shape}")

    # Reshape to [N, L, 6]
    codes, N, L = _ensure_shape(arr, seq_len=args.seq_len)
    print(f"Reshaped to [N,L,6] = {codes.shape}")

    print("Converting 6-D FSQ codes -> integer token IDs ...")
    ids = fsq_codes_to_ids(codes, levels=levels, chunk_size=args.chunk, device=args.device)
    print(f"Done. ids shape: {ids.shape} (dtype={ids.dtype})")

    # Prepare output dict
    out = {
        "token_ids": ids,
        "levels": np.array(levels, dtype=np.int32),
    }
    # Copy labels if present
    if "labels" in npz.files:
        out["labels"] = npz["labels"]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(args.output, **out)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()

