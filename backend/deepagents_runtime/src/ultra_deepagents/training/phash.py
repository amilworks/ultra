"""Self-contained 64-bit perceptual hash for the GoldGate leakage defenses.

The kernel is fully specified so cross-time comparisons are meaningful: the
inventory header records PHASH_KERNEL, and the freeze-time hasher refuses to
compare hashes produced by a different kernel (plan section 5e — the same
discipline as the gate engine's kernel_version rule). Do NOT change the
algorithm without minting a new kernel id and re-hashing the corpus.

Algorithm (phash64/goldgate-dct@1):
  1. Convert to 8-bit grayscale (PIL 'L', ITU-R 601-2 luma).
  2. Resize to 32x32 with LANCZOS resampling.
  3. 2D DCT-II (orthogonal), take the top-left 8x8 low-frequency block.
  4. Threshold each of the 64 coefficients against the median of the block
     EXCLUDING the DC term; bit = coefficient > median.
  5. Pack row-major into 64 bits, rendered as 16 hex chars.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
from PIL import Image

PHASH_KERNEL = "phash64/goldgate-dct@1/gray-lanczos32-dct8x8-median-ex-dc"


@functools.lru_cache(maxsize=2)
def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    matrix = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * i + 1) * k / (2.0 * n))
    matrix[0, :] = np.sqrt(1.0 / n)
    return matrix


def phash64(image: Image.Image) -> str:
    gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float64)
    dct = _dct_matrix(32)
    coefficients = dct @ pixels @ dct.T
    block = coefficients[:8, :8]
    flat = block.flatten()
    median = np.median(flat[1:])  # exclude the DC term
    bits = flat > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def phash64_file(path: str | Path) -> str:
    with Image.open(path) as image:
        return phash64(image)


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")
