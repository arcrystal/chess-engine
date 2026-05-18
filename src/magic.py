"""Magic-bitboard generation and caching.

Public surface:
    BISHOP_MASKS:  uint64[64]
    BISHOP_MAGICS: uint64[64]
    BISHOP_SHIFTS: int8[64]
    BISHOP_ATTACKS: uint64[64, 512]
    ROOK_MASKS:    uint64[64]
    ROOK_MAGICS:   uint64[64]
    ROOK_SHIFTS:   int8[64]
    ROOK_ATTACKS:  uint64[64, 4096]
    bishop_attack_index(sq, occ_uint64) -> int (table index)
    rook_attack_index(sq, occ_uint64) -> int

Tables are generated deterministically (seeded RNG per square) and cached to
``~/.cache/chessengine/magic_v1.npz``.

The slider lookup itself lives in ``attacks.py``; this module owns generation.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
CACHE_VERSION = "v1"
CACHE_PATH = Path.home() / ".cache" / "chessengine" / f"magic_{CACHE_VERSION}.npz"

BISHOP_MASKS = np.zeros(64, dtype=np.uint64)
BISHOP_MAGICS = np.zeros(64, dtype=np.uint64)
BISHOP_SHIFTS = np.zeros(64, dtype=np.int8)
BISHOP_ATTACKS = np.zeros((64, 512), dtype=np.uint64)

ROOK_MASKS = np.zeros(64, dtype=np.uint64)
ROOK_MAGICS = np.zeros(64, dtype=np.uint64)
ROOK_SHIFTS = np.zeros(64, dtype=np.int8)
ROOK_ATTACKS = np.zeros((64, 4096), dtype=np.uint64)


def _popcount(x: int) -> int:
    return bin(x).count("1")


def _bishop_mask(sq: int) -> int:
    mask = 0
    r, f = divmod(sq, 8)
    for dr, df in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        nr, nf = r + dr, f + df
        while 0 < nr < 7 and 0 < nf < 7:
            mask |= 1 << (nr * 8 + nf)
            nr += dr
            nf += df
    return mask


def _rook_mask(sq: int) -> int:
    mask = 0
    r, f = divmod(sq, 8)
    for i in range(r + 1, 7):
        mask |= 1 << (i * 8 + f)
    for i in range(r - 1, 0, -1):
        mask |= 1 << (i * 8 + f)
    for i in range(f + 1, 7):
        mask |= 1 << (r * 8 + i)
    for i in range(f - 1, 0, -1):
        mask |= 1 << (r * 8 + i)
    return mask


def _bishop_attacks_for(sq: int, occ: int) -> int:
    attacks = 0
    r, f = divmod(sq, 8)
    for dr, df in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        nr, nf = r + dr, f + df
        while 0 <= nr < 8 and 0 <= nf < 8:
            attacks |= 1 << (nr * 8 + nf)
            if occ & (1 << (nr * 8 + nf)):
                break
            nr += dr
            nf += df
    return attacks


def _rook_attacks_for(sq: int, occ: int) -> int:
    attacks = 0
    r, f = divmod(sq, 8)
    for dr, df in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nf = r + dr, f + df
        while 0 <= nr < 8 and 0 <= nf < 8:
            attacks |= 1 << (nr * 8 + nf)
            if occ & (1 << (nr * 8 + nf)):
                break
            nr += dr
            nf += df
    return attacks


def _set_occupancy(index: int, mask: int, bits: int) -> int:
    occ = 0
    m = mask
    for i in range(bits):
        # Extract the next mask bit.
        lsb = m & -m
        if (index >> i) & 1:
            occ |= lsb
        m &= m - 1
    return occ


def _find_magic(sq: int, mask_fn, attack_fn, rng: random.Random, max_attempts: int = 20_000_000):
    mask = mask_fn(sq)
    bits = _popcount(mask)
    n = 1 << bits
    occupancies = [_set_occupancy(i, mask, bits) for i in range(n)]
    attacks = [attack_fn(sq, occ) for occ in occupancies]
    shift = 64 - bits
    sentinel = -1  # "slot not yet written"

    for _ in range(max_attempts):
        # Sparse-magic trick: ANDing three 64-bit randoms biases toward low popcount.
        magic = rng.getrandbits(64) & rng.getrandbits(64) & rng.getrandbits(64)
        used = [sentinel] * n
        ok = True
        for i in range(n):
            idx = ((occupancies[i] * magic) & 0xFFFFFFFFFFFFFFFF) >> shift
            if used[idx] == sentinel:
                used[idx] = attacks[i]
            elif used[idx] != attacks[i]:
                ok = False
                break
        if ok:
            table = np.zeros(n, dtype=np.uint64)
            for i in range(n):
                idx = ((occupancies[i] * magic) & 0xFFFFFFFFFFFFFFFF) >> shift
                table[idx] = attacks[i]
            return mask, magic, shift, table
    raise RuntimeError(f"failed to find magic for square {sq} after {max_attempts} attempts")


def _generate() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    bm = np.zeros(64, dtype=np.uint64)
    bma = np.zeros(64, dtype=np.uint64)
    bs = np.zeros(64, dtype=np.int8)
    bt = np.zeros((64, 512), dtype=np.uint64)
    rm = np.zeros(64, dtype=np.uint64)
    rma = np.zeros(64, dtype=np.uint64)
    rs = np.zeros(64, dtype=np.int8)
    rt = np.zeros((64, 4096), dtype=np.uint64)

    for sq in range(64):
        rng = random.Random(0xC0FFEE + sq)
        mask, magic, shift, table = _find_magic(sq, _bishop_mask, _bishop_attacks_for, rng)
        bm[sq] = np.uint64(mask)
        bma[sq] = np.uint64(magic)
        bs[sq] = shift
        bt[sq, : table.size] = table

        rng = random.Random(0xDEC0DE + sq)
        mask, magic, shift, table = _find_magic(sq, _rook_mask, _rook_attacks_for, rng)
        rm[sq] = np.uint64(mask)
        rma[sq] = np.uint64(magic)
        rs[sq] = shift
        rt[sq, : table.size] = table

    return bm, bma, bs, bt, rm, rma, rs, rt


def _save(bm, bma, bs, bt, rm, rma, rs, rt) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # ``np.savez`` always appends ``.npz`` — write to a sibling name we control.
    tmp = CACHE_PATH.with_name(CACHE_PATH.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(
            fh,
            bishop_masks=bm, bishop_magics=bma, bishop_shifts=bs, bishop_attacks=bt,
            rook_masks=rm, rook_magics=rma, rook_shifts=rs, rook_attacks=rt,
        )
    os.replace(tmp, CACHE_PATH)


def _load_into_module(bm, bma, bs, bt, rm, rma, rs, rt) -> None:
    BISHOP_MASKS[:] = bm
    BISHOP_MAGICS[:] = bma
    BISHOP_SHIFTS[:] = bs
    BISHOP_ATTACKS[:] = bt
    ROOK_MASKS[:] = rm
    ROOK_MAGICS[:] = rma
    ROOK_SHIFTS[:] = rs
    ROOK_ATTACKS[:] = rt


def initialize(force_regenerate: bool = False) -> None:
    """Populate the module-level magic tables, generating + caching if needed."""
    if not force_regenerate and CACHE_PATH.exists():
        try:
            data = np.load(CACHE_PATH)
            _load_into_module(
                data["bishop_masks"], data["bishop_magics"], data["bishop_shifts"], data["bishop_attacks"],
                data["rook_masks"], data["rook_magics"], data["rook_shifts"], data["rook_attacks"],
            )
            return
        except Exception:
            pass  # fall through to regenerate
    bm, bma, bs, bt, rm, rma, rs, rt = _generate()
    _save(bm, bma, bs, bt, rm, rma, rs, rt)
    _load_into_module(bm, bma, bs, bt, rm, rma, rs, rt)


# Initialize at import.
initialize()
