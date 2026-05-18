"""Attack lookups for all pieces.

Non-slider attacks (knight, king, pawn) are precomputed at import.
Slider attacks (bishop, rook, queen) go through magic bitboards.

All functions accept Python ints OR ``np.uint64`` and return Python ints
representing 64-bit unsigned bitboards (this keeps consumer code simple — the
hot loops live in numpy/numba land and operate directly on the arrays.)
"""

from __future__ import annotations

import numpy as np

from magic import (
    BISHOP_MASKS, BISHOP_MAGICS, BISHOP_SHIFTS, BISHOP_ATTACKS,
    ROOK_MASKS, ROOK_MAGICS, ROOK_SHIFTS, ROOK_ATTACKS,
)

MASK64 = 0xFFFFFFFFFFFFFFFF

KNIGHT_ATTACKS = np.zeros(64, dtype=np.uint64)
KING_ATTACKS = np.zeros(64, dtype=np.uint64)
WHITE_PAWN_ATTACKS = np.zeros(64, dtype=np.uint64)
BLACK_PAWN_ATTACKS = np.zeros(64, dtype=np.uint64)


def _precompute() -> None:
    KNIGHT_DELTAS = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
    KING_DELTAS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

    for sq in range(64):
        r, f = divmod(sq, 8)

        n = 0
        for dr, df in KNIGHT_DELTAS:
            nr, nf = r + dr, f + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                n |= 1 << (nr * 8 + nf)
        KNIGHT_ATTACKS[sq] = np.uint64(n)

        k = 0
        for dr, df in KING_DELTAS:
            nr, nf = r + dr, f + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                k |= 1 << (nr * 8 + nf)
        KING_ATTACKS[sq] = np.uint64(k)

        wp = 0
        bp = 0
        if r < 7:
            if f > 0:
                wp |= 1 << ((r + 1) * 8 + (f - 1))
            if f < 7:
                wp |= 1 << ((r + 1) * 8 + (f + 1))
        if r > 0:
            if f > 0:
                bp |= 1 << ((r - 1) * 8 + (f - 1))
            if f < 7:
                bp |= 1 << ((r - 1) * 8 + (f + 1))
        WHITE_PAWN_ATTACKS[sq] = np.uint64(wp)
        BLACK_PAWN_ATTACKS[sq] = np.uint64(bp)


_precompute()


def knight_attacks(sq: int) -> int:
    return int(KNIGHT_ATTACKS[sq])


def king_attacks(sq: int) -> int:
    return int(KING_ATTACKS[sq])


def pawn_attacks(sq: int, is_white: bool) -> int:
    return int(WHITE_PAWN_ATTACKS[sq] if is_white else BLACK_PAWN_ATTACKS[sq])


def bishop_attacks(sq: int, occ: int) -> int:
    mask = int(BISHOP_MASKS[sq])
    magic = int(BISHOP_MAGICS[sq])
    shift = int(BISHOP_SHIFTS[sq])
    idx = ((int(occ) & mask) * magic & MASK64) >> shift
    return int(BISHOP_ATTACKS[sq, idx])


def rook_attacks(sq: int, occ: int) -> int:
    mask = int(ROOK_MASKS[sq])
    magic = int(ROOK_MAGICS[sq])
    shift = int(ROOK_SHIFTS[sq])
    idx = ((int(occ) & mask) * magic & MASK64) >> shift
    return int(ROOK_ATTACKS[sq, idx])


def queen_attacks(sq: int, occ: int) -> int:
    return bishop_attacks(sq, occ) | rook_attacks(sq, occ)
