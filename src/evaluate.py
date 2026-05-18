"""Single-pass int32 evaluation over the ``Position`` bitboards.

Returns a score from the side-to-move's perspective (negamax convention):
positive numbers mean the side to move is winning.
"""

from __future__ import annotations

import numpy as np

from constants import (
    WHITE, BLACK,
    WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK,
    FILE_A, FILE_H, NOT_FILE_A, NOT_FILE_H, MASK64,
)
from attacks import knight_attacks, king_attacks, pawn_attacks, bishop_attacks, rook_attacks, queen_attacks
from position import Position


# ---- Material values ----
# (mg, eg) — midgame and endgame piece values.
MG_VAL = (82, 337, 365, 477, 1025, 0)  # P, N, B, R, Q, K
EG_VAL = (94, 281, 297, 512, 936, 0)

# Phase weights (PeSTO-style): pawn=0, knight=1, bishop=1, rook=2, queen=4, king=0.
PHASE_VAL = (0, 1, 1, 2, 4, 0)
PHASE_MAX = 24


# ---- Piece-square tables (PeSTO values, mid+endgame), from white's POV
# These are 8×8 arrays indexed [rank, file] where rank 0 is white's first rank.
# Source: Ronald Friederich's PeSTO eval (public domain).

_PAWN_MG = [
      0,   0,   0,   0,   0,   0,   0,   0,
    -35,  -1, -20, -23, -15,  24,  38, -22,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -14,  13,   6,  21,  23,  12,  17, -23,
     -6,   7,  26,  31,  65,  56,  25, -20,
     98, 134,  61,  95,  68, 126,  34, -11,
      0,   0,   0,   0,   0,   0,   0,   0,
]
_PAWN_EG = [
      0,   0,   0,   0,   0,   0,   0,   0,
     13,   8,   8,  10,  13,   0,   2,  -7,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
     32,  24,  13,   5,  -2,   4,  17,  17,
     94, 100,  85,  67,  56,  53,  82,  84,
    178, 173, 158, 134, 147, 132, 165, 187,
      0,   0,   0,   0,   0,   0,   0,   0,
]
_KNIGHT_MG = [
   -105, -21, -58, -33, -17, -28, -19,  -23,
    -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -23,  -9,  12,  10,  19,  17,  25,  -16,
    -13,   4,  16,  13,  28,  19,  21,   -8,
     -9,  17,  19,  53,  37,  69,  18,   22,
    -47,  60,  37,  65,  84, 129,  73,   44,
    -73, -41,  72,  36,  23,  62,   7,  -17,
   -167, -89, -34, -49,  61, -97, -15, -107,
]
_KNIGHT_EG = [
    -29, -51, -23, -15, -22, -18, -50, -64,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -58, -38, -13, -28, -31, -27, -63, -99,
]
_BISHOP_MG = [
    -33,  -3, -14, -21, -13, -12, -39, -21,
      4,  15,  16,   0,   7,  21,  33,   1,
      0,  15,  15,  15,  14,  27,  18,  10,
     -6,  13,  13,  26,  34,  12,  10,   4,
     -4,   5,  19,  50,  37,  37,   7,  -2,
    -16,  37,  43,  40,  35,  50,  37,  -2,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -29,   4, -82, -37, -25, -42,   7,  -8,
]
_BISHOP_EG = [
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
     -3,   9,  12,   9,  14,  10,   3,   2,
      2,  -8,   0,  -1,  -2,   6,   0,   4,
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
]
_ROOK_MG = [
    -19, -13,   1,  17,  16,   7, -37, -26,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -24, -11,   7,  26,  24,  35,  -8, -20,
     -5,  19,  26,  36,  17,  45,  61,  16,
     27,  32,  58,  62,  80,  67,  26,  44,
     32,  42,  32,  51,  63,   9,  31,  43,
]
_ROOK_EG = [
     -9,   2,   3,  -1,  -5, -13,   4, -20,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
      4,   3,  13,   1,   2,   1,  -1,   2,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
     11,  13,  13,  11,  -3,   3,   8,   3,
     13,  10,  18,  15,  12,  12,   8,   5,
]
_QUEEN_MG = [
     -1, -18,  -9,  10, -15, -25, -31, -50,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -28,   0,  29,  12,  59,  44,  43,  45,
]
_QUEEN_EG = [
    -33, -28, -22, -43,  -5, -32, -20, -41,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -16, -27,  15,   6,   9,  17,  10,   5,
     -18,  28,  19,  47,  31,  34,  39,  23,
      3,  22,  24,  45,  57,  40,  57,  36,
    -20,   6,   9,  49,  47,  35,  19,   9,
    -17,  20,  32,  41,  58,  25,  30,   0,
     -9,  22,  22,  27,  27,  19,  10,  20,
]
_KING_MG = [
    -15,  36,  12, -54,   8, -28,  24,  14,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -14, -14, -22, -46, -44, -30, -15, -27,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -17, -20, -12, -27, -30, -25, -14, -36,
     -9,  24,   2, -16, -20,   6,  22, -22,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
    -65,  23,  16, -15, -56, -34,   2,  13,
]
_KING_EG = [
    -53, -34, -21, -11, -28, -14, -24, -43,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -18,  -4,  21,  24,  27,  23,   9, -11,
     -8,  22,  24,  27,  26,  33,  26,   3,
     10,  17,  23,  15,  20,  45,  44,  13,
    -12,  17,  14,  17,  17,  38,  23,  11,
    -74, -35, -18, -18, -11,  15,   4, -17,
]


def _mk_pst(table: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Convert one (rank-major, white POV) PST into two int32[64] views:
    one for white (no flip), one for black (vertically mirrored).

    Indexing: pst[sq] where sq is the canonical 0..63 square index (a1=0).

    The raw PeSTO tables are also rank-major but with rank 0 = white's first
    rank (a1) at the *start* of the list — exactly the same layout as our
    square index. So no flip is needed for white. For black, mirror vertically.
    """
    arr = np.array(table, dtype=np.int32)
    white = arr.copy()
    black = np.empty(64, dtype=np.int32)
    for sq in range(64):
        # Black uses the table flipped vertically so the table reads from
        # black's POV — i.e. black rank 8 (sq 56..63) uses what was white
        # rank 1 (table entries 0..7).
        r, f = sq >> 3, sq & 7
        black[sq] = arr[(7 - r) * 8 + f]
    return white, black


PST_MG_W = np.zeros((6, 64), dtype=np.int32)
PST_MG_B = np.zeros((6, 64), dtype=np.int32)
PST_EG_W = np.zeros((6, 64), dtype=np.int32)
PST_EG_B = np.zeros((6, 64), dtype=np.int32)

for i, (mg, eg) in enumerate([
    (_PAWN_MG, _PAWN_EG),
    (_KNIGHT_MG, _KNIGHT_EG),
    (_BISHOP_MG, _BISHOP_EG),
    (_ROOK_MG, _ROOK_EG),
    (_QUEEN_MG, _QUEEN_EG),
    (_KING_MG, _KING_EG),
]):
    PST_MG_W[i], PST_MG_B[i] = _mk_pst(mg)
    PST_EG_W[i], PST_EG_B[i] = _mk_pst(eg)


# ---- Misc bonuses ----
BISHOP_PAIR_MG = 30
BISHOP_PAIR_EG = 50
TEMPO = 10


def evaluate(pos: Position) -> int:
    """Return centipawn evaluation from the side-to-move's perspective."""
    from kernels import evaluate_k, _fill_psts, _PST_MG_W
    if int(_PST_MG_W.sum()) == 0:
        _fill_psts()
    return int(evaluate_k(pos.pieces, pos.occ[0], pos.occ[1], pos.occ[2], int(pos.state[0])))
