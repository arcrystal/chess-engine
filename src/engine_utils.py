"""Adapter exposing the array-path API used by older tests and ``engine.py``.

The previous code base imported:
    KNIGHT_MOVES, BISHOP_MOVES, ROOK_MOVES, QUEEN_MOVES  # int8[N,2] deltas
    generate_legal_moves(gs) -> list[(int8,int8,int8,int8,int8)]
    apply_move(gs, move) -> None
    undo_move(gs, move, *captured_info) -> None

This module re-creates that surface on top of the bitboard engine.
Move tuples are ``(from_row, from_col, to_row, to_col, promotion)`` where
row 0 is rank 8 and ``promotion`` is the piece constant (``KNIGHT``, …,
``QUEEN``) or 0.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from constants import (
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    FLAG_NORMAL, FLAG_CASTLE, FLAG_EP, FLAG_PROMOTION,
    PROMO_KNIGHT, PROMO_BISHOP, PROMO_ROOK, PROMO_QUEEN,
)
from move import encode, from_sq, to_sq, promo as promo_field, flag
from movegen import generate_legal
from game import GameState, sq_from_rc, rc_from_sq

# Delta tables expected by the legacy evaluator.
KNIGHT_MOVES = np.array(
    [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)],
    dtype=np.int8,
)
KING_MOVES = np.array(
    [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)],
    dtype=np.int8,
)
BISHOP_MOVES = np.array([(-1, -1), (-1, 1), (1, -1), (1, 1)], dtype=np.int8)
ROOK_MOVES = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)], dtype=np.int8)
QUEEN_MOVES = np.array(
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
    dtype=np.int8,
)


Move5 = Tuple[np.int8, np.int8, np.int8, np.int8, np.int8]

_PROMO_TO_PIECE = (KNIGHT, BISHOP, ROOK, QUEEN)
_PIECE_TO_PROMO = {KNIGHT: PROMO_KNIGHT, BISHOP: PROMO_BISHOP, ROOK: PROMO_ROOK, QUEEN: PROMO_QUEEN}


def _packed_to_tuple(mv: int) -> Move5:
    f = from_sq(mv)
    t = to_sq(mv)
    fr, fc = rc_from_sq(f)
    tr, tc = rc_from_sq(t)
    if flag(mv) == FLAG_PROMOTION:
        p = _PROMO_TO_PIECE[promo_field(mv)]
    else:
        p = 0
    return (np.int8(fr), np.int8(fc), np.int8(tr), np.int8(tc), np.int8(p))


def _tuple_to_packed_candidates(t: Move5):
    fr, fc, tr, tc, p = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4])
    f = sq_from_rc(fr, fc)
    s = sq_from_rc(tr, tc)
    # Try normal, ep, castle, promotion variants.
    out = []
    if p == 0:
        out.append(encode(f, s, 0, FLAG_NORMAL))
        out.append(encode(f, s, 0, FLAG_CASTLE))
        out.append(encode(f, s, 0, FLAG_EP))
    else:
        out.append(encode(f, s, _PIECE_TO_PROMO[p], FLAG_PROMOTION))
    return out


def generate_legal_moves(gs: GameState) -> List[Move5]:
    pos = gs.sync()
    legal = generate_legal(pos)
    return [_packed_to_tuple(m) for m in legal]


def apply_move(gs: GameState, move: Move5) -> None:
    pos = gs.sync()
    legal = generate_legal(pos)
    candidates = set(_tuple_to_packed_candidates(move))
    chosen = 0
    for m in legal:
        if m in candidates:
            chosen = m
            break
        # Allow caller to omit promotion flag when promoting (older tests use 0)
        fr, fc, tr, tc, p = move
        f = sq_from_rc(int(fr), int(fc))
        s = sq_from_rc(int(tr), int(tc))
        if from_sq(m) == f and to_sq(m) == s and int(p) == 0:
            chosen = m
            break
    if chosen == 0:
        raise ValueError(f"illegal move {move}")
    pos.make_move(chosen)
    gs.write_back(pos)


def undo_move(gs: GameState, move=None, *captured_info) -> None:
    pos = gs.sync()
    if not pos._undo:
        raise RuntimeError("undo_move with empty stack")
    pos.undo_move()
    gs.write_back(pos)
