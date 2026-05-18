"""Exhaustive promotion correctness."""

from __future__ import annotations

import pytest

from constants import KNIGHT, BISHOP, ROOK, QUEEN
from fen import parse_fen
from movegen import generate_legal
from move import flag, promo, from_sq, to_sq, encode, to_uci
from constants import FLAG_PROMOTION


PROMO_PIECE_VALUES = [KNIGHT, BISHOP, ROOK, QUEEN]
_PROMO_TO_OFFSET = {KNIGHT: 0, BISHOP: 1, ROOK: 2, QUEEN: 3}
_PROMO_CHAR = {KNIGHT: "n", BISHOP: "b", ROOK: "r", QUEEN: "q"}


@pytest.mark.parametrize("piece", PROMO_PIECE_VALUES)
def test_white_straight_promotion(piece):
    # White pawn on a7, no obstruction.
    pos = parse_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    moves = generate_legal(pos)
    promos = [m for m in moves if flag(m) == FLAG_PROMOTION]
    targets = {(from_sq(m), to_sq(m), promo(m)) for m in promos}
    assert (48, 56, _PROMO_TO_OFFSET[piece]) in targets


@pytest.mark.parametrize("piece", PROMO_PIECE_VALUES)
def test_black_straight_promotion(piece):
    # Black pawn on a2.
    pos = parse_fen("4k3/8/8/8/8/8/p7/4K3 b - - 0 1")
    moves = generate_legal(pos)
    promos = [m for m in moves if flag(m) == FLAG_PROMOTION]
    targets = {(from_sq(m), to_sq(m), promo(m)) for m in promos}
    assert (8, 0, _PROMO_TO_OFFSET[piece]) in targets


@pytest.mark.parametrize("piece", PROMO_PIECE_VALUES)
def test_white_capture_promotion(piece):
    # White pawn on a7, black piece on b8.
    pos = parse_fen("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    moves = generate_legal(pos)
    promos = [m for m in moves if flag(m) == FLAG_PROMOTION and to_sq(m) == 57]  # b8
    assert any(promo(m) == _PROMO_TO_OFFSET[piece] for m in promos), \
        f"missing capture+promo to {_PROMO_CHAR[piece]}"


@pytest.mark.parametrize("piece", PROMO_PIECE_VALUES)
def test_black_capture_promotion(piece):
    # Black pawn on a2, white piece on b1.
    pos = parse_fen("4k3/8/8/8/8/8/p7/1N2K3 b - - 0 1")
    moves = generate_legal(pos)
    promos = [m for m in moves if flag(m) == FLAG_PROMOTION and to_sq(m) == 1]  # b1
    assert any(promo(m) == _PROMO_TO_OFFSET[piece] for m in promos)


def test_promotion_changes_piece_type():
    """After promoting to a queen, the piece-on-square is the queen, not a pawn."""
    from constants import WQ, WB, WN, WR
    pos = parse_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    mv = encode(48, 56, _PROMO_TO_OFFSET[QUEEN], FLAG_PROMOTION)
    pos.make_move(mv)
    assert int(pos.piece_on_square[56]) - 1 == WQ
    pos.undo_move()
    # After undo, a7 has a pawn again.
    from constants import WP
    assert int(pos.piece_on_square[48]) - 1 == WP

    # Same for other promotion targets.
    for piece, pi in [(KNIGHT, WN), (BISHOP, WB), (ROOK, WR)]:
        pos = parse_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        mv = encode(48, 56, _PROMO_TO_OFFSET[piece], FLAG_PROMOTION)
        pos.make_move(mv)
        assert int(pos.piece_on_square[56]) - 1 == pi, f"{piece} promo failed"
