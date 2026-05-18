"""Zobrist invariants."""

from __future__ import annotations

import random

import pytest

from position import Position
from movegen import generate_legal
from move import from_uci


def test_make_unmake_preserves_hash():
    random.seed(0xC0FFEE)
    pos = Position()
    for _ in range(5000):
        mvs = generate_legal(pos)
        if not mvs:
            pos = Position()
            continue
        h0 = int(pos.zobrist)
        mv = random.choice(mvs)
        pos.make_move(mv)
        pos.undo_move()
        assert int(pos.zobrist) == h0, "make+unmake changed hash"


def test_transposition_via_knight_moves():
    pa = Position()
    pb = Position()
    for u in ("g1f3", "g8f6", "b1c3", "b8c6"):
        pa.make_move(from_uci(u, generate_legal(pa)))
    for u in ("b1c3", "b8c6", "g1f3", "g8f6"):
        pb.make_move(from_uci(u, generate_legal(pb)))
    assert int(pa.zobrist) == int(pb.zobrist)


def test_different_positions_distinct():
    p = Position()
    h0 = int(p.zobrist)
    p.make_move(from_uci("e2e4", generate_legal(p)))
    assert int(p.zobrist) != h0
