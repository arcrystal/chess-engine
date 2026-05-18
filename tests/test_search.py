"""Tactical sanity for the search."""

from __future__ import annotations

import pytest

from fen import parse_fen
from search import Searcher, SearchLimits
from move import to_uci
from constants import MATE_IN_MAX


def _best(fen: str, depth: int = 5) -> tuple[int, str]:
    pos = parse_fen(fen)
    s = Searcher()
    mv = s.search(pos, SearchLimits(max_depth=depth, movetime_ms=10000))
    return s.root_score, to_uci(mv)


def test_mate_in_one_scholars():
    score, mv = _best("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 2 3", depth=4)
    assert mv == "f3f7"
    assert score >= MATE_IN_MAX


def test_capture_immediate_free_piece():
    # Black queen on d6 is unprotected; white queen on d1 can take it on the d-file.
    score, mv = _best("4k3/8/3q4/8/8/8/8/3Q1K2 w - - 0 1", depth=2)
    assert mv == "d1d6", f"expected d1d6, got {mv}"
    assert score > 500, f"expected winning eval, got {score}"


def test_no_blundering_queen():
    # Starting position eval — engine should not give up material on move 1.
    from position import Position
    pos = Position()
    s = Searcher()
    s.search(pos, SearchLimits(max_depth=3, movetime_ms=10000))
    # Score shouldn't be wildly negative for white.
    assert s.root_score > -200
