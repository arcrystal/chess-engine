"""Perft against ``python-chess`` for the six canonical positions.

Depths kept modest for the test suite (under ~3 s total). Deeper runs
live in ``bench/`` if you want them.
"""

from __future__ import annotations

import chess
import pytest

from fen import parse_fen
from perft import perft, lockstep_compare


CASES = [
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
     [(1, 20), (2, 400), (3, 8902), (4, 197281)]),
    ("Kiwipete",  "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
     [(1, 48), (2, 2039), (3, 97862)]),
    ("Pos3",      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
     [(1, 14), (2, 191), (3, 2812), (4, 43238)]),
    ("Pos4",      "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2pP/R2Q1RK1 w kq - 0 1",
     [(1, 6), (2, 280), (3, 9346)]),
    ("Pos5",      "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
     [(1, 44), (2, 1486), (3, 62379)]),
    ("Pos6",      "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
     [(1, 46), (2, 2079), (3, 89890)]),
]


@pytest.mark.parametrize("name,fen,table", CASES, ids=[c[0] for c in CASES])
def test_perft_node_counts(name, fen, table):
    pos = parse_fen(fen)
    for depth, expected in table:
        n = perft(pos, depth)
        assert n == expected, f"{name} d{depth}: got {n} expected {expected}"


@pytest.mark.parametrize("name,fen,table", CASES, ids=[c[0] for c in CASES])
def test_lockstep_matches_python_chess(name, fen, table):
    pos = parse_fen(fen)
    ref = chess.Board(fen)
    max_d = min(3, max(d for d, _ in table))
    lockstep_compare(pos, ref, max_d)
