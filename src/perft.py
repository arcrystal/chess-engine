"""Perft for the ``Position`` engine, plus a python-chess lockstep validator."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from movegen import generate_legal
from move import to_uci, from_uci
from position import Position
import kernels


_PERFT_SCRATCH = np.zeros((64, 256), dtype=np.uint16)


def perft(pos: Position, depth: int) -> int:
    """Fast path: hand the entire walk to the numba kernel.

    The kernel mutates copies of the position arrays so the caller's
    ``Position`` is left untouched.
    """
    if depth <= 0:
        return 1
    return int(kernels.perft_k(
        pos.pieces.copy(),
        pos.occ.copy(),
        pos.piece_on_square.copy(),
        pos.state.copy(),
        depth,
        _PERFT_SCRATCH,
    ))


def perft_python(pos: Position, depth: int) -> int:
    """Pure-Python reference, kept for cross-checking the numba kernel."""
    if depth == 0:
        return 1
    moves = generate_legal(pos)
    if depth == 1:
        return len(moves)
    n = 0
    for mv in moves:
        pos.make_move(mv)
        n += perft_python(pos, depth - 1)
        pos.undo_move()
    return n


def perft_divide(pos: Position, depth: int) -> dict[str, int]:
    """Return {uci_move: subtree_node_count} at the given depth."""
    result: dict[str, int] = {}
    for mv in generate_legal(pos):
        pos.make_move(mv)
        result[to_uci(mv)] = perft(pos, depth - 1) if depth > 1 else 1
        pos.undo_move()
    return result


# ---- Lockstep against python-chess ----

def lockstep_compare(pos: Position, ref_board, depth: int) -> None:
    """Walk both engines, asserting move-set equality at every node.

    Raises ``AssertionError`` with diagnostic info on the first mismatch.
    """
    if depth == 0:
        return
    our = sorted(to_uci(m) for m in generate_legal(pos))
    ref = sorted(m.uci() for m in ref_board.legal_moves)
    if our != ref:
        only_ours = set(our) - set(ref)
        only_ref = set(ref) - set(our)
        raise AssertionError(
            f"move mismatch at fen={ref_board.fen()}\n"
            f"  only ours: {sorted(only_ours)}\n"
            f"  only ref:  {sorted(only_ref)}"
        )
    if depth == 1:
        return
    for uci in our:
        mv = from_uci(uci, generate_legal(pos))
        pos.make_move(mv)
        ref_board.push_uci(uci)
        lockstep_compare(pos, ref_board, depth - 1)
        pos.undo_move()
        ref_board.pop()
