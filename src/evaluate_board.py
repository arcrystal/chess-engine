"""Legacy shim: ``evaluate_board(gs) -> int`` from white's POV.

The full evaluation lives in ``evaluate.py`` and is keyed off the bitboard
``Position``. This shim just materialises a Position from a ``GameState``
and converts the side-to-move score back to the white-POV convention the
old tests assumed.
"""

from __future__ import annotations

from game import GameState
from evaluate import evaluate as _evaluate


def evaluate_board(gs: GameState) -> int:
    pos = gs.sync()
    score = _evaluate(pos)
    return score if gs.white_to_move else -score
