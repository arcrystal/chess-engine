"""End-to-end UCI conversation via ``python-chess``."""

from __future__ import annotations

import sys

import chess
import chess.engine
import pytest


PY = sys.executable


def _engine():
    return chess.engine.SimpleEngine.popen_uci([PY, "-m", "src.uci"], timeout=60)


def test_uci_handshake_and_play():
    eng = _engine()
    board = chess.Board()
    res = eng.play(board, chess.engine.Limit(depth=3))
    assert res.move is not None
    assert res.move in board.legal_moves
    eng.quit()


def test_uci_self_play_10_plies():
    eng_w = _engine()
    eng_b = _engine()
    board = chess.Board()
    for _ in range(10):
        if board.is_game_over():
            break
        eng = eng_w if board.turn == chess.WHITE else eng_b
        res = eng.play(board, chess.engine.Limit(time=0.05))
        assert res.move in board.legal_moves
        board.push(res.move)
    eng_w.quit()
    eng_b.quit()
