"""Array-surface ``GameState``: a thin wrapper around ``Position``.

For backwards compatibility with the previous (8×8 ``int8`` board) API used
by ``tests/`` and ``evaluate_board``. The board is row-major with row 0
corresponding to rank 8 (so ``board[0]`` is the black back rank in the
starting position). Pieces are encoded as signed integers: positive =
white, negative = black, magnitude = piece type (``PAWN=1`` … ``KING=6``).

The Position core lives in ``position.py`` and is materialised on demand
when callers ask for legal moves, evaluation, or move application. Tests
that *modify* ``gs.board`` directly are supported: ``sync_to_position``
will rebuild a Position from the current ``board`` array.
"""

from __future__ import annotations

import numpy as np

from constants import (
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, EMPTY,
    WHITE, BLACK,
    WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK,
    CR_WK, CR_WQ, CR_BK, CR_BQ, CR_ALL,
)
from position import (
    Position,
    STATE_SIDE, STATE_EP, STATE_CASTLING, STATE_HALFMOVE, STATE_FULLMOVE,
)


_SIGNED_TO_PI = {
    PAWN: WP, KNIGHT: WN, BISHOP: WB, ROOK: WR, QUEEN: WQ, KING: WK,
    -PAWN: BP, -KNIGHT: BN, -BISHOP: BB, -ROOK: BR, -QUEEN: BQ, -KING: BK,
}
_PI_TO_SIGNED = {v: k for k, v in _SIGNED_TO_PI.items()}


def sq_from_rc(row: int, col: int) -> int:
    """Old (row=0 is rank 8) → square index (a1=0)."""
    return (7 - row) * 8 + col


def rc_from_sq(sq: int) -> tuple[int, int]:
    return 7 - (sq >> 3), sq & 7


class GameState:
    """Thin wrapper exposing the legacy ``board[8,8]`` representation.

    Internally maintains a ``Position`` synchronised on demand. ``sync()``
    rebuilds the Position from the array fields if they may have been
    modified externally.
    """

    __slots__ = (
        "board",
        "white_to_move",
        "castling_rights",
        "en_passant_target",
        "halfmove_clock",
        "fullmove_number",
        "_pos",
        "_dirty",
    )

    def __init__(self) -> None:
        self.board = np.zeros((8, 8), dtype=np.int8)
        self.white_to_move = True
        self.castling_rights = np.ones(4, dtype=np.int8)
        self.en_passant_target = np.array([-1, -1], dtype=np.int8)
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self._pos = Position()
        self._dirty = False
        self.reset()

    def reset(self) -> None:
        # Initial board: row 0 = rank 8 (black side).
        self.board[:] = 0
        for c in range(8):
            self.board[1, c] = -PAWN
            self.board[6, c] = PAWN
        back = [ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK]
        for c, p in enumerate(back):
            self.board[0, c] = -p
            self.board[7, c] = p
        self.white_to_move = True
        self.castling_rights[:] = 1
        self.en_passant_target[:] = -1
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self._pos = Position()
        self._dirty = False

    def switch_turn(self) -> None:
        self.white_to_move = not self.white_to_move
        if self.white_to_move:
            self.fullmove_number += 1
        self._dirty = True

    # ----- Position sync -----

    def mark_dirty(self) -> None:
        self._dirty = True

    def sync(self) -> Position:
        """Always rebuild ``_pos`` from the array fields.

        Direct writes to ``self.board`` (or the other fields) bypass any
        dirty flag we could maintain, so the safe behaviour is to rebuild
        on every call. This is fast enough for tests; the hot search path
        stays on ``Position`` directly.
        """
        p = Position()
        p.clear()
        for r in range(8):
            for c in range(8):
                v = int(self.board[r, c])
                if v == 0:
                    continue
                pi = _SIGNED_TO_PI[v]
                sq = sq_from_rc(r, c)
                p.pieces[pi] |= np.uint64(1 << sq)
        p.state[STATE_SIDE] = WHITE if self.white_to_move else BLACK
        cr = 0
        if int(self.castling_rights[0]):
            cr |= CR_WK
        if int(self.castling_rights[1]):
            cr |= CR_WQ
        if int(self.castling_rights[2]):
            cr |= CR_BK
        if int(self.castling_rights[3]):
            cr |= CR_BQ
        p.state[STATE_CASTLING] = cr
        ep_r = int(self.en_passant_target[0])
        ep_c = int(self.en_passant_target[1])
        if ep_r < 0 or ep_c < 0:
            p.state[STATE_EP] = -1
        else:
            p.state[STATE_EP] = sq_from_rc(ep_r, ep_c)
        p.state[STATE_HALFMOVE] = int(self.halfmove_clock)
        p.state[STATE_FULLMOVE] = int(self.fullmove_number)
        p._refresh_derived()
        self._pos = p
        self._dirty = False
        return p

    def write_back(self, pos: Position) -> None:
        """Copy a ``Position`` into our array surface."""
        self.board[:] = 0
        for pi in range(12):
            bb = int(pos.pieces[pi])
            while bb:
                sq = (bb & -bb).bit_length() - 1
                r, c = rc_from_sq(sq)
                self.board[r, c] = _PI_TO_SIGNED[pi]
                bb &= bb - 1
        self.white_to_move = pos.state[STATE_SIDE] == WHITE
        cr = int(pos.state[STATE_CASTLING])
        self.castling_rights[0] = 1 if cr & CR_WK else 0
        self.castling_rights[1] = 1 if cr & CR_WQ else 0
        self.castling_rights[2] = 1 if cr & CR_BK else 0
        self.castling_rights[3] = 1 if cr & CR_BQ else 0
        ep = int(pos.state[STATE_EP])
        if ep < 0:
            self.en_passant_target[:] = -1
        else:
            r, c = rc_from_sq(ep)
            self.en_passant_target[0] = r
            self.en_passant_target[1] = c
        self.halfmove_clock = int(pos.state[STATE_HALFMOVE])
        self.fullmove_number = int(pos.state[STATE_FULLMOVE])
        self._pos = pos
        self._dirty = False
