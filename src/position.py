"""Bitboard position representation.

Data layout (all numpy arrays — designed to be hot-path / numba-friendly):

* ``pieces[12]``        uint64 — bitboards for each (color, piece-type).
                                 0..5 = white P,N,B,R,Q,K ;  6..11 = black.
* ``occ[3]``            uint64 — [white, black, all].
* ``piece_on_square[64]`` int8  — 0 = empty, else (piece_index + 1).
* ``state[6]``          int32  — see indices below.
* ``zobrist``           uint64 — incremental position hash (set by ``zobrist``).
* ``history[N]`` (list) — saved (zobrist) per ply, for repetition detection.
* ``undo[N, M]`` (list) — per-ply undo records (captured piece, ep, castling, hm, zobrist).

State indices:
    STATE_SIDE       = 0    (0=white to move, 1=black)
    STATE_EP         = 1    (square 0..63 or -1)
    STATE_CASTLING   = 2    (bitmask: WK=1, WQ=2, BK=4, BQ=8)
    STATE_HALFMOVE   = 3    (50-move-rule counter)
    STATE_FULLMOVE   = 4
    STATE_CHECK      = 5    (1 if side to move is in check, else 0; lazily filled)
"""

from __future__ import annotations

import numpy as np

from constants import (
    WHITE, BLACK,
    WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK, NUM_PIECES,
    CR_WK, CR_WQ, CR_BK, CR_BQ, CR_ALL,
    FLAG_NORMAL, FLAG_CASTLE, FLAG_EP, FLAG_PROMOTION,
    PROMO_KNIGHT, PROMO_BISHOP, PROMO_ROOK, PROMO_QUEEN,
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    A1, H1, A8, H8, E1, E8, C1, G1, C8, G8, D1, F1, D8, F8,
    MASK64,
)
from move import from_sq, to_sq, promo, flag
from zobrist import Z_PIECES, Z_CASTLING, Z_EP, Z_SIDE

STATE_SIDE = 0
STATE_EP = 1
STATE_CASTLING = 2
STATE_HALFMOVE = 3
STATE_FULLMOVE = 4
STATE_CHECK = 5
STATE_LEN = 6

# Piece-index → (color, piece-type) and back.
_PT_OF_PI = (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING) * 2  # piece-type for piece-index 0..11
_COLOR_OF_PI = (WHITE,) * 6 + (BLACK,) * 6  # color for piece-index 0..11

# Bit-fiddle helpers.
def _bit(sq: int) -> int:
    return 1 << sq


_STARTING_PIECES = (
    0x000000000000FF00,  # WP
    0x0000000000000042,  # WN
    0x0000000000000024,  # WB
    0x0000000000000081,  # WR
    0x0000000000000008,  # WQ
    0x0000000000000010,  # WK
    0x00FF000000000000,  # BP
    0x4200000000000000,  # BN
    0x2400000000000000,  # BB
    0x8100000000000000,  # BR
    0x0800000000000000,  # BQ
    0x1000000000000000,  # BK
)

# Pre-mapping: when a rook moves *to* or *from* (a1, h1, a8, h8) we clear the
# corresponding castling bit.  This is per square — every move that touches the
# square (moving from it, or capturing onto it) clears the relevant bit(s).
_CASTLING_CLEAR = np.full(64, CR_ALL, dtype=np.int32)
_CASTLING_CLEAR[A1] = CR_ALL & ~CR_WQ
_CASTLING_CLEAR[H1] = CR_ALL & ~CR_WK
_CASTLING_CLEAR[A8] = CR_ALL & ~CR_BQ
_CASTLING_CLEAR[H8] = CR_ALL & ~CR_BK
_CASTLING_CLEAR[E1] = CR_ALL & ~(CR_WK | CR_WQ)
_CASTLING_CLEAR[E8] = CR_ALL & ~(CR_BK | CR_BQ)


class Position:
    """Mutable chess position. Designed for make/unmake search."""

    __slots__ = ("pieces", "occ", "piece_on_square", "state", "zobrist", "_undo", "_history")

    def __init__(self) -> None:
        self.pieces = np.zeros(NUM_PIECES, dtype=np.uint64)
        self.occ = np.zeros(3, dtype=np.uint64)
        self.piece_on_square = np.zeros(64, dtype=np.int8)
        self.state = np.zeros(STATE_LEN, dtype=np.int32)
        self.zobrist = np.uint64(0)
        self._undo: list[tuple] = []
        self._history: list[int] = []
        self.set_startpos()

    # ----- Construction helpers -----

    def set_startpos(self) -> None:
        self.pieces[:] = 0
        for pi in range(NUM_PIECES):
            self.pieces[pi] = np.uint64(_STARTING_PIECES[pi])
        self.state[STATE_SIDE] = WHITE
        self.state[STATE_EP] = -1
        self.state[STATE_CASTLING] = CR_ALL
        self.state[STATE_HALFMOVE] = 0
        self.state[STATE_FULLMOVE] = 1
        self.state[STATE_CHECK] = 0
        self._refresh_derived()
        self._undo.clear()
        self._history.clear()
        # Zobrist filled by zobrist.compute(...) when that module exists.
        self.zobrist = np.uint64(0)

    def clear(self) -> None:
        self.pieces[:] = 0
        self.state[:] = 0
        self.state[STATE_EP] = -1
        self.state[STATE_FULLMOVE] = 1
        self._refresh_derived()
        self.zobrist = np.uint64(0)
        self._undo.clear()
        self._history.clear()

    def _refresh_derived(self) -> None:
        white = np.uint64(0)
        for pi in range(0, 6):
            white |= self.pieces[pi]
        black = np.uint64(0)
        for pi in range(6, 12):
            black |= self.pieces[pi]
        self.occ[0] = white
        self.occ[1] = black
        self.occ[2] = white | black
        self.piece_on_square[:] = 0
        for pi in range(NUM_PIECES):
            bb = int(self.pieces[pi])
            while bb:
                lsb = bb & -bb
                sq = lsb.bit_length() - 1
                self.piece_on_square[sq] = pi + 1
                bb &= bb - 1
        # Recompute Zobrist hash.
        h = np.uint64(0)
        for pi in range(NUM_PIECES):
            bb = int(self.pieces[pi])
            while bb:
                sq = (bb & -bb).bit_length() - 1
                h ^= Z_PIECES[pi, sq]
                bb &= bb - 1
        h ^= Z_CASTLING[int(self.state[STATE_CASTLING]) & 0xF]
        ep = int(self.state[STATE_EP])
        if ep >= 0:
            h ^= Z_EP[ep & 7]
        if self.state[STATE_SIDE] == BLACK:
            h ^= Z_SIDE
        self.zobrist = h

    # ----- Accessors -----

    @property
    def side_to_move(self) -> int:
        return int(self.state[STATE_SIDE])

    @property
    def white_to_move(self) -> bool:
        return self.state[STATE_SIDE] == WHITE

    @property
    def ep_square(self) -> int:
        return int(self.state[STATE_EP])

    @property
    def castling(self) -> int:
        return int(self.state[STATE_CASTLING])

    @property
    def halfmove(self) -> int:
        return int(self.state[STATE_HALFMOVE])

    @property
    def fullmove(self) -> int:
        return int(self.state[STATE_FULLMOVE])

    def piece_at(self, sq: int) -> int:
        """Return piece-index (0..11) at ``sq`` or -1 if empty."""
        v = int(self.piece_on_square[sq])
        return v - 1 if v else -1

    # ----- Move application -----

    def make_move(self, mv: int) -> None:
        import kernels  # local to avoid top-level circular import at module load
        prev_ep = int(self.state[STATE_EP])
        prev_castling = int(self.state[STATE_CASTLING])
        prev_halfmove = int(self.state[STATE_HALFMOVE])
        prev_fullmove = int(self.state[STATE_FULLMOVE])
        prev_zobrist = np.uint64(self.zobrist)
        captured_pi, captured_sq, new_zobrist = kernels.make_move_k(
            self.pieces, self.occ, self.piece_on_square, self.state,
            prev_zobrist, np.uint16(mv),
        )
        self.zobrist = np.uint64(new_zobrist)
        self._undo.append((
            int(mv), int(captured_pi), int(captured_sq),
            prev_ep, prev_castling, prev_halfmove, prev_fullmove,
            prev_zobrist,
        ))
        self._history.append(int(prev_zobrist))

    def undo_move(self) -> None:
        import kernels
        if not self._undo:
            raise RuntimeError("undo_move called with empty stack")
        (
            mv, captured_pi, captured_sq,
            prev_ep, prev_castling, prev_halfmove, prev_fullmove,
            prev_zobrist,
        ) = self._undo.pop()
        if self._history:
            self._history.pop()
        kernels.undo_move_k(
            self.pieces, self.occ, self.piece_on_square, self.state, np.uint16(mv),
            np.int32(captured_pi), np.int32(captured_sq),
            np.int32(prev_ep), np.int32(prev_castling),
            np.int32(prev_halfmove), np.int32(prev_fullmove),
        )
        self.state[STATE_CHECK] = 0
        self.zobrist = prev_zobrist

    # ----- Cloning -----

    def copy(self) -> "Position":
        p = Position.__new__(Position)
        p.pieces = self.pieces.copy()
        p.occ = self.occ.copy()
        p.piece_on_square = self.piece_on_square.copy()
        p.state = self.state.copy()
        p.zobrist = self.zobrist
        p._undo = list(self._undo)
        p._history = list(self._history)
        return p

    # ----- Display -----

    def board_ascii(self) -> str:
        glyph = ".PNBRQKpnbrqk"
        lines = []
        for r in range(7, -1, -1):
            row = []
            for f in range(8):
                pi = int(self.piece_on_square[r * 8 + f])
                row.append(glyph[pi])
            lines.append(" ".join(row))
        return "\n".join(lines)
