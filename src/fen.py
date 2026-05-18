"""FEN parsing and serialisation for ``Position``."""

from __future__ import annotations

import numpy as np

from constants import (
    WHITE, BLACK,
    WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK,
    CR_WK, CR_WQ, CR_BK, CR_BQ, CR_ALL,
    MASK64,
)
from position import (
    Position,
    STATE_SIDE, STATE_EP, STATE_CASTLING, STATE_HALFMOVE, STATE_FULLMOVE, STATE_CHECK,
)

_CHAR_TO_PI = {
    "P": WP, "N": WN, "B": WB, "R": WR, "Q": WQ, "K": WK,
    "p": BP, "n": BN, "b": BB, "r": BR, "q": BQ, "k": BK,
}
_PI_TO_CHAR = {v: k for k, v in _CHAR_TO_PI.items()}


def parse_fen(fen: str) -> Position:
    pos = Position()
    pos.clear()
    parts = fen.strip().split()
    if len(parts) < 4:
        raise ValueError(f"bad FEN (need ≥4 fields): {fen!r}")
    board, side, castling, ep = parts[0], parts[1], parts[2], parts[3]
    halfmove = parts[4] if len(parts) > 4 else "0"
    fullmove = parts[5] if len(parts) > 5 else "1"

    ranks = board.split("/")
    if len(ranks) != 8:
        raise ValueError(f"bad FEN board: {board!r}")
    for rank_idx, row in enumerate(ranks):
        r = 7 - rank_idx  # FEN ranks come top (rank 8) first
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
            elif ch in _CHAR_TO_PI:
                pi = _CHAR_TO_PI[ch]
                sq = r * 8 + f
                pos.pieces[pi] |= np.uint64(1 << sq)
                f += 1
            else:
                raise ValueError(f"bad piece char {ch!r}")
        if f != 8:
            raise ValueError(f"bad rank width: {row!r}")

    pos.state[STATE_SIDE] = WHITE if side == "w" else BLACK
    cr = 0
    if "K" in castling:
        cr |= CR_WK
    if "Q" in castling:
        cr |= CR_WQ
    if "k" in castling:
        cr |= CR_BK
    if "q" in castling:
        cr |= CR_BQ
    pos.state[STATE_CASTLING] = cr

    if ep == "-":
        pos.state[STATE_EP] = -1
    else:
        f_ch, r_ch = ep[0], ep[1]
        pos.state[STATE_EP] = (ord(f_ch) - ord("a")) + (int(r_ch) - 1) * 8

    pos.state[STATE_HALFMOVE] = int(halfmove)
    pos.state[STATE_FULLMOVE] = int(fullmove)
    pos.state[STATE_CHECK] = 0
    pos._refresh_derived()
    pos._undo.clear()
    pos._history.clear()
    pos.zobrist = np.uint64(0)
    return pos


def to_fen(pos: Position) -> str:
    rows = []
    for r in range(7, -1, -1):
        row = ""
        empty = 0
        for f in range(8):
            sq = r * 8 + f
            pi = int(pos.piece_on_square[sq]) - 1
            if pi < 0:
                empty += 1
            else:
                if empty:
                    row += str(empty)
                    empty = 0
                row += _PI_TO_CHAR[pi]
        if empty:
            row += str(empty)
        rows.append(row)
    board = "/".join(rows)
    side = "w" if pos.state[STATE_SIDE] == WHITE else "b"
    cr = int(pos.state[STATE_CASTLING])
    cstr = ""
    if cr & CR_WK:
        cstr += "K"
    if cr & CR_WQ:
        cstr += "Q"
    if cr & CR_BK:
        cstr += "k"
    if cr & CR_BQ:
        cstr += "q"
    if not cstr:
        cstr = "-"
    ep_sq = int(pos.state[STATE_EP])
    if ep_sq < 0:
        ep = "-"
    else:
        ep = f"{chr((ep_sq & 7) + ord('a'))}{(ep_sq >> 3) + 1}"
    return f"{board} {side} {cstr} {ep} {int(pos.state[STATE_HALFMOVE])} {int(pos.state[STATE_FULLMOVE])}"
