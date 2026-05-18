"""16-bit packed move encoding.

Bit layout (LSB first):
  bits 0..5   from square (0..63)
  bits 6..11  to square   (0..63)
  bits 12..13 promotion piece (0=N, 1=B, 2=R, 3=Q) — only meaningful when flag == PROMOTION
  bits 14..15 flag (0=NORMAL, 1=CASTLE, 2=EP, 3=PROMOTION)
"""

from constants import (
    FLAG_NORMAL, FLAG_CASTLE, FLAG_EP, FLAG_PROMOTION,
    PROMO_KNIGHT, PROMO_BISHOP, PROMO_ROOK, PROMO_QUEEN,
    KNIGHT, BISHOP, ROOK, QUEEN,
)

NULL_MOVE = 0


def encode(from_sq: int, to_sq: int, promo: int = 0, flag: int = FLAG_NORMAL) -> int:
    return (from_sq & 0x3F) | ((to_sq & 0x3F) << 6) | ((promo & 0x3) << 12) | ((flag & 0x3) << 14)


def from_sq(mv: int) -> int:
    return mv & 0x3F


def to_sq(mv: int) -> int:
    return (mv >> 6) & 0x3F


def promo(mv: int) -> int:
    return (mv >> 12) & 0x3


def flag(mv: int) -> int:
    return (mv >> 14) & 0x3


_PROMO_TO_PIECE = (KNIGHT, BISHOP, ROOK, QUEEN)
_PIECE_TO_PROMO = {KNIGHT: PROMO_KNIGHT, BISHOP: PROMO_BISHOP, ROOK: PROMO_ROOK, QUEEN: PROMO_QUEEN}


def promo_piece_type(mv: int) -> int:
    """Return the piece *type* (KNIGHT/BISHOP/ROOK/QUEEN) being promoted to."""
    return _PROMO_TO_PIECE[promo(mv)]


def piece_to_promo(piece_type: int) -> int:
    return _PIECE_TO_PROMO[piece_type]


_FILE_CHARS = "abcdefgh"


def to_uci(mv: int) -> str:
    if mv == 0:
        return "0000"
    f = from_sq(mv)
    t = to_sq(mv)
    s = f"{_FILE_CHARS[f & 7]}{(f >> 3) + 1}{_FILE_CHARS[t & 7]}{(t >> 3) + 1}"
    if flag(mv) == FLAG_PROMOTION:
        s += "nbrq"[promo(mv)]
    return s


def from_uci(s: str, legal_moves) -> int:
    """Resolve a UCI string against a list of legal packed moves.

    We can't decode in isolation because UCI strings don't encode flag bits.
    """
    s = s.strip().lower()
    if s == "0000":
        return 0
    if len(s) not in (4, 5):
        raise ValueError(f"bad UCI move: {s!r}")
    f = (ord(s[0]) - ord("a")) | ((ord(s[1]) - ord("1")) << 3)
    t = (ord(s[2]) - ord("a")) | ((ord(s[3]) - ord("1")) << 3)
    promo_ch = s[4] if len(s) == 5 else None
    target_promo = "nbrq".index(promo_ch) if promo_ch else -1
    for mv in legal_moves:
        if from_sq(mv) == f and to_sq(mv) == t:
            if target_promo == -1:
                if flag(mv) != FLAG_PROMOTION:
                    return mv
            else:
                if flag(mv) == FLAG_PROMOTION and promo(mv) == target_promo:
                    return mv
    raise ValueError(f"UCI move {s!r} not in legal list")


def square_to_str(sq: int) -> str:
    return f"{_FILE_CHARS[sq & 7]}{(sq >> 3) + 1}"
