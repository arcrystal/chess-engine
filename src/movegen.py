"""Bitboard move generation for ``Position`` — produces packed 16-bit moves.

The hot paths (pseudo-legal generation, ``is_square_attacked``) live in
``kernels.py`` as ``@njit`` functions; this module is a thin Python wrapper
that exposes them in the legal-move filter loop used by tests and search.
"""

from __future__ import annotations

from typing import List

import numpy as np

from constants import (
    WHITE, BLACK,
    WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK,
    CR_WK, CR_WQ, CR_BK, CR_BQ,
    FLAG_NORMAL, FLAG_CASTLE, FLAG_EP, FLAG_PROMOTION,
    PROMO_KNIGHT, PROMO_BISHOP, PROMO_ROOK, PROMO_QUEEN,
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    A1, B1, C1, D1, E1, F1, G1, H1, A8, B8, C8, D8, E8, F8, G8, H8,
    RANK_1, RANK_2, RANK_3, RANK_5, RANK_6, RANK_7, RANK_8,
    FILE_A, FILE_H, NOT_FILE_A, NOT_FILE_H,
    MASK64,
)
from move import encode
from attacks import (
    knight_attacks, king_attacks, pawn_attacks,
    bishop_attacks, rook_attacks, queen_attacks,
)
from position import Position
import kernels


_SCRATCH = np.zeros(256, dtype=np.uint16)


# ----- Attack queries (delegate to numba kernels) -----

def is_square_attacked(pos: Position, sq: int, by_white: bool) -> bool:
    return bool(kernels.is_square_attacked_k(pos.pieces, pos.occ[2], sq, by_white))


def is_check(pos: Position, side: int | None = None) -> bool:
    if side is None:
        side = pos.side_to_move
    return bool(kernels.is_check_k(pos.pieces, pos.occ[2], side))


# ----- Pseudo-legal pawn moves -----

def _emit_pushes(out: List[int], targets: int, delta: int) -> None:
    """Emit non-promotion pushes for a set of target bits."""
    tmp = targets
    while tmp:
        t = (tmp & -tmp).bit_length() - 1
        tmp &= tmp - 1
        f = t - delta
        out.append(encode(f, t, 0, FLAG_NORMAL))


def _emit_promotions(out: List[int], targets: int, delta: int) -> None:
    tmp = targets
    while tmp:
        t = (tmp & -tmp).bit_length() - 1
        tmp &= tmp - 1
        f = t - delta
        out.append(encode(f, t, PROMO_QUEEN, FLAG_PROMOTION))
        out.append(encode(f, t, PROMO_ROOK, FLAG_PROMOTION))
        out.append(encode(f, t, PROMO_BISHOP, FLAG_PROMOTION))
        out.append(encode(f, t, PROMO_KNIGHT, FLAG_PROMOTION))


def _gen_pawn_moves(pos: Position, side: int, out: List[int]) -> None:
    occ = int(pos.occ[2])
    empty = (~occ) & MASK64
    if side == WHITE:
        pawns = int(pos.pieces[WP])
        enemy = int(pos.occ[1])
        single = (pawns << 8) & empty
        double = ((single & RANK_3) << 8) & empty
        left = ((pawns & NOT_FILE_A) << 7) & enemy
        right = ((pawns & NOT_FILE_H) << 9) & enemy
        promo_rank = RANK_8
        delta_single = 8
        delta_double = 16
        delta_left = 7
        delta_right = 9
    else:
        pawns = int(pos.pieces[BP])
        enemy = int(pos.occ[0])
        single = (pawns >> 8) & empty
        double = ((single & RANK_6) >> 8) & empty
        left = ((pawns & NOT_FILE_H) >> 7) & enemy
        right = ((pawns & NOT_FILE_A) >> 9) & enemy
        promo_rank = RANK_1
        delta_single = -8
        delta_double = -16
        delta_left = -7
        delta_right = -9

    _emit_pushes(out, single & ~promo_rank, delta_single)
    _emit_promotions(out, single & promo_rank, delta_single)
    _emit_pushes(out, double, delta_double)
    _emit_pushes(out, left & ~promo_rank, delta_left)
    _emit_promotions(out, left & promo_rank, delta_left)
    _emit_pushes(out, right & ~promo_rank, delta_right)
    _emit_promotions(out, right & promo_rank, delta_right)

    # En passant.
    ep = pos.ep_square
    if ep >= 0:
        # White pawns that attack ep == squares from which a black-pawn-attack would hit ep ==
        # BLACK_PAWN_ATTACKS[ep]. By symmetry, pawn_attacks(ep, not side) gives the squares
        # from which our side's pawns can capture ep.
        attackers = pawn_attacks(ep, side == BLACK) & pawns
        tmp = attackers
        while tmp:
            f = (tmp & -tmp).bit_length() - 1
            tmp &= tmp - 1
            out.append(encode(f, ep, 0, FLAG_EP))


# ----- Pseudo-legal piece moves -----

def _gen_piece_moves(pos: Position, side: int, out: List[int]) -> None:
    occ = int(pos.occ[2])
    own = int(pos.occ[side])
    not_own = (~own) & MASK64

    # Knights
    n_pi = WN if side == WHITE else BN
    bb = int(pos.pieces[n_pi])
    while bb:
        f = (bb & -bb).bit_length() - 1
        bb &= bb - 1
        moves = knight_attacks(f) & not_own
        while moves:
            t = (moves & -moves).bit_length() - 1
            moves &= moves - 1
            out.append(encode(f, t, 0, FLAG_NORMAL))

    # Bishops
    b_pi = WB if side == WHITE else BB
    bb = int(pos.pieces[b_pi])
    while bb:
        f = (bb & -bb).bit_length() - 1
        bb &= bb - 1
        moves = bishop_attacks(f, occ) & not_own
        while moves:
            t = (moves & -moves).bit_length() - 1
            moves &= moves - 1
            out.append(encode(f, t, 0, FLAG_NORMAL))

    # Rooks
    r_pi = WR if side == WHITE else BR
    bb = int(pos.pieces[r_pi])
    while bb:
        f = (bb & -bb).bit_length() - 1
        bb &= bb - 1
        moves = rook_attacks(f, occ) & not_own
        while moves:
            t = (moves & -moves).bit_length() - 1
            moves &= moves - 1
            out.append(encode(f, t, 0, FLAG_NORMAL))

    # Queens
    q_pi = WQ if side == WHITE else BQ
    bb = int(pos.pieces[q_pi])
    while bb:
        f = (bb & -bb).bit_length() - 1
        bb &= bb - 1
        moves = queen_attacks(f, occ) & not_own
        while moves:
            t = (moves & -moves).bit_length() - 1
            moves &= moves - 1
            out.append(encode(f, t, 0, FLAG_NORMAL))

    # King — pseudo-legal "can move to any non-own square"; legality filter handles checks.
    k_pi = WK if side == WHITE else BK
    bb = int(pos.pieces[k_pi])
    if bb:
        f = (bb & -bb).bit_length() - 1
        moves = king_attacks(f) & not_own
        while moves:
            t = (moves & -moves).bit_length() - 1
            moves &= moves - 1
            out.append(encode(f, t, 0, FLAG_NORMAL))


# ----- Castling -----

def _gen_castling(pos: Position, side: int, out: List[int]) -> None:
    occ = int(pos.occ[2])
    rights = pos.castling
    enemy_white = (side == BLACK)  # the side that attacks our king's path
    if side == WHITE:
        # King on e1 (assumed; if not, rights would have been cleared)
        if (rights & CR_WK) and not (occ & ((1 << F1) | (1 << G1))):
            if not (
                is_square_attacked(pos, E1, by_white=enemy_white)
                or is_square_attacked(pos, F1, by_white=enemy_white)
                or is_square_attacked(pos, G1, by_white=enemy_white)
            ):
                out.append(encode(E1, G1, 0, FLAG_CASTLE))
        if (rights & CR_WQ) and not (occ & ((1 << B1) | (1 << C1) | (1 << D1))):
            if not (
                is_square_attacked(pos, E1, by_white=enemy_white)
                or is_square_attacked(pos, D1, by_white=enemy_white)
                or is_square_attacked(pos, C1, by_white=enemy_white)
            ):
                out.append(encode(E1, C1, 0, FLAG_CASTLE))
    else:
        if (rights & CR_BK) and not (occ & ((1 << F8) | (1 << G8))):
            if not (
                is_square_attacked(pos, E8, by_white=enemy_white)
                or is_square_attacked(pos, F8, by_white=enemy_white)
                or is_square_attacked(pos, G8, by_white=enemy_white)
            ):
                out.append(encode(E8, G8, 0, FLAG_CASTLE))
        if (rights & CR_BQ) and not (occ & ((1 << B8) | (1 << C8) | (1 << D8))):
            if not (
                is_square_attacked(pos, E8, by_white=enemy_white)
                or is_square_attacked(pos, D8, by_white=enemy_white)
                or is_square_attacked(pos, C8, by_white=enemy_white)
            ):
                out.append(encode(E8, C8, 0, FLAG_CASTLE))


# ----- Public API -----

def generate_pseudo_legal(pos: Position) -> List[int]:
    n = kernels.generate_pseudo_legal_into(pos.pieces, pos.occ, pos.state, _SCRATCH)
    return [int(_SCRATCH[i]) for i in range(n)]


def generate_legal(pos: Position) -> List[int]:
    """Pseudo-legal then filter via make/unmake + ``is_check``."""
    side = pos.side_to_move
    legal: List[int] = []
    for mv in generate_pseudo_legal(pos):
        pos.make_move(mv)
        # The side that just moved is ``side``. If still in check, illegal.
        if not is_check(pos, side=side):
            legal.append(mv)
        pos.undo_move()
    return legal
