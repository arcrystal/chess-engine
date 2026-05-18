"""Numba-accelerated kernels for the hot path.

Public entry points:
    bishop_attacks_n(sq, occ)         -> uint64
    rook_attacks_n(sq, occ)           -> uint64
    queen_attacks_n(sq, occ)          -> uint64
    is_square_attacked_k(pieces, occ_all, sq, by_white) -> bool
    is_check_k(pieces, occ_all, side) -> bool
    generate_pseudo_legal_into(pieces, occ, state, out) -> n_moves
    make_move_k(pieces, occ, p_o_s, state, mv) -> packed prev (captured, etc.)
    undo_move_k(pieces, occ, p_o_s, state, mv, prev_*)
    perft_k(pieces, occ, p_o_s, state, depth, scratch_moves) -> int

All state arrays are operated on in-place. Inputs and outputs use
``np.uint64``/``np.int8``/``np.int32`` typing to keep numba happy.
"""

from __future__ import annotations

import numpy as np
from numba import njit, types
from numba.extending import overload

from magic import (
    BISHOP_MASKS, BISHOP_MAGICS, BISHOP_SHIFTS, BISHOP_ATTACKS,
    ROOK_MASKS, ROOK_MAGICS, ROOK_SHIFTS, ROOK_ATTACKS,
)
from attacks import KNIGHT_ATTACKS, KING_ATTACKS, WHITE_PAWN_ATTACKS, BLACK_PAWN_ATTACKS
from zobrist import Z_PIECES, Z_CASTLING, Z_EP, Z_SIDE


MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
ZERO = np.uint64(0)
ONE = np.uint64(1)

# State indices (mirror position.py).
S_SIDE = 0
S_EP = 1
S_CASTLING = 2
S_HALFMOVE = 3
S_FULLMOVE = 4

# Move flags.
F_NORMAL = 0
F_CASTLE = 1
F_EP = 2
F_PROMOTION = 3

# Castling clear-mask table (square -> bits NOT to clear).
CASTLING_CLEAR_TABLE = np.full(64, 15, dtype=np.int32)  # CR_ALL = 15
CASTLING_CLEAR_TABLE[0] = 15 & ~2   # A1: clear WQ
CASTLING_CLEAR_TABLE[7] = 15 & ~1   # H1: clear WK
CASTLING_CLEAR_TABLE[56] = 15 & ~8  # A8: clear BQ
CASTLING_CLEAR_TABLE[63] = 15 & ~4  # H8: clear BK
CASTLING_CLEAR_TABLE[4] = 15 & ~(1 | 2)   # E1: clear WK + WQ
CASTLING_CLEAR_TABLE[60] = 15 & ~(4 | 8)  # E8: clear BK + BQ

# Rank masks.
RANK_3 = np.uint64(0x00000000_00FF0000)
RANK_6 = np.uint64(0x0000FF00_00000000)
RANK_1 = np.uint64(0x00000000_000000FF)
RANK_8 = np.uint64(0xFF000000_00000000)
NOT_FILE_A = np.uint64(0xFEFEFEFE_FEFEFEFE)
NOT_FILE_H = np.uint64(0x7F7F7F7F_7F7F7F7F)


@njit(inline="always")
def bishop_attacks_n(sq, occ):
    mask = BISHOP_MASKS[sq]
    magic = BISHOP_MAGICS[sq]
    shift = BISHOP_SHIFTS[sq]
    idx = ((occ & mask) * magic) >> np.uint64(shift)
    return BISHOP_ATTACKS[sq, idx]


@njit(inline="always")
def rook_attacks_n(sq, occ):
    mask = ROOK_MASKS[sq]
    magic = ROOK_MAGICS[sq]
    shift = ROOK_SHIFTS[sq]
    idx = ((occ & mask) * magic) >> np.uint64(shift)
    return ROOK_ATTACKS[sq, idx]


@njit(inline="always")
def queen_attacks_n(sq, occ):
    return bishop_attacks_n(sq, occ) | rook_attacks_n(sq, occ)


@njit(inline="always")
def lsb_index(bb):
    """Return least-significant-set-bit index (0..63) of a non-zero ``uint64``.

    The two-halves dance avoids the bit-63 overflow that breaks the naive
    ``log2(int64(bb & -bb))`` approach.
    """
    x = bb & ((~bb) + ONE)  # uint64 LSB isolation
    if x < np.uint64(0x100000000):
        return int(np.log2(np.float64(x)))
    return 32 + int(np.log2(np.float64(x >> np.uint64(32))))


@njit(inline="always")
def popcount64(bb):
    n = 0
    x = bb
    while x != ZERO:
        x &= x - ONE
        n += 1
    return n


@njit
def is_square_attacked_k(pieces, occ_all, sq, by_white):
    if by_white:
        if (BLACK_PAWN_ATTACKS[sq] & pieces[0]) != ZERO:
            return True
        if (KNIGHT_ATTACKS[sq] & pieces[1]) != ZERO:
            return True
        if (KING_ATTACKS[sq] & pieces[5]) != ZERO:
            return True
        b = bishop_attacks_n(sq, occ_all)
        if (b & (pieces[2] | pieces[4])) != ZERO:
            return True
        r = rook_attacks_n(sq, occ_all)
        if (r & (pieces[3] | pieces[4])) != ZERO:
            return True
    else:
        if (WHITE_PAWN_ATTACKS[sq] & pieces[6]) != ZERO:
            return True
        if (KNIGHT_ATTACKS[sq] & pieces[7]) != ZERO:
            return True
        if (KING_ATTACKS[sq] & pieces[11]) != ZERO:
            return True
        b = bishop_attacks_n(sq, occ_all)
        if (b & (pieces[8] | pieces[10])) != ZERO:
            return True
        r = rook_attacks_n(sq, occ_all)
        if (r & (pieces[9] | pieces[10])) != ZERO:
            return True
    return False


@njit
def is_check_k(pieces, occ_all, side):
    king_pi = 5 if side == 0 else 11
    king_bb = pieces[king_pi]
    if king_bb == ZERO:
        return False
    sq = lsb_index(king_bb)
    return is_square_attacked_k(pieces, occ_all, sq, by_white=(side == 1))


@njit(inline="always")
def encode_move(f, t, p, fl):
    return np.uint16((f & 63) | ((t & 63) << 6) | ((p & 3) << 12) | ((fl & 3) << 14))


@njit
def gen_pawn_moves(pieces, occ_w, occ_b, occ_all, side, ep_sq, out, n):
    empty = (~occ_all) & MASK64
    if side == 0:
        pawns = pieces[0]
        enemy = occ_b
        single = (pawns << ONE * np.uint64(8)) & empty
        double = ((single & RANK_3) << ONE * np.uint64(8)) & empty
        left = ((pawns & NOT_FILE_A) << ONE * np.uint64(7)) & enemy
        right = ((pawns & NOT_FILE_H) << ONE * np.uint64(9)) & enemy
        promo_rank = RANK_8
        d_single = 8
        d_double = 16
        d_left = 7
        d_right = 9
    else:
        pawns = pieces[6]
        enemy = occ_w
        single = (pawns >> np.uint64(8)) & empty
        double = ((single & RANK_6) >> np.uint64(8)) & empty
        left = ((pawns & NOT_FILE_H) >> np.uint64(7)) & enemy
        right = ((pawns & NOT_FILE_A) >> np.uint64(9)) & enemy
        promo_rank = RANK_1
        d_single = -8
        d_double = -16
        d_left = -7
        d_right = -9

    # Single pushes (non-promo).
    tmp = single & ~promo_rank
    while tmp != ZERO:
        ts = lsb_index(tmp)
        tmp &= tmp - ONE
        f = ts - d_single
        out[n] = encode_move(f, ts, 0, F_NORMAL); n += 1
    # Single push promotions.
    tmp = single & promo_rank
    while tmp != ZERO:
        ts = lsb_index(tmp)
        tmp &= tmp - ONE
        f = ts - d_single
        out[n] = encode_move(f, ts, 3, F_PROMOTION); n += 1  # Q
        out[n] = encode_move(f, ts, 2, F_PROMOTION); n += 1  # R
        out[n] = encode_move(f, ts, 1, F_PROMOTION); n += 1  # B
        out[n] = encode_move(f, ts, 0, F_PROMOTION); n += 1  # N
    # Double pushes.
    tmp = double
    while tmp != ZERO:
        ts = lsb_index(tmp)
        tmp &= tmp - ONE
        f = ts - d_double
        out[n] = encode_move(f, ts, 0, F_NORMAL); n += 1
    # Captures left.
    tmp = left & ~promo_rank
    while tmp != ZERO:
        ts = lsb_index(tmp)
        tmp &= tmp - ONE
        f = ts - d_left
        out[n] = encode_move(f, ts, 0, F_NORMAL); n += 1
    tmp = left & promo_rank
    while tmp != ZERO:
        ts = lsb_index(tmp)
        tmp &= tmp - ONE
        f = ts - d_left
        out[n] = encode_move(f, ts, 3, F_PROMOTION); n += 1
        out[n] = encode_move(f, ts, 2, F_PROMOTION); n += 1
        out[n] = encode_move(f, ts, 1, F_PROMOTION); n += 1
        out[n] = encode_move(f, ts, 0, F_PROMOTION); n += 1
    # Captures right.
    tmp = right & ~promo_rank
    while tmp != ZERO:
        ts = lsb_index(tmp)
        tmp &= tmp - ONE
        f = ts - d_right
        out[n] = encode_move(f, ts, 0, F_NORMAL); n += 1
    tmp = right & promo_rank
    while tmp != ZERO:
        ts = lsb_index(tmp)
        tmp &= tmp - ONE
        f = ts - d_right
        out[n] = encode_move(f, ts, 3, F_PROMOTION); n += 1
        out[n] = encode_move(f, ts, 2, F_PROMOTION); n += 1
        out[n] = encode_move(f, ts, 1, F_PROMOTION); n += 1
        out[n] = encode_move(f, ts, 0, F_PROMOTION); n += 1

    # En passant.
    if ep_sq >= 0:
        if side == 0:
            attackers = BLACK_PAWN_ATTACKS[ep_sq] & pawns
        else:
            attackers = WHITE_PAWN_ATTACKS[ep_sq] & pawns
        tmp = attackers
        while tmp != ZERO:
            f = lsb_index(tmp)
            tmp &= tmp - ONE
            out[n] = encode_move(f, ep_sq, 0, F_EP); n += 1

    return n


@njit
def gen_piece_moves(pieces, occ_w, occ_b, occ_all, side, out, n):
    own = occ_w if side == 0 else occ_b
    not_own = (~own) & MASK64

    # Knights
    n_pi = 1 if side == 0 else 7
    bb = pieces[n_pi]
    while bb != ZERO:
        f = lsb_index(bb)
        bb &= bb - ONE
        moves = KNIGHT_ATTACKS[f] & not_own
        while moves != ZERO:
            t = lsb_index(moves)
            moves &= moves - ONE
            out[n] = encode_move(f, t, 0, F_NORMAL); n += 1

    # Bishops
    b_pi = 2 if side == 0 else 8
    bb = pieces[b_pi]
    while bb != ZERO:
        f = lsb_index(bb)
        bb &= bb - ONE
        moves = bishop_attacks_n(f, occ_all) & not_own
        while moves != ZERO:
            t = lsb_index(moves)
            moves &= moves - ONE
            out[n] = encode_move(f, t, 0, F_NORMAL); n += 1

    # Rooks
    r_pi = 3 if side == 0 else 9
    bb = pieces[r_pi]
    while bb != ZERO:
        f = lsb_index(bb)
        bb &= bb - ONE
        moves = rook_attacks_n(f, occ_all) & not_own
        while moves != ZERO:
            t = lsb_index(moves)
            moves &= moves - ONE
            out[n] = encode_move(f, t, 0, F_NORMAL); n += 1

    # Queens
    q_pi = 4 if side == 0 else 10
    bb = pieces[q_pi]
    while bb != ZERO:
        f = lsb_index(bb)
        bb &= bb - ONE
        moves = queen_attacks_n(f, occ_all) & not_own
        while moves != ZERO:
            t = lsb_index(moves)
            moves &= moves - ONE
            out[n] = encode_move(f, t, 0, F_NORMAL); n += 1

    # King
    k_pi = 5 if side == 0 else 11
    bb = pieces[k_pi]
    if bb != ZERO:
        f = lsb_index(bb)
        moves = KING_ATTACKS[f] & not_own
        while moves != ZERO:
            t = lsb_index(moves)
            moves &= moves - ONE
            out[n] = encode_move(f, t, 0, F_NORMAL); n += 1

    return n


@njit
def gen_castling(pieces, occ_all, state, out, n):
    side = state[S_SIDE]
    rights = state[S_CASTLING]
    enemy_white = side == 1
    if side == 0:
        # WK
        if (rights & 1) != 0 and (occ_all & ((ONE << np.uint64(5)) | (ONE << np.uint64(6)))) == ZERO:
            if not (is_square_attacked_k(pieces, occ_all, 4, enemy_white)
                    or is_square_attacked_k(pieces, occ_all, 5, enemy_white)
                    or is_square_attacked_k(pieces, occ_all, 6, enemy_white)):
                out[n] = encode_move(4, 6, 0, F_CASTLE); n += 1
        # WQ
        if (rights & 2) != 0 and (occ_all & ((ONE << np.uint64(1)) | (ONE << np.uint64(2)) | (ONE << np.uint64(3)))) == ZERO:
            if not (is_square_attacked_k(pieces, occ_all, 4, enemy_white)
                    or is_square_attacked_k(pieces, occ_all, 3, enemy_white)
                    or is_square_attacked_k(pieces, occ_all, 2, enemy_white)):
                out[n] = encode_move(4, 2, 0, F_CASTLE); n += 1
    else:
        # BK
        if (rights & 4) != 0 and (occ_all & ((ONE << np.uint64(61)) | (ONE << np.uint64(62)))) == ZERO:
            if not (is_square_attacked_k(pieces, occ_all, 60, enemy_white)
                    or is_square_attacked_k(pieces, occ_all, 61, enemy_white)
                    or is_square_attacked_k(pieces, occ_all, 62, enemy_white)):
                out[n] = encode_move(60, 62, 0, F_CASTLE); n += 1
        # BQ
        if (rights & 8) != 0 and (occ_all & ((ONE << np.uint64(57)) | (ONE << np.uint64(58)) | (ONE << np.uint64(59)))) == ZERO:
            if not (is_square_attacked_k(pieces, occ_all, 60, enemy_white)
                    or is_square_attacked_k(pieces, occ_all, 59, enemy_white)
                    or is_square_attacked_k(pieces, occ_all, 58, enemy_white)):
                out[n] = encode_move(60, 58, 0, F_CASTLE); n += 1
    return n


@njit
def generate_pseudo_legal_into(pieces, occ, state, out):
    side = state[S_SIDE]
    n = gen_pawn_moves(pieces, occ[0], occ[1], occ[2], side, state[S_EP], out, 0)
    n = gen_piece_moves(pieces, occ[0], occ[1], occ[2], side, out, n)
    n = gen_castling(pieces, occ[2], state, out, n)
    return n


@njit
def make_move_k(pieces, occ, p_o_s, state, zobrist_in, mv):
    """In-place make-move with incremental zobrist update.

    Returns ``(captured_pi, captured_sq, new_zobrist)``. The caller saves
    ``zobrist_in`` for undo.
    """
    f = int(mv & 63)
    t = int((mv >> 6) & 63)
    fl = int((mv >> 14) & 3)
    side = state[S_SIDE]
    them = 1 - side

    mover = int(p_o_s[f]) - 1
    bit_f = ONE << np.uint64(f)
    bit_t = ONE << np.uint64(t)

    captured_pi = -1
    captured_sq = -1
    h = zobrist_in
    prev_ep = state[S_EP]
    prev_castling = state[S_CASTLING]

    # 1) Capture.
    if fl == F_EP:
        captured_sq = t - 8 if side == 0 else t + 8
        captured_pi = int(p_o_s[captured_sq]) - 1
        pieces[captured_pi] &= ~(ONE << np.uint64(captured_sq)) & MASK64
        p_o_s[captured_sq] = 0
        h ^= Z_PIECES[captured_pi, captured_sq]
    else:
        tgt = int(p_o_s[t]) - 1
        if tgt >= 0:
            captured_pi = tgt
            captured_sq = t
            pieces[tgt] &= ~bit_t & MASK64
            p_o_s[t] = 0
            h ^= Z_PIECES[tgt, t]

    # 2) Move / promote.
    pieces[mover] &= ~bit_f & MASK64
    p_o_s[f] = 0
    h ^= Z_PIECES[mover, f]
    if fl == F_PROMOTION:
        promo = int((mv >> 12) & 3)
        new_pi = (0 if side == 0 else 6) + (1 + promo)
        pieces[new_pi] |= bit_t
        p_o_s[t] = new_pi + 1
        h ^= Z_PIECES[new_pi, t]
    else:
        pieces[mover] |= bit_t
        p_o_s[t] = mover + 1
        h ^= Z_PIECES[mover, t]

    # 3) Castle: also move rook.
    if fl == F_CASTLE:
        if t == 6:    # G1
            rf, rt, rook_pi = 7, 5, 3
        elif t == 2:  # C1
            rf, rt, rook_pi = 0, 3, 3
        elif t == 62: # G8
            rf, rt, rook_pi = 63, 61, 9
        else:         # C8
            rf, rt, rook_pi = 56, 59, 9
        pieces[rook_pi] &= ~(ONE << np.uint64(rf)) & MASK64
        pieces[rook_pi] |= (ONE << np.uint64(rt))
        p_o_s[rf] = 0
        p_o_s[rt] = rook_pi + 1
        h ^= Z_PIECES[rook_pi, rf]
        h ^= Z_PIECES[rook_pi, rt]

    # 4) Castling rights.
    new_castling = state[S_CASTLING] & CASTLING_CLEAR_TABLE[f] & CASTLING_CLEAR_TABLE[t]
    if new_castling != prev_castling:
        h ^= Z_CASTLING[prev_castling & 0xF]
        h ^= Z_CASTLING[new_castling & 0xF]
    state[S_CASTLING] = new_castling

    # 5) En-passant.
    mover_pt = mover % 6
    new_ep = -1
    if mover_pt == 0:  # PAWN
        if (t - f) == 16 or (t - f) == -16:
            new_ep = (f + t) >> 1
    if prev_ep >= 0:
        h ^= Z_EP[prev_ep & 7]
    if new_ep >= 0:
        h ^= Z_EP[new_ep & 7]
    state[S_EP] = new_ep

    # 6) Clocks.
    if mover_pt == 0 or captured_pi >= 0:
        state[S_HALFMOVE] = 0
    else:
        state[S_HALFMOVE] = state[S_HALFMOVE] + 1
    if side == 1:
        state[S_FULLMOVE] = state[S_FULLMOVE] + 1

    # 7) Side flip.
    state[S_SIDE] = them
    h ^= Z_SIDE

    # 8) Refresh occupancy.
    w = ZERO
    for pi in range(0, 6):
        w |= pieces[pi]
    b = ZERO
    for pi in range(6, 12):
        b |= pieces[pi]
    occ[0] = w
    occ[1] = b
    occ[2] = w | b

    return captured_pi, captured_sq, h


@njit
def undo_move_k(pieces, occ, p_o_s, state, mv,
                captured_pi, captured_sq,
                prev_ep, prev_castling, prev_halfmove, prev_fullmove):
    f = int(mv & 63)
    t = int((mv >> 6) & 63)
    fl = int((mv >> 14) & 3)
    them = state[S_SIDE]
    side = 1 - them

    bit_f = ONE << np.uint64(f)
    bit_t = ONE << np.uint64(t)

    dest_pi = int(p_o_s[t]) - 1
    if fl == F_PROMOTION:
        pieces[dest_pi] &= ~bit_t & MASK64
        pawn_pi = 0 if side == 0 else 6
        pieces[pawn_pi] |= bit_f
        p_o_s[t] = 0
        p_o_s[f] = pawn_pi + 1
    else:
        pieces[dest_pi] &= ~bit_t & MASK64
        pieces[dest_pi] |= bit_f
        p_o_s[t] = 0
        p_o_s[f] = dest_pi + 1

    if captured_pi >= 0:
        pieces[captured_pi] |= (ONE << np.uint64(captured_sq))
        p_o_s[captured_sq] = captured_pi + 1

    if fl == F_CASTLE:
        if t == 6:
            rf, rt, rook_pi = 7, 5, 3
        elif t == 2:
            rf, rt, rook_pi = 0, 3, 3
        elif t == 62:
            rf, rt, rook_pi = 63, 61, 9
        else:
            rf, rt, rook_pi = 56, 59, 9
        pieces[rook_pi] &= ~(ONE << np.uint64(rt)) & MASK64
        pieces[rook_pi] |= (ONE << np.uint64(rf))
        p_o_s[rt] = 0
        p_o_s[rf] = rook_pi + 1

    state[S_SIDE] = side
    state[S_EP] = prev_ep
    state[S_CASTLING] = prev_castling
    state[S_HALFMOVE] = prev_halfmove
    state[S_FULLMOVE] = prev_fullmove

    w = ZERO
    for pi in range(0, 6):
        w |= pieces[pi]
    b = ZERO
    for pi in range(6, 12):
        b |= pieces[pi]
    occ[0] = w
    occ[1] = b
    occ[2] = w | b


# ---- Evaluation kernel ----
# Material values (mg, eg).
MG_VAL = np.array([82, 337, 365, 477, 1025, 0], dtype=np.int32)
EG_VAL = np.array([94, 281, 297, 512, 936, 0], dtype=np.int32)
PHASE_VAL = np.array([0, 1, 1, 2, 4, 0], dtype=np.int32)
PHASE_MAX = 24
TEMPO = 10
BISHOP_PAIR_MG = 30
BISHOP_PAIR_EG = 50

# PSTs in [piece, square] form (white-POV). Will be filled at import.
_PST_MG_W = np.zeros((6, 64), dtype=np.int32)
_PST_EG_W = np.zeros((6, 64), dtype=np.int32)
_PST_MG_B = np.zeros((6, 64), dtype=np.int32)
_PST_EG_B = np.zeros((6, 64), dtype=np.int32)


def _fill_psts():
    from evaluate import PST_MG_W, PST_EG_W, PST_MG_B, PST_EG_B
    _PST_MG_W[:] = PST_MG_W
    _PST_EG_W[:] = PST_EG_W
    _PST_MG_B[:] = PST_MG_B
    _PST_EG_B[:] = PST_EG_B


@njit
def evaluate_k(pieces, occ_w, occ_b, occ_all, side):
    """Return centipawn evaluation from the side-to-move's perspective."""
    mg = 0
    eg = 0
    phase = 0

    for pt in range(6):
        # White
        bb = pieces[pt]
        while bb != ZERO:
            sq = lsb_index(bb)
            bb &= bb - ONE
            mg += MG_VAL[pt] + _PST_MG_W[pt, sq]
            eg += EG_VAL[pt] + _PST_EG_W[pt, sq]
            phase += PHASE_VAL[pt]
        # Black
        bb = pieces[pt + 6]
        while bb != ZERO:
            sq = lsb_index(bb)
            bb &= bb - ONE
            mg -= MG_VAL[pt] + _PST_MG_B[pt, sq]
            eg -= EG_VAL[pt] + _PST_EG_B[pt, sq]
            phase += PHASE_VAL[pt]

    # Bishop pair
    if popcount64(pieces[2]) >= 2:
        mg += BISHOP_PAIR_MG
        eg += BISHOP_PAIR_EG
    if popcount64(pieces[8]) >= 2:
        mg -= BISHOP_PAIR_MG
        eg -= BISHOP_PAIR_EG

    # Mobility (popcount of pseudo-attacks for sliders + knight, minus own).
    mob = 0
    # White knights
    bb = pieces[1]
    while bb != ZERO:
        sq = lsb_index(bb); bb &= bb - ONE
        mob += popcount64(KNIGHT_ATTACKS[sq] & ~occ_w & MASK64)
    bb = pieces[7]
    while bb != ZERO:
        sq = lsb_index(bb); bb &= bb - ONE
        mob -= popcount64(KNIGHT_ATTACKS[sq] & ~occ_b & MASK64)
    # Bishops
    bb = pieces[2]
    while bb != ZERO:
        sq = lsb_index(bb); bb &= bb - ONE
        mob += popcount64(bishop_attacks_n(sq, occ_all) & ~occ_w & MASK64)
    bb = pieces[8]
    while bb != ZERO:
        sq = lsb_index(bb); bb &= bb - ONE
        mob -= popcount64(bishop_attacks_n(sq, occ_all) & ~occ_b & MASK64)
    # Rooks
    bb = pieces[3]
    while bb != ZERO:
        sq = lsb_index(bb); bb &= bb - ONE
        mob += popcount64(rook_attacks_n(sq, occ_all) & ~occ_w & MASK64)
    bb = pieces[9]
    while bb != ZERO:
        sq = lsb_index(bb); bb &= bb - ONE
        mob -= popcount64(rook_attacks_n(sq, occ_all) & ~occ_b & MASK64)
    # Queens
    bb = pieces[4]
    while bb != ZERO:
        sq = lsb_index(bb); bb &= bb - ONE
        mob += popcount64(queen_attacks_n(sq, occ_all) & ~occ_w & MASK64)
    bb = pieces[10]
    while bb != ZERO:
        sq = lsb_index(bb); bb &= bb - ONE
        mob -= popcount64(queen_attacks_n(sq, occ_all) & ~occ_b & MASK64)

    mg += mob * 2
    eg += mob * 2

    if phase > PHASE_MAX:
        phase = PHASE_MAX
    score = (mg * phase + eg * (PHASE_MAX - phase)) // PHASE_MAX

    if side == 0:
        return score + TEMPO
    return -score + TEMPO


@njit
def perft_k(pieces, occ, p_o_s, state, depth, scratch):
    """In-place perft using preallocated ``scratch`` buffer.

    ``scratch`` is ``np.uint16[MAX_PLY, 256]`` to hold per-ply move lists.
    """
    if depth == 0:
        return 1
    n = generate_pseudo_legal_into(pieces, occ, state, scratch[depth - 1])
    side = state[S_SIDE]
    nodes = 0
    for i in range(n):
        mv = scratch[depth - 1, i]
        prev_ep = state[S_EP]
        prev_castling = state[S_CASTLING]
        prev_halfmove = state[S_HALFMOVE]
        prev_fullmove = state[S_FULLMOVE]
        # Zobrist isn't needed for perft, pass through with 0.
        captured_pi, captured_sq, _ = make_move_k(pieces, occ, p_o_s, state, ZERO, mv)
        if not is_check_k(pieces, occ[2], side):
            nodes += perft_k(pieces, occ, p_o_s, state, depth - 1, scratch)
        undo_move_k(pieces, occ, p_o_s, state, mv,
                    captured_pi, captured_sq,
                    prev_ep, prev_castling, prev_halfmove, prev_fullmove)
    return nodes
