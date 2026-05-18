"""Alpha-beta search with iterative deepening, transposition table,
killer + history move ordering, MVV-LVA captures, null-move pruning,
late-move reductions, and quiescence search.

The search operates on a ``Position`` and works in centipawns. All scores are
from the side-to-move's perspective (negamax).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from constants import (
    WHITE, BLACK,
    NUM_PIECES,
    WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK,
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
    FLAG_NORMAL, FLAG_CASTLE, FLAG_EP, FLAG_PROMOTION,
    PROMO_QUEEN,
    MATE, MATE_IN_MAX, INF, MAX_PLY, MAX_MOVES,
)
from move import encode, from_sq, to_sq, promo, flag, to_uci
from position import (
    Position,
    STATE_SIDE, STATE_EP, STATE_CASTLING, STATE_HALFMOVE,
)
from movegen import generate_legal, generate_pseudo_legal, is_check
from evaluate import evaluate


# ---- TT ----
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

_TT_DTYPE = np.dtype([
    ("key", "<u8"),
    ("move", "<u2"),
    ("score", "<i4"),
    ("depth", "i2"),
    ("flag", "u1"),
    ("age", "u1"),
])


@dataclass
class TT:
    size_log2: int = 20  # 2^20 entries × 16 bytes ≈ 16 MB
    table: np.ndarray = field(init=False)
    age: int = 0

    def __post_init__(self) -> None:
        self.table = np.zeros(1 << self.size_log2, dtype=_TT_DTYPE)

    def clear(self) -> None:
        self.table.fill(0)
        self.age = 0

    def probe(self, key: int) -> Optional[np.void]:
        idx = key & ((1 << self.size_log2) - 1)
        e = self.table[idx]
        if int(e["key"]) == key:
            return e
        return None

    def store(self, key: int, move: int, score: int, depth: int, flag_: int) -> None:
        idx = key & ((1 << self.size_log2) - 1)
        e = self.table[idx]
        # Always replace; depth-preferred ordering helps slightly but keep it simple.
        if int(e["key"]) != key or int(e["depth"]) <= depth or int(e["age"]) != self.age:
            e["key"] = np.uint64(key & 0xFFFFFFFFFFFFFFFF)
            e["move"] = np.uint16(move)
            e["score"] = np.int32(score)
            e["depth"] = np.int16(depth)
            e["flag"] = np.uint8(flag_)
            e["age"] = np.uint8(self.age)


# ---- Move ordering helpers ----
# MVV-LVA score: 6*victim + (6 - attacker)
_PIECE_VAL = (100, 320, 330, 500, 900, 20000, 100, 320, 330, 500, 900, 20000)


def _mvvlva(pos: Position, mv: int) -> int:
    t = to_sq(mv)
    fl = flag(mv)
    if fl == FLAG_EP:
        return 105  # pawn captures pawn
    victim_pi = int(pos.piece_on_square[t]) - 1
    if victim_pi < 0:
        return 0
    attacker_pi = int(pos.piece_on_square[from_sq(mv)]) - 1
    victim_pt = victim_pi % 6
    attacker_pt = attacker_pi % 6
    return 1000 + _PIECE_VAL[victim_pi] - attacker_pt


def _move_score(pos: Position, mv: int, ply: int, tt_move: int, killers: List[List[int]], history: np.ndarray) -> int:
    fl = flag(mv)
    if mv == tt_move:
        return 1_000_000
    if fl == FLAG_PROMOTION and promo(mv) == PROMO_QUEEN:
        return 900_000
    t = to_sq(mv)
    # Capture detection
    is_capture = fl == FLAG_EP or pos.piece_on_square[t] != 0
    if is_capture:
        return 800_000 + _mvvlva(pos, mv)
    if mv == killers[ply][0]:
        return 700_000
    if mv == killers[ply][1]:
        return 600_000
    pi = int(pos.piece_on_square[from_sq(mv)]) - 1
    return int(history[pi, t]) if pi >= 0 else 0


@dataclass
class SearchLimits:
    max_depth: int = 64
    movetime_ms: int = -1  # -1 = unlimited
    nodes_limit: int = -1
    stop_flag: Callable[[], bool] = lambda: False


@dataclass
class SearchInfo:
    depth: int = 0
    score: int = 0
    nodes: int = 0
    time_ms: int = 0
    pv: List[int] = field(default_factory=list)


class Searcher:
    def __init__(self, tt: Optional[TT] = None) -> None:
        self.tt = tt or TT()
        self.killers = [[0, 0] for _ in range(MAX_PLY)]
        self.history = np.zeros((NUM_PIECES, 64), dtype=np.int32)
        self.nodes = 0
        self.stop = False
        self.t0 = 0.0
        self.limits = SearchLimits()
        self.best_move = 0
        self.root_score = 0
        # Pre-search game history for repetition detection (positions reached so far).
        self.game_history: List[int] = []
        # During search, search_path holds the positions visited in the current line.
        self.search_path: List[int] = []

    # ---- Public API ----

    def search(self, pos: Position, limits: SearchLimits,
               on_info: Optional[Callable[[SearchInfo], None]] = None) -> int:
        self.tt.age = (self.tt.age + 1) & 0xFF
        self.killers = [[0, 0] for _ in range(MAX_PLY)]
        # Decay history (don't reset entirely between moves).
        self.history //= 2
        self.nodes = 0
        self.stop = False
        self.t0 = time.perf_counter()
        self.limits = limits
        self.best_move = 0
        self.root_score = 0

        info = SearchInfo()
        best_move = 0
        prev_score = 0

        for depth in range(1, limits.max_depth + 1):
            score = self._negamax(pos, depth, -INF, INF, 0, do_null=True)
            if self.stop:
                break
            self.root_score = score

            # Extract PV from TT (best-effort).
            pv = self._extract_pv(pos, depth)
            info.depth = depth
            info.score = score
            info.nodes = self.nodes
            info.time_ms = int((time.perf_counter() - self.t0) * 1000)
            info.pv = pv
            if pv:
                best_move = pv[0]
            if on_info is not None:
                on_info(info)
            prev_score = score
            # Mate found?
            if score >= MATE_IN_MAX or score <= -MATE_IN_MAX:
                break
        self.best_move = best_move
        return best_move

    # ---- Internal ----

    def _should_stop(self) -> bool:
        if self.stop:
            return True
        if self.limits.stop_flag():
            self.stop = True
            return True
        if self.limits.nodes_limit >= 0 and self.nodes >= self.limits.nodes_limit:
            self.stop = True
            return True
        if self.limits.movetime_ms > 0:
            elapsed = (time.perf_counter() - self.t0) * 1000
            if elapsed >= self.limits.movetime_ms:
                self.stop = True
                return True
        return False

    def _is_draw(self, pos: Position, ply: int) -> bool:
        # 50-move rule.
        if int(pos.state[STATE_HALFMOVE]) >= 100:
            # Mate-or-stalemate could still happen; engines usually check first.
            # We treat halfmove>=100 as a draw if not in check + we're not at root.
            if not is_check(pos):
                return True
        # Repetition: a repeat in the search path or any earlier-game repeat == draw.
        key = int(pos.zobrist)
        # Check search path (current line).
        count = 0
        for k in self.search_path:
            if k == key:
                count += 1
        for k in self.game_history:
            if k == key:
                count += 1
        if count >= 1:  # one prior occurrence + current = twofold; counted as draw inside search.
            return True
        return False

    def _negamax(self, pos: Position, depth: int, alpha: int, beta: int, ply: int, do_null: bool) -> int:
        self.nodes += 1
        if (self.nodes & 4095) == 0 and self._should_stop():
            return 0

        if ply > 0 and self._is_draw(pos, ply):
            return 0

        if depth <= 0:
            return self._quiescence(pos, alpha, beta, ply)

        in_check = is_check(pos)
        # Search extension: in check -> +1 ply.
        if in_check:
            depth += 1

        # TT probe.
        key = int(pos.zobrist)
        tt_move = 0
        tte = self.tt.probe(key)
        if tte is not None:
            tt_depth = int(tte["depth"])
            tt_move = int(tte["move"])
            if tt_depth >= depth and ply > 0:
                tt_score = int(tte["score"])
                tt_flag = int(tte["flag"])
                if tt_flag == TT_EXACT:
                    return tt_score
                if tt_flag == TT_LOWER and tt_score >= beta:
                    return tt_score
                if tt_flag == TT_UPPER and tt_score <= alpha:
                    return tt_score

        # Null-move pruning.
        if (do_null and depth >= 3 and not in_check
                and self._has_non_pawn_material(pos)):
            self._do_null(pos)
            self.search_path.append(key)
            r = 3 if depth > 6 else 2
            score = -self._negamax(pos, depth - 1 - r, -beta, -beta + 1, ply + 1, do_null=False)
            self.search_path.pop()
            self._undo_null(pos)
            if self.stop:
                return 0
            if score >= beta:
                return beta

        moves = generate_legal(pos)
        if not moves:
            return -MATE + ply if in_check else 0

        # Order moves.
        scored = [(_move_score(pos, m, ply, tt_move, self.killers, self.history), m) for m in moves]
        scored.sort(key=lambda x: x[0], reverse=True)

        best_score = -INF
        best_move = 0
        alpha_orig = alpha

        self.search_path.append(key)
        try:
            for i, (_s, mv) in enumerate(scored):
                pos.make_move(mv)
                # LMR.
                reduction = 0
                fl = flag(mv)
                is_capture = fl == FLAG_EP or int(pos.piece_on_square[to_sq(mv)]) > 0 and False  # we already moved; use heuristic instead
                # Reduction conditions: depth>=3, move index>=3, non-capture, non-promotion, not in-check at root
                if (depth >= 3 and i >= 3 and fl == FLAG_NORMAL and not in_check
                        and not is_check(pos)):
                    reduction = 1 if i < 6 else 2

                if i == 0:
                    score = -self._negamax(pos, depth - 1, -beta, -alpha, ply + 1, do_null=True)
                else:
                    score = -self._negamax(pos, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, do_null=True)
                    if not self.stop and score > alpha and reduction > 0:
                        score = -self._negamax(pos, depth - 1, -alpha - 1, -alpha, ply + 1, do_null=True)
                    if not self.stop and alpha < score < beta:
                        score = -self._negamax(pos, depth - 1, -beta, -alpha, ply + 1, do_null=True)

                pos.undo_move()
                if self.stop:
                    return 0

                if score > best_score:
                    best_score = score
                    best_move = mv
                    if score > alpha:
                        alpha = score
                        if alpha >= beta:
                            # Beta cutoff.
                            if fl == FLAG_NORMAL and int(self.killers[ply][0]) != mv:
                                self.killers[ply][1] = self.killers[ply][0]
                                self.killers[ply][0] = mv
                            pi = int(pos.piece_on_square[from_sq(mv)]) - 1
                            if pi >= 0 and fl == FLAG_NORMAL:
                                self.history[pi, to_sq(mv)] += depth * depth
                            break
        finally:
            self.search_path.pop()

        # Store TT.
        if best_score <= alpha_orig:
            tt_flag = TT_UPPER
        elif best_score >= beta:
            tt_flag = TT_LOWER
        else:
            tt_flag = TT_EXACT
        self.tt.store(key, best_move, best_score, depth, tt_flag)
        return best_score

    def _quiescence(self, pos: Position, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        if (self.nodes & 4095) == 0 and self._should_stop():
            return 0

        stand_pat = evaluate(pos)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        # Generate captures only (and promotions).
        in_check = is_check(pos)
        if in_check:
            moves = generate_legal(pos)
        else:
            moves = [m for m in generate_legal(pos)
                     if flag(m) == FLAG_EP
                     or flag(m) == FLAG_PROMOTION
                     or int(pos.piece_on_square[to_sq(m)]) > 0]

        scored = [(_mvvlva(pos, m), m) for m in moves]
        scored.sort(key=lambda x: x[0], reverse=True)

        for _s, mv in scored:
            pos.make_move(mv)
            score = -self._quiescence(pos, -beta, -alpha, ply + 1)
            pos.undo_move()
            if self.stop:
                return 0
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    # ---- Null move ----

    def _has_non_pawn_material(self, pos: Position) -> bool:
        side = int(pos.state[STATE_SIDE])
        if side == WHITE:
            return bool(int(pos.pieces[WN]) | int(pos.pieces[WB]) | int(pos.pieces[WR]) | int(pos.pieces[WQ]))
        else:
            return bool(int(pos.pieces[BN]) | int(pos.pieces[BB]) | int(pos.pieces[BR]) | int(pos.pieces[BQ]))

    def _do_null(self, pos: Position) -> None:
        # Save just enough state to roll back. ep_square also matters.
        prev_side = int(pos.state[STATE_SIDE])
        prev_ep = int(pos.state[STATE_EP])
        prev_zobrist = pos.zobrist
        prev_halfmove = int(pos.state[STATE_HALFMOVE])
        from zobrist import Z_SIDE, Z_EP
        h = pos.zobrist
        if prev_ep >= 0:
            h ^= Z_EP[prev_ep & 7]
        h ^= Z_SIDE
        pos.zobrist = np.uint64(int(h) & 0xFFFFFFFFFFFFFFFF)
        pos.state[STATE_SIDE] = 1 - prev_side
        pos.state[STATE_EP] = -1
        pos.state[STATE_HALFMOVE] = prev_halfmove + 1
        # Push tiny undo marker.
        pos._undo.append(("NULL", prev_side, prev_ep, prev_zobrist, prev_halfmove))

    def _undo_null(self, pos: Position) -> None:
        rec = pos._undo.pop()
        assert rec[0] == "NULL"
        _, prev_side, prev_ep, prev_zobrist, prev_halfmove = rec
        pos.state[STATE_SIDE] = prev_side
        pos.state[STATE_EP] = prev_ep
        pos.state[STATE_HALFMOVE] = prev_halfmove
        pos.zobrist = prev_zobrist

    # ---- PV extraction ----

    def _extract_pv(self, pos: Position, depth: int) -> List[int]:
        pv: List[int] = []
        made = 0
        for _ in range(depth):
            tte = self.tt.probe(int(pos.zobrist))
            if tte is None:
                break
            mv = int(tte["move"])
            if mv == 0:
                break
            # Verify legality (TT might hold stale move from a different position w/ same hash).
            if mv not in generate_legal(pos):
                break
            pv.append(mv)
            pos.make_move(mv)
            made += 1
        for _ in range(made):
            pos.undo_move()
        return pv
