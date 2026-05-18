"""Minimal UCI shim.

Supports:
    uci, isready, ucinewgame, quit
    position startpos [moves ...]
    position fen <FEN> [moves ...]
    go [movetime N] [depth N] [wtime ... btime ... [winc ... binc ...]]
    stop

Time management: simple fixed-fraction. Refinements (ponder, MultiPV, etc.)
are out of scope for v1.
"""

from __future__ import annotations

import sys
import threading
from typing import Iterable, Optional

from constants import MATE, MATE_IN_MAX
from position import Position
from fen import parse_fen, to_fen
from move import to_uci, from_uci
from movegen import generate_legal
from search import Searcher, SearchLimits, SearchInfo


ENGINE_NAME = "Cheshire"
ENGINE_AUTHOR = "ChessEngine (Anthropic worktree)"


class UCI:
    def __init__(self) -> None:
        self.pos = Position()
        self.searcher = Searcher()
        self.stop_flag = False
        self._search_thread: Optional[threading.Thread] = None

    def _emit(self, line: str) -> None:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def _stop_cb(self) -> bool:
        return self.stop_flag

    def cmd_uci(self) -> None:
        self._emit(f"id name {ENGINE_NAME}")
        self._emit(f"id author {ENGINE_AUTHOR}")
        self._emit("uciok")

    def cmd_isready(self) -> None:
        self._emit("readyok")

    def cmd_ucinewgame(self) -> None:
        self.pos = Position()
        self.searcher.tt.clear()
        self.searcher.game_history.clear()

    def cmd_position(self, tokens: list[str]) -> None:
        if not tokens:
            return
        i = 0
        if tokens[0] == "startpos":
            self.pos = Position()
            i = 1
        elif tokens[0] == "fen":
            # FEN has 6 fields.
            fen = " ".join(tokens[1:7])
            self.pos = parse_fen(fen)
            i = 7
        else:
            return
        self.searcher.game_history = [int(self.pos.zobrist)]
        if i < len(tokens) and tokens[i] == "moves":
            for u in tokens[i + 1:]:
                legal = generate_legal(self.pos)
                try:
                    mv = from_uci(u, legal)
                except ValueError:
                    self._emit(f"info string illegal move {u}")
                    return
                self.pos.make_move(mv)
                self.searcher.game_history.append(int(self.pos.zobrist))

    def _compute_movetime(self, tokens: list[str]) -> tuple[int, int]:
        """Return (movetime_ms, max_depth) from the go tokens."""
        movetime = -1
        depth = 64
        wtime = btime = winc = binc = -1
        movestogo = -1
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t == "movetime" and i + 1 < len(tokens):
                movetime = int(tokens[i + 1]); i += 2
            elif t == "depth" and i + 1 < len(tokens):
                depth = int(tokens[i + 1]); i += 2
            elif t == "wtime" and i + 1 < len(tokens):
                wtime = int(tokens[i + 1]); i += 2
            elif t == "btime" and i + 1 < len(tokens):
                btime = int(tokens[i + 1]); i += 2
            elif t == "winc" and i + 1 < len(tokens):
                winc = int(tokens[i + 1]); i += 2
            elif t == "binc" and i + 1 < len(tokens):
                binc = int(tokens[i + 1]); i += 2
            elif t == "movestogo" and i + 1 < len(tokens):
                movestogo = int(tokens[i + 1]); i += 2
            elif t == "infinite":
                movetime = -1; depth = 64; i += 1
            else:
                i += 1
        if movetime < 0 and (wtime >= 0 or btime >= 0):
            white = self.pos.white_to_move
            t = wtime if white else btime
            inc = (winc if white else binc) or 0
            if t > 0:
                budget = t // 30 + inc // 2
                # Don't use more than 1/3 of remaining time.
                movetime = max(10, min(budget, t // 3))
        return movetime, depth

    def cmd_go(self, tokens: list[str]) -> None:
        movetime, depth = self._compute_movetime(tokens)
        self.stop_flag = False

        def runner():
            def on_info(info: SearchInfo):
                pv_str = " ".join(to_uci(m) for m in info.pv)
                score_str = self._format_score(info.score)
                nps = int(info.nodes / max(1, info.time_ms) * 1000)
                self._emit(
                    f"info depth {info.depth} score {score_str} "
                    f"nodes {info.nodes} nps {nps} time {info.time_ms} pv {pv_str}"
                )

            limits = SearchLimits(
                max_depth=depth,
                movetime_ms=movetime if movetime >= 0 else -1,
                stop_flag=self._stop_cb,
            )
            best = self.searcher.search(self.pos, limits, on_info=on_info)
            self._emit(f"bestmove {to_uci(best) if best else '0000'}")

        self._search_thread = threading.Thread(target=runner, daemon=True)
        self._search_thread.start()

    def cmd_stop(self) -> None:
        self.stop_flag = True
        if self._search_thread is not None:
            self._search_thread.join(timeout=5.0)
        self._search_thread = None

    @staticmethod
    def _format_score(score: int) -> str:
        if score >= MATE_IN_MAX:
            plies = MATE - score
            return f"mate {(plies + 1) // 2}"
        if score <= -MATE_IN_MAX:
            plies = MATE + score
            return f"mate -{(plies + 1) // 2}"
        return f"cp {score}"

    def loop(self, stream: Iterable[str] = None) -> None:
        if stream is None:
            stream = sys.stdin
        for raw in stream:
            line = raw.strip()
            if not line:
                continue
            tokens = line.split()
            cmd = tokens[0]
            args = tokens[1:]
            if cmd == "uci":
                self.cmd_uci()
            elif cmd == "isready":
                self.cmd_isready()
            elif cmd == "ucinewgame":
                self.cmd_ucinewgame()
            elif cmd == "position":
                self.cmd_position(args)
            elif cmd == "go":
                self.cmd_go(args)
            elif cmd == "stop":
                self.cmd_stop()
            elif cmd == "quit":
                self.stop_flag = True
                break
            elif cmd == "d":
                self._emit("info string\n" + self.pos.board_ascii())
                self._emit(f"info string fen {to_fen(self.pos)}")
            # Unknown commands are silently ignored per UCI.


def main() -> None:
    UCI().loop()


if __name__ == "__main__":
    main()
