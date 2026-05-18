# Overhaul Report

Reference: `/Users/acrystal/.claude/plans/1-not-sure-if-elegant-crystal.md`.

## What changed

The pre-overhaul code was unimportable on both paths (engine_utils missing,
`engine.py` IndentationError, magic data parsed from a 5.6 MB Python source
on every run, promotion mapping swapped pieces, castling never moved the
rook, …). It has been replaced with a unified bitboard core wrapped by two
surface APIs:

* `Position` (bitboard) and `GameState` (legacy 8×8 array) both delegate
  to a single ``@njit``-compiled kernel for the hot path.
* Magic tables regenerated deterministically and verified against
  `python-chess` for 3 200 random (square, occupancy) lookups.
* Perft validated against `python-chess` (lockstep set-equality and node-count)
  for all 6 canonical positions to depth ≥ 3, with deeper depths in the bench:
    * startpos d5 = 4 865 609
    * Kiwipete d4 = 4 085 603
    * Pos3 d5 = 674 624
    * Pos4 d4 = 438 345 (matches python-chess; chess-programming wiki value 422 333
      is for a different FEN)
    * Pos5 d4 = 2 103 487
    * Pos6 d4 = 3 894 594
* UCI shim plays via `python-chess.engine.SimpleEngine.popen_uci`.

## Files

Deleted (replaced):
* `src/engine.py` — replaced by `src/search.py`.
* `src/bitboard_game.py` — replaced by `src/position.py`.
* `src/bitboard_magic.py`, `src/bitboard_nomagic.py`, `src/magic_tables.py`
  — replaced by `src/magic.py` + `src/attacks.py` + `~/.cache/chessengine/magic_v1.npz`.
* `src/generate_moves.py` — replaced by `src/movegen.py` + `src/kernels.py`.
* `src/bitboard_perft.py` — replaced by `src/perft.py`.
* `src/test.py` — ad-hoc scratch, deleted.
* `magic_tables/{bishop,rook}_attack_table.pkl` and the `magic_tables/` directory.

New (canonical):
* `src/constants.py`, `src/move.py`, `src/magic.py`, `src/attacks.py`,
  `src/kernels.py`, `src/position.py`, `src/movegen.py`, `src/perft.py`,
  `src/zobrist.py`, `src/evaluate.py`, `src/search.py`, `src/fen.py`,
  `src/uci.py`, `src/game.py`, `src/engine_utils.py`, `src/evaluate_board.py`.
* `pyproject.toml`, `conftest.py`, `Makefile`.
* `bench/bench_perft.py` and `bench/baseline.json` / `bench/post.json`.
* `tests/test_perft.py`, `tests/test_promotion.py`, `tests/test_zobrist.py`,
  `tests/test_magic.py`, `tests/test_fen.py`, `tests/test_search.py`,
  `tests/test_uci.py`; rewrote `tests/test_castling.py` to use FENs (the
  original had illegal moves like `g8→g6` and `c1→c4` baked into the data).

## Measured deltas (Apple Silicon M-series, Python 3.13 + numba 0.62)

Baseline = the new pure-Python+numpy implementation that hadn't yet had
the hot path moved into ``@njit``. The "old" code in the repo couldn't run.

| Metric | Pre-numba | Post-numba | Speedup |
|---|---|---|---|
| Perft startpos d4 (wall-clock) | 2.34 s | 0.01 s | **215×** |
| Perft Kiwipete d3 | 1.12 s | 0.01 s | 215× |
| Perft Pos3 d4 | 0.60 s | 0.00 s | 208× |
| Perft Pos4 d3 | 0.12 s | 0.00 s | 215× |
| Perft Pos5 d3 | 0.75 s | 0.00 s | 217× |
| Perft Pos6 d3 | 1.02 s | 0.00 s | 220× |
| Eval ns/call (Kiwipete) | 31 723 | 1 869 | 17× |
| Search nps from startpos (1 s) | 8 246 | 14 869 | 1.8× |
| Search depth reached in 1 s | 6 | 6 | (depth dominated by Python search loop) |
| Cold import | ~3 s (parse 5.6 MB literal) + numba cold-compile | ~50 ms (mmap .npz) + ~10 s numba JIT first run | n/a |
| Magic-table cache size | 5.6 MB Python source | 2.3 MB `.npz` (mmap-loadable) | 2.4× smaller |

Steady-state perft nps with the numba kernel: **17–20 M nodes/second**.

## Where the remaining performance is

The search loop is still in Python (TT lookups against a numpy structured
array, move ordering through Python lambdas, recursion overhead, killer/history
table accesses, PV extraction). Moving it under `@njit` is the single
biggest remaining win (estimate 5–10× on search nps). It is left for a
follow-up because:

1. The search has many control-flow paths (TT, null-move, LMR, QS, mate
   scoring, repetition). Numba-ising it cleanly takes time; rushing it
   trades search bugs for nps.
2. The current state is functional and correct: the engine plays full
   games via UCI and finds mates within its depth horizon.

## Definition-of-Done checklist

* [x] `make test` passes 100% (79 tests).
* [x] `make perft` (the test suite + bench) lockstep-matches python-chess
  on all 6 canonical FENs to depth ≥ 3 in CI and deeper in the bench.
* [x] `make bench` produces JSON and a delta report; numbers above are
  rooted in `bench/baseline.json` / `bench/post.json`.
* [x] `python -c "from src import uci"` imports without error.
* [x] Both surface APIs (Position and GameState) work; the legacy tests
  exercise `GameState`, the new tests exercise `Position`.
* [x] Promotion correctness: 16 promotion tests cover {N, B, R, Q} ×
  {straight, capture} × {white, black} and `piece_on_square` post-promotion.
* [x] Castling correctness: success, blocked, through-check, into-check,
  rights-cleared-after-rook-move, mirror, etc.
* [x] Zobrist correctness: make/unmake invariance over 5 000 random moves;
  transposition via knight moves yields equal hashes.
* [x] Magic-table correctness: 2 000 (sq, occ) lookups vs python-chess.
* [x] UCI smoke test: handshake + 10-ply self-play with no crashes,
  malformed output, or hung `bestmove`.
* [x] Threefold-repetition + 50-move-rule + draw detection in `search._is_draw`.
* [x] Dead code removed (8 source files + 2 pickle files + 1 directory).
* [x] No fabricated benchmark numbers — every figure is in `bench/post.json`.
* [x] No new dependencies beyond `numpy`, `numba`, `python-chess`,
  `pytest`, `pytest-benchmark`, `hypothesis`.
* [x] README updated.
* [x] `bench/baseline.json` and `bench/post.json` captured.

## Remaining risks

* **Search rewrite into numba** — currently single-threaded Python; an
  in-numba search would reach 4–5 plies deeper at the same wall-clock.
* **Python 3.14 path** — numba 0.62 only goes up to 3.13. When numba 0.63+
  adds 3.14 support, lift the `requires-python` upper bound in `pyproject.toml`.
* **Tablebase / opening book / NNUE** — out of scope for v1.
* **Lazy SMP / root-split** — Phase-D; needs the native or numba-search
  pieces to actually pay off on Apple Silicon.
