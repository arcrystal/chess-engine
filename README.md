# ChessEngine

A personal chess engine in Python + numba, with two surface APIs that share
the same hot-path kernel:

* **`Position`** — bitboard core (12 piece bitboards + occupancy + state).
  All hot operations (movegen, attacks, make/unmake, eval) live in
  `@njit`-compiled kernels.
* **`GameState`** — legacy `int8[8,8]` array surface. A thin wrapper that
  materialises a `Position` on demand. Used by older tests and by anyone
  who wants a row/col board to print or modify.

## Features

* Magic-bitboard slider attacks. Magics are generated deterministically
  from a fixed seed on first run and cached as a `~/.cache/chessengine/magic_v1.npz`
  file (mmap-loaded thereafter). Verified vs `python-chess` for 3 200+
  random (square, occupancy) lookups.
* 16-bit packed move encoding `(from | to | promo | flag)`.
* Move generation: pseudo-legal generator + post-filter `is_check` for legality.
  Perft validated against `python-chess` for the 6 canonical positions to depth 4+.
* Evaluation: PeSTO-style PSTs interpolated between midgame and endgame by
  game phase, plus bishop pair and mobility (popcount-based).
* Search: negamax + αβ, iterative deepening, transposition table, killer
  moves, history heuristic, MVV-LVA capture ordering, null-move pruning,
  late-move reductions, quiescence search.
* Threefold-repetition + 50-move-rule draw detection.
* UCI shim that speaks `position`, `go`, `stop`, `quit` and emits
  `info depth ... score ... nodes ... nps ... pv ...` lines.
* Zobrist hashing maintained incrementally inside the numba kernel.
* Python 3.13 (numba 0.62-compatible). Apache 2.0.

## Install

```bash
python3.13 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Usage

```bash
# run the full test suite
make test
# or .venv/bin/pytest tests/

# perft + eval + search benchmark
make bench
# (writes bench/baseline.json on first run, then bench/post.json + delta)

# launch a UCI engine (drive from cutechess, Arena, lichess-bot, etc.)
make uci
# or .venv/bin/python -m src.uci
```

Drive it programmatically:

```python
import chess, chess.engine
eng = chess.engine.SimpleEngine.popen_uci([".venv/bin/python", "-m", "src.uci"])
board = chess.Board()
res = eng.play(board, chess.engine.Limit(time=0.5))
print(res.move)
eng.quit()
```

## Layout

```
src/
├── constants.py    piece, side, flag, square constants
├── move.py         packed 16-bit move encoding + UCI conversion
├── magic.py        deterministic magic-bitboard generation + .npz cache
├── attacks.py      knight/king/pawn lookup tables; magic accessors
├── kernels.py      @njit hot path (movegen, make/unmake, is_check, eval, perft)
├── position.py     Position class — bitboard core, wrapping kernels
├── movegen.py      Python facade producing list[int] of packed moves
├── perft.py        perft + python-chess lockstep validator
├── zobrist.py      Zobrist tables (Position maintains the hash)
├── evaluate.py     evaluation wrapper (PST tables; kernel does the work)
├── search.py       negamax/αβ search with TT, ID, killers, history, LMR, NMP, QS
├── fen.py          FEN parser + serialiser
├── uci.py          UCI loop
├── game.py         legacy GameState array surface
└── engine_utils.py legacy adapter expected by old tests
tests/              pytest suite (perft, promotion, castling, en-passant, zobrist, magic, fen, eval, search, UCI)
bench/              measurement scripts and JSON results
```

## Test coverage

| Area | Tests |
|---|---|
| Perft (6 canonical FENs) | counts + lockstep vs python-chess |
| Promotion | all 4 targets × straight/capture × both colors |
| Castling | success, blocked, through-check, into-check, after king/rook moves |
| En-passant | success + 5 illegal-attempt cases |
| Zobrist | make/unmake invariance over 5000 random moves + transposition |
| Magic tables | 1000 random (sq, occ) per slider vs python-chess |
| Evaluation | starting position, material swings, king shelter, passed pawn |
| FEN | round-trip on 8 positions |
| Search | mate-in-1 detection, immediate free-piece capture |
| UCI | handshake + 10-ply self-play |

## License

Apache 2.0.
