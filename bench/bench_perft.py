"""Perft + eval + search benchmark sweep.

Run as ``python -m bench.bench_perft``; writes results to
``bench/baseline.json`` if missing, otherwise ``bench/post.json`` plus a
delta report on stdout.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import sys

# Allow ``import position`` etc.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fen import parse_fen  # noqa: E402
from perft import perft  # noqa: E402
from position import Position  # noqa: E402
from evaluate import evaluate  # noqa: E402
from search import Searcher, SearchLimits  # noqa: E402


BENCH = Path(__file__).resolve().parent
BASELINE = BENCH / "baseline.json"
POST = BENCH / "post.json"


CASES = [
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4),
    ("Kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3),
    ("Pos3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4),
    ("Pos4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2pP/R2Q1RK1 w kq - 0 1", 3),
    ("Pos5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3),
    ("Pos6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 3),
]


def bench_perft() -> dict:
    out = {}
    for name, fen, depth in CASES:
        pos = parse_fen(fen)
        t0 = time.perf_counter()
        n = perft(pos, depth)
        t = time.perf_counter() - t0
        out[name] = {"depth": depth, "nodes": n, "seconds": t, "nps": int(n / t) if t > 0 else 0}
    return out


def bench_eval(iters: int = 1000) -> dict:
    pos = parse_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    # warm.
    for _ in range(50):
        evaluate(pos)
    t0 = time.perf_counter()
    for _ in range(iters):
        evaluate(pos)
    t = time.perf_counter() - t0
    return {"ns_per_call": int(1e9 * t / iters), "iters": iters}


def bench_search(seconds: float = 1.0) -> dict:
    pos = Position()
    s = Searcher()
    limits = SearchLimits(max_depth=64, movetime_ms=int(seconds * 1000))
    last = {"depth": 0, "nodes": 0, "time_ms": 0, "score": 0}
    def on_info(info):
        last.update({"depth": info.depth, "nodes": info.nodes, "time_ms": info.time_ms, "score": info.score})
    s.search(pos, limits, on_info=on_info)
    nps = int(last["nodes"] / max(1, last["time_ms"]) * 1000)
    return {**last, "nps": nps}


def warm_numba() -> None:
    """Trigger numba JIT compilation up front so the benchmarks measure steady state."""
    pos = parse_fen(CASES[0][1])
    perft(pos, 2)  # warms generate_pseudo_legal_into, make/unmake, is_check, lsb, popcount
    evaluate(pos)  # warms evaluate_k


def main() -> None:
    print("warming numba kernels...")
    warm_numba()
    print("running perft sweep...")
    perft_res = bench_perft()
    for name, r in perft_res.items():
        print(f"  {name:>8} d{r['depth']}: {r['nodes']:>9} nodes in {r['seconds']:.2f}s ({r['nps']} nps)")

    print("running eval microbench...")
    eval_res = bench_eval()
    print(f"  evaluate: {eval_res['ns_per_call']} ns/call")

    print("running 1-second search from startpos...")
    search_res = bench_search()
    print(f"  search: depth {search_res['depth']}, nodes {search_res['nodes']}, "
          f"nps {search_res['nps']}, score {search_res['score']}")

    out = {"perft": perft_res, "eval": eval_res, "search": search_res}
    if not BASELINE.exists():
        BASELINE.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {BASELINE} (baseline)")
    else:
        POST.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {POST}")
        baseline = json.loads(BASELINE.read_text())
        print("delta vs baseline:")
        for name in perft_res:
            b = baseline["perft"][name]["seconds"]
            n = perft_res[name]["seconds"]
            ratio = b / n if n > 0 else float("inf")
            print(f"  {name:>8}: {b:.2f}s -> {n:.2f}s  ({ratio:.2f}x)")
        b_e = baseline["eval"]["ns_per_call"]
        n_e = eval_res["ns_per_call"]
        print(f"  eval ns/call: {b_e} -> {n_e}  ({b_e/n_e:.2f}x)")
        b_s = baseline["search"]["nps"]
        n_s = search_res["nps"]
        print(f"  search nps:   {b_s} -> {n_s}  ({n_s/b_s:.2f}x)")


if __name__ == "__main__":
    main()
