PY := .venv/bin/python
PYTEST := .venv/bin/pytest

.PHONY: test perft bench uci clean

test:
	$(PYTEST) -q

perft:
	$(PY) -m bench.bench_perft

bench:
	$(PY) -m bench.bench_perft

uci:
	$(PY) -m src.uci

clean:
	rm -rf .numba_cache src/__pycache__ tests/__pycache__ bench/__pycache__
