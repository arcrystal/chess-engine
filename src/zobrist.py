"""Zobrist tables. ``Position`` imports the tables and maintains the hash.

A position hash is the XOR of:
    Z_PIECES[piece_index, square] for every piece on the board
    Z_CASTLING[castling_mask]
    Z_EP[ep_file] if ep_square >= 0
    Z_SIDE if black to move
"""

from __future__ import annotations

import random

import numpy as np

from constants import NUM_PIECES

_rng = random.Random(0x5EED_C0DE)

Z_PIECES = np.zeros((NUM_PIECES, 64), dtype=np.uint64)
for pi in range(NUM_PIECES):
    for sq in range(64):
        Z_PIECES[pi, sq] = np.uint64(_rng.getrandbits(64))

Z_CASTLING = np.zeros(16, dtype=np.uint64)
for c in range(16):
    Z_CASTLING[c] = np.uint64(_rng.getrandbits(64))

Z_EP = np.zeros(8, dtype=np.uint64)
for f in range(8):
    Z_EP[f] = np.uint64(_rng.getrandbits(64))

Z_SIDE = np.uint64(_rng.getrandbits(64))
