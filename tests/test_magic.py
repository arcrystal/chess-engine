"""Magic-bitboard correctness vs ``python-chess``."""

from __future__ import annotations

import random

import chess

from attacks import bishop_attacks, rook_attacks


def test_bishop_attacks_match_python_chess():
    rng = random.Random(0xBADC0FFEE)
    for _ in range(1000):
        sq = rng.randrange(64)
        occ = rng.getrandbits(64)
        ours = bishop_attacks(sq, occ)
        ref = int(chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & occ])
        assert ours == ref, f"sq={sq} occ={hex(occ)} ours={hex(ours)} ref={hex(ref)}"


def test_rook_attacks_match_python_chess():
    rng = random.Random(0xCAFE)
    for _ in range(1000):
        sq = rng.randrange(64)
        occ = rng.getrandbits(64)
        ours = rook_attacks(sq, occ)
        ref = int(
            chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & occ]
            | chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & occ]
        )
        assert ours == ref, f"sq={sq} occ={hex(occ)}"
