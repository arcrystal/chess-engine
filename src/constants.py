"""Core piece and side constants.

Square convention: bit 0 = a1, bit 7 = h1, bit 56 = a8, bit 63 = h8.
"""

# Piece types (1..6 reserved for the array-path piece values; 0 == empty).
EMPTY = 0
PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

# Sides.
WHITE = 0
BLACK = 1

# Piece-color index (used for pieces[12], piece_on_square[64]).
# 0..5 = white P,N,B,R,Q,K ; 6..11 = black P,N,B,R,Q,K ; -1 = empty (we encode 0 in piece_on_square as empty and shift by 1).
WP, WN, WB, WR, WQ, WK = 0, 1, 2, 3, 4, 5
BP, BN, BB, BR, BQ, BK = 6, 7, 8, 9, 10, 11
NUM_PIECES = 12

# piece_on_square uses 0 = empty, 1..12 = piece+1 (so int8 is enough).
EMPTY_SQUARE = 0

# Castling-right bitmask bits.
CR_WK = 1
CR_WQ = 2
CR_BK = 4
CR_BQ = 8
CR_ALL = CR_WK | CR_WQ | CR_BK | CR_BQ

# Move flags (bits 14..15 of the packed 16-bit move).
FLAG_NORMAL = 0
FLAG_CASTLE = 1
FLAG_EP = 2
FLAG_PROMOTION = 3

# Promotion target encoding inside the 16-bit move (bits 12..13).
# 0=knight, 1=bishop, 2=rook, 3=queen — corresponds to KNIGHT-2 etc.
PROMO_KNIGHT = 0
PROMO_BISHOP = 1
PROMO_ROOK = 2
PROMO_QUEEN = 3

# Useful square constants.
A1, B1, C1, D1, E1, F1, G1, H1 = 0, 1, 2, 3, 4, 5, 6, 7
A8, B8, C8, D8, E8, F8, G8, H8 = 56, 57, 58, 59, 60, 61, 62, 63

# Files.
FILE_A = 0x0101010101010101
FILE_B = FILE_A << 1
FILE_C = FILE_A << 2
FILE_D = FILE_A << 3
FILE_E = FILE_A << 4
FILE_F = FILE_A << 5
FILE_G = FILE_A << 6
FILE_H = FILE_A << 7
NOT_FILE_A = 0xFFFFFFFFFFFFFFFF ^ FILE_A
NOT_FILE_H = 0xFFFFFFFFFFFFFFFF ^ FILE_H

# Ranks.
RANK_1 = 0x00000000000000FF
RANK_2 = RANK_1 << 8
RANK_3 = RANK_1 << 16
RANK_4 = RANK_1 << 24
RANK_5 = RANK_1 << 32
RANK_6 = RANK_1 << 40
RANK_7 = RANK_1 << 48
RANK_8 = RANK_1 << 56

MASK64 = 0xFFFFFFFFFFFFFFFF

# Search constants.
MATE = 30000
MATE_IN_MAX = MATE - 1024
INF = 32000
MAX_PLY = 128
MAX_MOVES = 256
