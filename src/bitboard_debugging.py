from src.bitboard_game import BitboardGameState
from src.bitboard_gamestate_utils import attack_map_numba, apply_move_numba, update_occupancies_numba, undo_move_numba
from src.bitboard_nomagic import knight_attacks, king_attacks
from src.bitboard_magic import bishop_attacks, rook_attacks, queen_attacks
from src.constants import board_str
from src.bitboard_utils import rank_mask, file_mask, bb_to_squares
from numba import uint64, boolean
from numba import types
from numba.typed import List
from src.generate_moves import *
import warnings
warnings.simplefilter("ignore")

def print_board(gs, return_str=False):
    """Print the current board with pieces."""
    # Create a list of piece symbols
    piece_symbols = {
        "white_pawns": "P", "black_pawns": "p",
        "white_knights": "N", "black_knights": "n",
        "white_bishops": "B", "black_bishops": "b",
        "white_rooks": "R", "black_rooks": "r",
        "white_queens": "Q", "black_queens": "q",
        "white_king": "K", "black_king": "k"
    }

    # Iterate over the board squares (0 to 63)
    board_str = ""
    for rank in range(7, -1, -1):
        row = ""
        for file in range(8):
            square_index = (rank * 8) + file
            piece_found = False

            # Check if the piece is on this square
            for piece_type in piece_symbols:
                bb = getattr(gs, piece_type)
                # Ensure that square_index is cast to np.uint64
                if bb & (1 << square_index):
                    row += piece_symbols[piece_type] + " "
                    piece_found = True
                    break

            # If no piece found, print a dot for an empty square
            if not piece_found:
                row += "* "
                
        if return_str:
            
            board_str += row + "\n"
        else:
            print(row)
    
    if return_str:
        return board_str
    
def print_bitboard(squares: int, bitboard: int, label: str = ""):
    if label:
        print(f"{label}")
    print("  +-----------------+")
    for rank in range(7, -1, -1):
        row = f"{rank + 1} |"
        for file in range(8):
            square = rank * 8 + file
            row += " X" if (int(bitboard) >> square) & 1 else " O" if square in squares else " ."
        row += " |"
        print(row)
    print("  +-----------------+")
    print("    a b c d e f g h\n")
    
def index_to_square(index):
    """Convert a 0-63 index to a chessboard square in algebraic notation."""
    rank = (index // 8) + 1
    file = chr(index % 8 + ord('a'))
    return f"{file}{rank}"
    
def get_standard_algebraic(move_or_loc):
    """Convert a list of moves from index notation to algebraic notation."""
    if isinstance(move_or_loc, tuple):
        from_sq, to_sq, promo = move_or_loc
        from_square_algebraic = index_to_square(from_sq)
        to_square_algebraic = index_to_square(to_sq)
        return f"{from_square_algebraic}{to_square_algebraic}"
    else:
        return index_to_square(move_or_loc)
    
if __name__=="__main__":
    print()

    gs = BitboardGameState()

    move_state_type = types.Tuple((
        uint64, uint64, uint64, uint64, uint64, uint64,   # white pieces
        uint64, uint64, uint64, uint64, uint64, uint64,   # black pieces
        types.UniTuple(types.int8, 4),                    # castling_rights
        types.int32,                                      # en_passant_target
        types.int32,                                      # halfmove_clock
        types.int32                                       # fullmove_number
    ))

    n = 0
    move_info = List.empty_list(move_state_type)
    for mv1 in generate_all_moves(gs):
        move = apply_move_numba(gs, mv1)
        move_info.append(move)
        update_occupancies_numba(gs)
        gs.white_to_move = not gs.white_to_move
        for mv2 in generate_all_moves(gs):
            move = apply_move_numba(gs, mv1)
            move_info.append(move)
            update_occupancies_numba(gs)
            gs.white_to_move = not gs.white_to_move
            for mv3 in generate_all_moves(gs):
                n += 1

            undo_move_numba(gs, move_info)
            update_occupancies_numba(gs)
            gs.white_to_move = not gs.white_to_move

        undo_move_numba(gs, move_info)
        update_occupancies_numba(gs)
        gs.white_to_move = not gs.white_to_move

    print("Moves @ depth=3:", n)
    print()

    gs = BitboardGameState()
    move = apply_move_numba(gs, (10,26,0)) #g1h3
    update_occupancies_numba(gs)
    gs.white_to_move = not gs.white_to_move
    move = apply_move_numba(gs, (51,43,0)) #g1h3
    update_occupancies_numba(gs)
    gs.white_to_move = not gs.white_to_move

    empty = ~gs.occupied
    enemy = gs.black_occupancy if gs.white_to_move else gs.white_occupancy
    pawns = gs.white_pawns if gs.white_to_move else gs.black_pawns
    occupancy = gs.white_occupancy if gs.white_to_move else gs.black_occupancy
    if gs.white_to_move:
        single_push = (pawns << 8) & empty
        double_push = ((single_push & rank_mask(2)) << 8) & empty
        left_attacks = (pawns << 7) & enemy & ~file_mask(7)
        right_attacks = (pawns << 9) & enemy & ~file_mask(0)
    else:
        single_push = (pawns >> 8) & empty
        double_push = ((single_push & rank_mask(5)) >> 8) & empty
        left_attacks = (pawns >> 9) & enemy & ~file_mask(7)
        right_attacks = (pawns >> 7) & enemy & ~file_mask(0)

    print()
    print_board(gs)
    print()

    print("Pawn moves bb:")
    moves = (left_attacks | right_attacks | single_push | double_push)
    squares = bb_to_squares(pawns)
    print_bitboard(squares, moves)
    print("Pawn generated moves:")
    for m in generate_pawn_moves(gs, pawns, gs.white_to_move):
        print(get_standard_algebraic(m))
    print()

    rooks = gs.white_rooks if gs.white_to_move else gs.black_rooks
    print("Rook moves bb:")
    moves = 0
    squares = bb_to_squares(rooks)
    for sq in squares:
        moves |= rook_attacks(sq, gs.occupied) & ~occupancy
    print_bitboard(squares, moves)
    print("Rook generated moves:")
    for m in generate_rook_moves(gs, rooks, gs.white_to_move):
        print(get_standard_algebraic(m))
    print()

    knights = gs.white_knights if gs.white_to_move else gs.black_knights
    print("Knight moves bb:")
    moves = 0
    squares = bb_to_squares(knights)
    for sq in squares:
        moves |= knight_attacks(sq) & ~occupancy

    print_bitboard(squares, moves)
    for m in generate_knight_moves(gs, knights, gs.white_to_move):
        print(get_standard_algebraic(m))
    print()

    bishops = gs.white_bishops if gs.white_to_move else gs.black_bishops
    print("Bishop moves bb:")
    squares = bb_to_squares(bishops)
    moves = 0
    for sq in squares:
        moves |= bishop_attacks(sq, gs.occupied) & ~occupancy
    print_bitboard(squares, moves)
    for m in generate_bishop_moves(gs, bishops, gs.white_to_move):
        print(get_standard_algebraic(m))
    print()

    queens = gs.white_queens if gs.white_to_move else gs.black_queens
    print("Queen moves bb:")
    squares = bb_to_squares(queens)
    moves = 0
    for sq in squares:
        moves |= queen_attacks(sq, gs.occupied) & ~occupancy
    print_bitboard(squares, moves)
    for m in generate_queen_moves(gs, queens, gs.white_to_move):
        print(get_standard_algebraic(m))
    print()

    king = gs.white_king if gs.white_to_move else gs.black_king
    print("King moves bb:")
    squares = bb_to_squares(king)
    enemy_attacks = attack_map_numba(gs, not gs.white_to_move)
    moves = 0
    for sq in squares:
        moves |= king_attacks(sq) & ~occupancy & ~enemy_attacks
    print_bitboard(squares, moves)
    for m in generate_king_moves(gs, king, gs.white_to_move):
        print(get_standard_algebraic(m))
    print()

    print(board_str)