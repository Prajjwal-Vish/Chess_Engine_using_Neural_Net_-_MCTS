# codes/core/utils.py (Final Correction: Circular import removed)

import chess
import numpy as np

# --- THIS LINE WAS REMOVED ---
# from codes.core.mcts import MCTS  <-- This caused the circular import. It's not needed here.

def _get_material_value(board):
    """Calculates the material value of the board from White's perspective."""
    material = 0
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9
    }
    for piece_type in piece_values:
        material += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        material -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
    return material

def create_move_map():
    """
    Creates a stable dictionary mapping UCI moves to an index.
    This version is specifically engineered to produce a POLICY_SIZE of 4076
    by only considering Queen and Knight promotions, matching the trained model.
    """
    moves = []
    promotions = [chess.QUEEN, chess.KNIGHT]
    
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            if from_sq == to_sq:
                continue
            
            from_file = chess.square_file(from_sq)
            from_rank = chess.square_rank(from_sq)
            to_file = chess.square_file(to_sq)
            to_rank = chess.square_rank(to_sq)

            is_promo = False
            if from_rank == 6 and to_rank == 7 and abs(from_file - to_file) <= 1:
                is_promo = True
            elif from_rank == 1 and to_rank == 0 and abs(from_file - to_file) <= 1:
                is_promo = True

            if is_promo:
                for promo_piece in promotions:
                    moves.append(chess.Move(from_sq, to_sq, promotion=promo_piece))
            else:
                moves.append(chess.Move(from_sq, to_sq))

    unique_uci_moves = sorted(list(set([m.uci() for m in moves])))
    return {move: i for i, move in enumerate(unique_uci_moves)}

MOVE_TO_INDEX = create_move_map()
POLICY_SIZE = len(MOVE_TO_INDEX)

def board_to_input(board, history):
    """Converts a board state into the definitive 24x8x8 numpy array."""
    input_data = np.zeros((24, 8, 8), dtype=np.float32)

    for sq, p in board.piece_map().items():
        r, c = chess.square_rank(sq), chess.square_file(sq)
        p_idx = p.piece_type - 1 + (6 if p.color == chess.BLACK else 0)
        input_data[p_idx, r, c] = 1

    input_data[12, :, :] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
    input_data[13, :, :] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
    input_data[14, :, :] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
    input_data[15, :, :] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0
    input_data[16, :, :] = 1 if board.turn == chess.WHITE else 0
    input_data[17, :, :] = board.fullmove_number / 100.0
    input_data[18, :, :] = board.halfmove_clock / 100.0
    if board.ep_square:
        r, c = chess.square_rank(board.ep_square), chess.square_file(board.ep_square)
        input_data[19, r, c] = 1
    
    current_fen = board.fen()
    rep_count = history.count(current_fen)
    if rep_count >= 1:
        input_data[20, :, :] = 1.0
    if rep_count >= 2:
        input_data[21, :, :] = 1.0

    material_diff = _get_material_value(board)
    input_data[22, :, :] = np.tanh(material_diff / 10.0)
    
    white_material = 39 - _get_material_value(board.copy().mirror())
    black_material = 39 - _get_material_value(board)
    total_material = white_material + black_material
    input_data[23, :, :] = total_material / 78.0

    return input_data
