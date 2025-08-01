# codes/view_data.py (Complete Version)

import os
import chess
import numpy as np

# --- Configuration ---
# Set the path to your main project folder
PROJECT_PATH = "C:/Users/GFG0645/Desktop/chess_engine_project" 
# The folder with your depth-12 data
DATA_DIR = os.path.join(PROJECT_PATH, "prepared_data_d12_multipv") 

# --- Helper functions to understand the data ---

def create_move_map():
    """
    Recreates the exact same move-to-index map that was used to generate the data.
    This is essential for correctly interpreting the policy array.
    """
    moves = []
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            if from_sq != to_sq: moves.append(chess.Move(from_sq, to_sq))
    for from_sq in chess.SQUARES:
        if chess.square_rank(from_sq) == 6:
             for to_sq in range(chess.A8, chess.H8 + 1):
                if abs(chess.square_file(from_sq) - chess.square_file(to_sq)) <= 1: moves.append(chess.Move(from_sq, to_sq, promotion=chess.QUEEN))
        if chess.square_rank(from_sq) == 1:
            for to_sq in range(chess.A1, chess.H1 + 1):
                if abs(chess.square_file(from_sq) - chess.square_file(to_sq)) <= 1: moves.append(chess.Move(from_sq, to_sq, promotion=chess.QUEEN))
    unique_uci_moves = sorted(list(set([m.uci() for m in moves])))
    # We need the reverse map: from index back to move
    return {i: move_string for i, move_string in enumerate(unique_uci_moves)}

INDEX_TO_MOVE = create_move_map()

def board_from_input(input_data):
    """
    Takes a 25x8x8 numpy array and converts it back into a python-chess Board object.
    """
    board = chess.Board(fen=None) # Create a completely empty board
    
    # Place pieces based on the first 12 planes
    for plane_idx in range(12):
        piece_type = (plane_idx % 6) + 1
        piece_color = chess.WHITE if plane_idx < 6 else chess.BLACK
        piece = chess.Piece(piece_type, piece_color)
        
        for r in range(8):
            for c in range(8):
                if input_data[plane_idx, r, c] == 1:
                    square = chess.square(c, r)
                    board.set_piece_at(square, piece)
    
    # Set castling rights and turn from the remaining planes
    board.has_kingside_castling_rights = lambda color: input_data[12, 0, 0] == 1 if color == chess.WHITE else input_data[14, 0, 0] == 1
    board.has_queenside_castling_rights = lambda color: input_data[13, 0, 0] == 1 if color == chess.WHITE else input_data[15, 0, 0] == 1
    board.turn = chess.WHITE if input_data[16, 0, 0] == 1 else chess.BLACK
    
    return board

# --- Main Viewing Logic ---

try:
    print(f"Loading dataset from: {DATA_DIR}")
    inputs = np.load(os.path.join(DATA_DIR, "inputs.npy"))
    targets = np.load(os.path.join(DATA_DIR, "targets.npy"))
    policies = np.load(os.path.join(DATA_DIR, "policies.npy"))
    print(f"✅ Successfully loaded {len(inputs)} positions.")

    # <<< CHOOSE WHICH POSITIONS TO VIEW >>>
    # Use a list to view multiple positions, e.g., [0, 1, 98, 99]

    positions_to_view = 30
    # <<< -------------------------------- >>>

    for position_index in range(positions_to_view):
        if 0 <= position_index < len(inputs):
            # Get the data for the chosen position
            board_data = inputs[position_index]
            value_data = targets[position_index]
            policy_data = policies[position_index]

            # Reconstruct the board
            board = board_from_input(board_data)
            
            # Find the best move from the policy data
            best_move_index = np.argmax(policy_data)
            best_move_uci = INDEX_TO_MOVE[best_move_index]

            print("\n" + "="*40)
            print(f"          VIEWING POSITION #{position_index}")
            print("="*40)
            
            # Print the reconstructed board
            print(board)
            
            print("\n--- Position Info & Data ---")
            print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
            print(f"FEN: {board.fen()}")
            print("-" * 20)
            print(f"Value (Win Chance): {value_data:.4f}")
            print(f"Policy (Best Move Suggested): {best_move_uci}")
            print("="*40)

        else:
            print(f"❌ ERROR: Invalid position index: {position_index}. Please choose a number between 0 and {len(inputs) - 1}.")

except FileNotFoundError:
    print(f"❌ ERROR: Could not find the .npy files in the directory: {DATA_DIR}")
    print("Please make sure the data has been generated and the path is correct.")