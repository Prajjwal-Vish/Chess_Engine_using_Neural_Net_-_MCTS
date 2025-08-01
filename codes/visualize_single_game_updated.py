import os
import chess
import numpy as np

# --- Configuration ---
# Set the path to your main project folder
PROJECT_PATH = "C:/Users/GFG0645/Desktop/chess_engine_project" 
# IMPORTANT: Point this to the new multi-move data directory
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
    # This logic correctly interprets the 8x8 planes
    board.has_kingside_castling_rights = lambda color: np.any(input_data[12, :, :]) if color == chess.WHITE else np.any(input_data[14, :, :])
    board.has_queenside_castling_rights = lambda color: np.any(input_data[13, :, :]) if color == chess.WHITE else np.any(input_data[15, :, :])
    board.turn = chess.WHITE if np.any(input_data[16, :, :]) else chess.BLACK
    
    return board

# --- Main Viewing Logic ---
try:
    print(f"Loading dataset from: {DATA_DIR}")
    inputs = np.load(os.path.join(DATA_DIR, "inputs.npy"))
    targets = np.load(os.path.join(DATA_DIR, "targets.npy"))
    policies = np.load(os.path.join(DATA_DIR, "policies.npy"))
    print(f"✅ Successfully loaded {len(inputs)} positions.")

    # <<< CHOOSE WHICH POSITIONS TO VIEW >>>
    # Use a list to view multiple specific positions
    positions_to_view = 20
    # <<< -------------------------------- >>>

    for position_index in range(positions_to_view):
        if 0 <= position_index < len(inputs):
            # Get the data for the chosen position
            board_data = inputs[position_index]
            value_data = targets[position_index]
            policy_data = policies[position_index]

            # Reconstruct the board
            board = board_from_input(board_data)
            
            print("\n" + "="*50)
            print(f"          VIEWING POSITION #{position_index}")
            print("="*50)
            
            # Print the reconstructed board
            print(board)
            
            print("\n--- Position Info & Data ---")
            print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
            print(f"FEN: {board.fen()}")
            print("-" * 20)
            print(f"Value (Win Chance): {value_data:.4f}")
            print("Policy (Suggested Moves & Probabilities):")

            # --- NEW: Find and display all moves with non-zero probability ---
            # Find all indices in the policy array that have a value
            move_indices = np.where(policy_data > 0)[0]
            
            move_probs = []
            for idx in move_indices:
                # Get the move string and its probability
                move_probs.append((INDEX_TO_MOVE[idx], policy_data[idx]))
            
            # Sort the moves by their probability, from highest to lowest
            move_probs.sort(key=lambda item: item[1], reverse=True)

            if not move_probs:
                print("  No moves found in policy for this position.")
            else:
                # Print each move and its probability, formatted as a percentage
                for move, prob in move_probs:
                    print(f"  - {move}: {prob:.2%}")
            print("="*50)

        else:
            print(f"❌ ERROR: Invalid position index: {position_index}. Please choose a number between 0 and {len(inputs) - 1}.")

except FileNotFoundError:
    print(f"❌ ERROR: Could not find the .npy files in the directory: {DATA_DIR}")
    print("Please make sure the multi-move data has been generated and the path is correct.")
