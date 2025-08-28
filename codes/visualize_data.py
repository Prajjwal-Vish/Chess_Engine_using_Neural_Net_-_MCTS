# codes/view_patched_data.py

import os
import chess
import numpy as np

# --- Configuration ---
PROJECT_PATH = "C:/Users/GFG0645/Desktop/chess_engine_project" 
# The folder with your NEW 24-plane patched dataset
DATA_DIR = os.path.join(PROJECT_PATH, "prepared_data_24_layers") 

# --- Helper functions to understand the data ---

def create_move_map():
    """Recreates the move-to-index map to interpret the policy array."""
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
    return {i: move_string for i, move_string in enumerate(unique_uci_moves)}

INDEX_TO_MOVE = create_move_map()

def board_from_input(input_data):
    """Takes a 24x8x8 numpy array and converts it back into a python-chess Board object."""
    board = chess.Board(fen=None)
    for plane_idx in range(12):
        piece_type = (plane_idx % 6) + 1
        piece_color = chess.WHITE if plane_idx < 6 else chess.BLACK
        piece = chess.Piece(piece_type, piece_color)
        for r in range(8):
            for c in range(8):
                if input_data[plane_idx, r, c] == 1:
                    board.set_piece_at(chess.square(c, r), piece)
    
    board.turn = chess.WHITE if np.any(input_data[16]) else chess.BLACK
    # Note: FEN reconstruction from planes is complex; this provides the visual board.
    return board

# --- Main Viewing Logic ---
try:
    print(f"Loading dataset from: {DATA_DIR}")
    inputs = np.load(os.path.join(DATA_DIR, "inputs.npy"))
    targets = np.load(os.path.join(DATA_DIR, "targets.npy"))
    policies = np.load(os.path.join(DATA_DIR, "policies.npy"))
    print(f"✅ Successfully loaded {len(inputs)} positions.")

    # <<< CHOOSE WHICH POSITIONS TO VIEW >>>
    positions_to_view = [121,32112] 
    # <<< -------------------------------- >>>

    # Define descriptions for each of the 24 planes
    plane_descriptions = [
        "Plane 0: White Pawns", "Plane 1: White Knights", "Plane 2: White Bishops",
        "Plane 3: White Rooks", "Plane 4: White Queens", "Plane 5: White King",
        "Plane 6: Black Pawns", "Plane 7: Black Knights", "Plane 8: Black Bishops",
        "Plane 9: Black Rooks", "Plane 10: Black Queens", "Plane 11: Black King",
        "Plane 12: White Kingside Castling Rights", "Plane 13: White Queenside Castling Rights",
        "Plane 14: Black Kingside Castling Rights", "Plane 15: Black Queenside Castling Rights",
        "Plane 16: Side to Move (1=White, 0=Black)", "Plane 17: Total Move Count",
        "Plane 18: Fifty-Move Rule Counter", "Plane 19: En Passant Square",
        "Plane 20: Repetition Count (1)", "Plane 21: Repetition Count (2)",
        "Plane 22: Material Difference (Current Player's View)", "Plane 23: Total Material"
    ]


    for position_index in positions_to_view:
        if 0 <= position_index < len(inputs):
            board_data = inputs[position_index]
            value_data = targets[position_index]
            policy_data = policies[position_index]
            board = board_from_input(board_data)
            
            print("\n" + "#"*60)
            print(f"##########          VIEWING POSITION #{position_index}          ##########")
            print("#"*60)
            
            print("\n--- Board Representation ---")
            print(board)
            
            # Print fen
            print(f"FEN: {board.fen()}")

            print("\n--- Value and Policy Data ---")
            print(f"Value (Win Chance): {value_data:.4f}")
            print("Policy (Top 5 Moves & Probabilities):")
            move_indices = np.where(policy_data > 0)[0]
            move_probs = sorted([(INDEX_TO_MOVE[idx], policy_data[idx]) for idx in move_indices], key=lambda item: item[1], reverse=True)
            for move, prob in move_probs[:5]:
                print(f"  - {move}: {prob:.2%}")

            print("\n" + "="*60)
            print("          DETAILED INPUT PLANE ANALYSIS")
            print("="*60)
            for i in range(board_data.shape[0]):
                plane = board_data[i]
                description = plane_descriptions[i]
                
                # Provide a summary of the plane's content
                summary = ""
                if np.all(plane == 0):
                    summary = "All 0s"
                elif np.all(plane == 1):
                    summary = "All 1s"
                elif np.sum(plane) > 0 and i < 22: # Piece/state planes
                     summary = f"Sparse data ({int(np.sum(plane))} '1's)"
                else: # Scaled value planes
                    unique_vals = np.unique(plane)
                    if len(unique_vals) == 1:
                        summary = f"Constant value: {unique_vals[0]:.4f}"
                    else:
                        summary = "Varied data"
                
                print(f"\n--- {description} --- (Summary: {summary})")
                print(f"Dimension: {plane.shape}")
                # Print the actual 8x8 grid
                print(np.round(plane, 2))
            print("="*60)

        else:
            print(f"❌ ERROR: Invalid position index: {position_index}.")

except FileNotFoundError:
    print(f"❌ ERROR: Could not find the .npy files in the directory: {DATA_DIR}")

