import os
import chess
import numpy as np

# --- Configuration ---
# Make sure these paths match your project structure
DRIVE_PROJECT_PATH = "C:/Users/GFG0645/Desktop/chess_engine_project" # For local VS Code
# DRIVE_PROJECT_PATH = '/content/drive/MyDrive/chess' # For Colab
DATA_DIR = os.path.join(DRIVE_PROJECT_PATH, "prepared_data_d12") # The folder with your depth-12 data

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
    board = chess.Board(None) # Create an empty board
    # Piece planes (0-11)
    for plane_idx in range(12):
        piece_type = (plane_idx % 6) + 1
        piece_color = chess.WHITE if plane_idx < 6 else chess.BLACK
        piece = chess.Piece(piece_type, piece_color)
        
        for r in range(8):
            for c in range(8):
                if input_data[plane_idx, r, c] == 1:
                    square = chess.square(c, r)
                    board.set_piece_at(square, piece)
    
    # Castling and turn planes
    board.has_kingside_castling_rights = lambda c: input_data[12, 0, 0] if c == chess.WHITE else input_data[14, 0, 0]
    board.has_queenside_castling_rights = lambda c: input_data[13, 0, 0] if c == chess.WHITE else input_data[15, 0, 0]
    board.turn = chess.WHITE if input_data[16, 0, 0] == 1 else chess.BLACK
    return board

# --- Main Viewing Logic ---

try:
    print(f"Loading dataset from: {DATA_DIR}")
    inputs = np.load(os.path.join(DATA_DIR, "inputs.npy"))
    targets = np.load(os.path.join(DATA_DIR, "targets.npy"))
    policies = np.load(os.path.join(DATA_DIR, "policies.npy"))
    print(f"✅ Successfully loaded {len(inputs)} positions.")

    # <<< CHOOSE WHICH POSITION TO VIEW >>>
    position_to_view = 1000 # You can change this number to see any position in your dataset
    # <<< ----------------------------- >>>

    if 0 <= position_to_view < len(inputs):
        # Get the data for the chosen position
        board_data = inputs[position_to_view]
        value_data = targets[position_to_view]
        policy_data = policies[position_to_view]

        # Reconstruct the board
        board = board_from_input(board_data)
        
        # Find the best move from the policy data
        best_move_index = np.argmax(policy_data)
        best_move_uci = INDEX_TO_MOVE[best_move_index]

        print("\n" + "="*40)
        print(f"          VIEWING POSITION #{position_to_view}")
        print("="*40)
        
        # Print the board
        print(board)
        
        print("\n--- DATA ANALYSIS ---")
        print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
        print(f"Value (Win Chance): {value_data:.4f}")
        print(f"Policy (Best Move Suggested): {best_move_uci}")
        print("="*40)

    else:
        print(f"❌ ERROR: Invalid position index. Please choose a number between 0 and {len(inputs) - 1}.")

except FileNotFoundError:
    print(f"❌ ERROR: Could not find the .npy files in the directory: {DATA_DIR}")
    print("Please make sure you have generated the dataset and the path is correct.")