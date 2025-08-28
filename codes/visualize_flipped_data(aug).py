# codes/visualize_augmentation.py

import os
import chess
import numpy as np
import torch # We need torch for the flip operation

# --- Configuration ---
PROJECT_PATH = "C:/Users/GFG0645/Desktop/chess_engine_project" 
DATA_DIR = os.path.join(PROJECT_PATH, "prepared_data_final_move_hist") 

# --- Helper functions (identical to your other scripts) ---

def create_move_maps():
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
    move_to_index = {move: i for i, move in enumerate(unique_uci_moves)}
    index_to_move = {i: move for move, i in move_to_index.items()}
    
    # Create the map for flipping policy vectors
    flip_map = np.zeros(len(unique_uci_moves), dtype=np.int32)
    for i, uci in enumerate(unique_uci_moves):
        move = chess.Move.from_uci(uci)
        flipped_from = chess.square_mirror(move.from_square)
        flipped_to = chess.square_mirror(move.to_square)
        flipped_move = chess.Move(flipped_from, flipped_to, move.promotion)
        if flipped_move.uci() in move_to_index:
            flip_map[i] = move_to_index[flipped_move.uci()]
        else:
            flip_map[i] = i
            
    return index_to_move, flip_map

INDEX_TO_MOVE, POLICY_FLIP_MAP = create_move_maps()

def board_from_input(input_data):
    board = chess.Board(fen=None)
    for plane_idx in range(12):
        piece_type = (plane_idx % 6) + 1
        piece_color = chess.WHITE if plane_idx < 6 else chess.BLACK
        piece = chess.Piece(piece_type, piece_color)
        for r in range(8):
            for c in range(8):
                if input_data[plane_idx, r, c] == 1:
                    board.set_piece_at(chess.square(c, r), piece)
    board.turn = chess.WHITE if np.any(input_data[16, :, :]) else chess.BLACK
    return board

def display_policy(policy_data):
    """Helper to find and display the top moves from a policy vector."""
    move_indices = np.where(policy_data > 0)[0]
    move_probs = []
    for idx in move_indices:
        move_probs.append((INDEX_TO_MOVE[idx], policy_data[idx]))
    move_probs.sort(key=lambda item: item[1], reverse=True)
    for move, prob in move_probs[:5]: # Display top 5
        print(f"  - {move}: {prob:.2%}")

# --- Main Visualization Logic ---
try:
    print(f"Loading dataset from: {DATA_DIR}")
    inputs = np.load(os.path.join(DATA_DIR, "inputs.npy"))
    targets = np.load(os.path.join(DATA_DIR, "targets_patched.npy"))
    policies = np.load(os.path.join(DATA_DIR, "policies.npy"))
    print(f"✅ Successfully loaded {len(inputs)} positions.")

    # <<< CHOOSE WHICH POSITION TO VISUALIZE >>>
    position_to_view = 10
    # <<< ---------------------------------- >>>

    # --- 1. Original Data ---
    original_input = inputs[position_to_view]
    original_target = targets[position_to_view]
    original_policy = policies[position_to_view]
    
    original_board = board_from_input(original_input)

    print("\n" + "="*50)
    print(f"        ORIGINAL DATA (Position #{position_to_view})")
    print("="*50)
    print(original_board)
    print("\n--- Original Data ---")
    print(f"Value: {original_target:.4f}")
    print("Policy (Top Moves):")
    display_policy(original_policy)
    
    # --- 2. Augmented (Flipped) Data ---
    # Convert to PyTorch tensors to use the flip function
    input_tensor = torch.from_numpy(original_input)
    policy_tensor = torch.from_numpy(original_policy)

    # Perform the augmentation
    flipped_input_tensor = torch.flip(input_tensor, [2]) # Flip horizontally (on the 'columns' axis)
    flipped_target = -original_target # Invert the value
    flipped_policy = policy_tensor[POLICY_FLIP_MAP] # Remap the policy vector

    # Convert back to numpy for visualization
    flipped_input = flipped_input_tensor.numpy()
    flipped_board = board_from_input(flipped_input)
    
    print("\n" + "="*50)
    print(f"        AUGMENTED DATA (Flipped Position #{position_to_view})")
    print("="*50)
    print(flipped_board)
    print("\n--- Flipped Data ---")
    print(f"Value: {flipped_target:.4f}")
    print("Policy (Top Moves):")
    display_policy(flipped_policy.numpy())
    print("="*50)

except FileNotFoundError:
    print(f"❌ ERROR: Could not find .npy files in {DATA_DIR}")
