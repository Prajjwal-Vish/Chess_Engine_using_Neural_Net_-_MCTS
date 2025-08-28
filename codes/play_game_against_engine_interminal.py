# # codes/play_cli.py (Updated for 24-layer model)

# import os
# import sys
# import chess
# import chess.pgn
# import torch
# import time

# # --- Add the project's root directory to the Python path ---
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(project_root)

# # --- CORRECT: Import the classes from our core modules ---
# from codes.core.model import AlphaZeroNet # Assuming this is the corrected model
# from codes.core.mcts import MCTS         # Assuming this is the corrected MCTS
# from codes.core.utils import POLICY_SIZE # This should be the correct size (e.g., 4076)

# # --- Main Game Logic ---
# if __name__ == '__main__':
    
#     # --- CONFIGURATION ---
#     MODEL_FILENAME = "chess_alphazero_1.1.pth" # Your current model file
#     MCTS_SIMULATIONS = 800 # Number of simulations per move. Increase for stronger play.
#     HUMAN_PLAYS_AS = chess.WHITE # Change to chess.BLACK to play as Black
    
#     # --- Set your project path ---
#     # Use your actual project path where the model is located
#     PROJECT_PATH = r"C:\Users\prajjwal\Desktop\chess_engine_project" 
#     MODEL_PATH = os.path.join(PROJECT_PATH, "models", MODEL_FILENAME)
    
#     print("Loading trained model...")
#     # Use CPU for inference if you don't have a dedicated GPU for playing
#     device = torch.device("cpu") 
#     try:
#         # --- CORRECTED: Initialize with 24 input channels ---
#         model = AlphaZeroNet(num_residual_blocks=20, policy_size=POLICY_SIZE, input_channels=24).to(device)
#         model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
#         model.eval()
#         print(f"✅ Model '{MODEL_FILENAME}' loaded successfully.")
#     except FileNotFoundError:
#         print(f"❌ ERROR: Model file not found at '{MODEL_PATH}'. Please check the path.")
#         sys.exit(1)

#     # Initialize MCTS with the loaded model
#     mcts = MCTS(model, MOVE_TO_INDEX)
    
#     board = chess.Board()
#     game = chess.pgn.Game()
#     node = game
    
#     # --- NEW: History tracking is now required for the model input ---
#     history = []

#     while not board.is_game_over():
#         print("\n" + "="*30)
#         print(board)
#         print("="*30)

#         if board.turn == HUMAN_PLAYS_AS:
#             # --- Human's Turn ---
#             move = None
#             while move is None:
#                 move_str = input("Enter your move (e.g., e2e4): ")
#                 try:
#                     move = chess.Move.from_uci(move_str)
#                     if move not in board.legal_moves:
#                         print("!!! Illegal move, please try again. !!!")
#                         move = None
#                 except ValueError:
#                     print("!!! Invalid move format, please use UCI (e.g., e2e4). !!!")
#                     move = None
#         else:
#             # --- Engine's Turn ---
#             print("Engine is thinking...")
#             start_time = time.time()
            
#             # --- CORRECTED: Pass the history to the MCTS search ---
#             # Note: Your MCTS get_move_analysis must accept the history list
#             best_move, top_moves = mcts.get_move_analysis(board, history, MCTS_SIMULATIONS)
            
#             end_time = time.time()
            
#             move = best_move
            
#             if move is None:
#                 print("Engine returned no move. Game might be over.")
#                 break
            
#             print(f"Engine chose: {move.uci()} (took {end_time - start_time:.2f}s)")
#             print("\n--- Top 5 Considered Moves (by visit count) ---")
#             for i, (m, visits) in enumerate(top_moves):
#                 print(f"{i+1}. {m.uci()} ({visits} visits)")
        
#         # Update board state and history
#         board.push(move)
#         history.append(board.fen()) # Add the new position to history
#         node = node.add_variation(move)

#     print("\n" + "="*30)
#     print("         GAME OVER")
#     print(f"Result: {board.result()}")
#     print("="*30)

#     print("\n--- Final Game PGN ---")
#     print(game)


# New code for the saved state model

# play_cli_final.py
# A script to play a game against your fully trained AlphaZero model.

import os
import sys
import chess
import chess.pgn
import torch
import time

# --- Add the project's root directory to the Python path ---
# This assumes your script is in the 'codes' folder.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# --- Import your classes from the core modules ---
from codes.core.model import AlphaZeroNet
from codes.core.mcts import MCTS
from codes.core.utils import POLICY_SIZE, board_to_input, MOVE_TO_INDEX

# --- Main Game Logic ---
if __name__ == '__main__':
    
    # --- CONFIGURATION ---
    # Point this to your new, larger checkpoint file.
    CHECKPOINT_FILENAME = "chess_alphazero_checkpoint.pth" 
    MCTS_SIMULATIONS = 800  # Increase for stronger play, decrease for faster moves.
    HUMAN_PLAYS_AS = chess.WHITE # Change to chess.BLACK to play as Black
    
    # --- Set your project path ---
    PROJECT_PATH = "C:/Users/GFG0645/Desktop/chess_engine_project" 
    CHECKPOINT_PATH = os.path.join(PROJECT_PATH, "models", CHECKPOINT_FILENAME)
    
    print("Loading trained model from checkpoint...")
    device = torch.device("cpu")
    try:
        # Initialize the model architecture first
        model = AlphaZeroNet(num_residual_blocks=20, policy_size=POLICY_SIZE, input_channels=24).to(device)
        
        # --- THIS IS THE KEY CHANGE ---
        # 1. Load the entire checkpoint dictionary.
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        # 2. Extract and load only the model's state dictionary.
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval() # Set the model to evaluation mode
        print(f"✅ Model loaded successfully from '{CHECKPOINT_FILENAME}'.")
    except FileNotFoundError:
        print(f"❌ ERROR: Checkpoint file not found at '{CHECKPOINT_PATH}'. Please check the path.")
        sys.exit(1)
    except KeyError:
        print(f"❌ ERROR: The checkpoint file is missing the 'model_state_dict' key.")
        sys.exit(1)

    # Initialize MCTS with the loaded model and move map
    mcts = MCTS(model, MOVE_TO_INDEX)
    
    board = chess.Board()
    game = chess.pgn.Game()
    node = game
    history = []

    while not board.is_game_over():
        print("\n" + "="*30)
        print(board)
        print("="*30)

        if board.turn == HUMAN_PLAYS_AS:
            # --- Human's Turn ---
            move = None
            while move is None:
                move_str = input("Enter your move (e.g., e2e4): ")
                try:
                    move = chess.Move.from_uci(move_str)
                    if move not in board.legal_moves:
                        print("!!! Illegal move, please try again. !!!")
                        move = None
                except ValueError:
                    print("!!! Invalid move format, please use UCI (e.g., e2e4). !!!")
        else:
            # --- Engine's Turn ---
            print("Engine is thinking...")
            start_time = time.time()
            
            best_move, top_moves = mcts.get_move_analysis(board, history, MCTS_SIMULATIONS)
            
            end_time = time.time()
            
            move = best_move
            
            if move is None:
                print("Engine returned no move. Game might be over.")
                break
            
            print(f"Engine chose: {move.uci()} (took {end_time - start_time:.2f}s)")
            print("\n--- Top 5 Considered Moves (by visit count) ---")
            for i, (m, visits) in enumerate(top_moves):
                print(f"{i+1}. {m.uci()} ({visits} visits)")
        
        board.push(move)
        history.append(board.fen())
        node = node.add_variation(move)

    print("\n" + "="*30)
    print("         GAME OVER")
    print(f"Result: {board.result()}")
    print("="*30)

    print("\n--- Final Game PGN ---")
    print(game)








