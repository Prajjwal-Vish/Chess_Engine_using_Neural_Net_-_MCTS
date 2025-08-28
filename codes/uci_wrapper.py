# codes/uci_wrapper.py
# A clean, refactored script to run the engine with a UCI-compatible GUI like Arena.
# This version imports classes from the core modules for better code organization.

import chess
import chess.engine
import torch
import os
import sys

# ======================================================================================
# SECTION 1: SETUP PYTHON PATH & IMPORTS
# This allows the script to find and import from your 'core' module.
# ======================================================================================

# Add the project's root directory (chess_engine_project) to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from codes.core.model import AlphaZeroNet
from codes.core.mcts import MCTS
from codes.core.utils import board_to_input, MOVE_TO_INDEX, POLICY_SIZE

# ======================================================================================
# SECTION 2: UCI ENGINE CLASS
# This class handles communication with the GUI.
# ======================================================================================

class MyEngine(chess.engine.SimpleEngine):
    def __init__(self):
        # --- CORRECTED: Configuration is now handled inside __init__ ---
        # This makes the class compatible with the new chess.engine.serve() method.
        self.device = DEVICE
        self.mcts_sims = MCTS_SIMULATIONS
        self.model = self._load_model(MODEL_PATH)
        self.mcts = MCTS(self.model, MOVE_TO_INDEX)

    def _load_model(self, model_path):
        """Loads the trained neural network model."""
        model = AlphaZeroNet(num_residual_blocks=20, policy_size=POLICY_SIZE, input_channels=24)
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            sys.stderr.write(f"ERROR: Model file not found at {model_path}\n")
            sys.stderr.flush()
            
        model.to(self.device)
        model.eval()
        return model

    def search(self, board: chess.Board, *args, **kwargs) -> chess.engine.PlayResult:
        """The main search function called by the GUI."""
        
        # Reconstruct game history for repetition checks
        history = []
        temp_board = board.copy()
        while temp_board.move_stack:
            history.append(temp_board.fen())
            temp_board.pop()
        history.reverse()

        # Get the best move from MCTS
        best_move = self.mcts.get_move(board, history, self.mcts_sims)
        
        # Return the best move found to the GUI
        return chess.engine.PlayResult(best_move, None)

# ======================================================================================
# SECTION 3: MAIN EXECUTION BLOCK
# ======================================================================================

if __name__ == "__main__":
    # --- CONFIGURATION (These are now global variables for the class) ---
    MODEL_PATH = "C:/Users/GFG0645/Desktop/chess_engine_project/models/chess_alphazero_1.1.pth"
    MCTS_SIMULATIONS = 800 
    DEVICE = torch.device("cpu") 
    
    # --- CORRECTED: Start the engine using the new method ---
    # The chess.engine.serve() function now handles the main loop.
    # It will automatically create an instance of your MyEngine class.
    chess.engine.serve(MyEngine)
