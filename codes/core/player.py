# codes/core/player.py (Corrected)

import torch
import chess
import numpy as np

from .mcts import MCTS
from .utils import board_to_input, MOVE_TO_INDEX

class Player:
    def __init__(self, model, player_type='mcts', name="Player", mcts_sims=800):
        self.model = model
        self.player_type = player_type
        self.name = name
        self.mcts_sims = mcts_sims
        self.device = next(model.parameters()).device
        if self.player_type == 'mcts':
            self.mcts_instance = MCTS(model)

    def get_move(self, board, history, move_count): # Added history parameter
        if self.player_type == 'mcts':
            temperature = 1.0 if move_count < 15 else 0.0
            return self.mcts_instance.get_move(board, self.mcts_sims, temperature)
        
        # --- THIS 'ELSE' BLOCK IS NOW CORRECTED ---
        else: # 'nn_only' player type
            with torch.no_grad():
                # Pass the history argument (even if empty)
                input_tensor = torch.tensor(
                    board_to_input(board, history), dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                
                # Model now returns raw logits
                _, policy_logits = self.model(input_tensor)
                
                # Apply softmax to logits to get probabilities
                policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
                
                legal_moves = list(board.legal_moves)
                move_probs = []
                for move in legal_moves:
                    if move.uci() in MOVE_TO_INDEX:
                        move_idx = MOVE_TO_INDEX[move.uci()]
                        move_probs.append((move, policy_probs[move_idx]))
                
                if not move_probs: 
                    # If no moves found (rare), return a random legal move
                    return np.random.choice(legal_moves) if legal_moves else None
                
                # Return the move with the highest probability
                return max(move_probs, key=lambda item: item[1])[0]