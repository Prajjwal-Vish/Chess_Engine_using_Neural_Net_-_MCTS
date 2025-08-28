# codes/core/mcts.py (Corrected for 24-layer model and UCI wrapper)

import chess
import torch
import numpy as np
import math

# Assumes this is in the same 'core' folder or the path is correctly set
from .utils import board_to_input

class Node:
    """A node in the Monte Carlo Tree Search tree."""
    def __init__(self, parent=None, prior_p=1.0):
        self.parent = parent
        self.children = {}  # A map from move to Node
        self.N = 0  # Visit count
        self.W = 0  # Total action value
        self.Q = 0  # Mean action value
        self.P = prior_p  # Prior probability of selecting this node

    def select(self, c_puct):
        """Select move among children with the highest UCB score."""
        return max(self.children.items(), key=lambda item: item[1].get_ucb_score(c_puct))

    def expand(self, board, policy_probs, move_map):
        """Expand the tree by creating new children nodes."""
        for move in board.legal_moves:
            if move not in self.children:
                move_uci = move.uci()
                if move_uci in move_map:
                    move_idx = move_map[move_uci]
                    self.children[move] = Node(parent=self, prior_p=policy_probs[move_idx])

    def update(self, value):
        """Update node with the result from a simulation."""
        self.N += 1
        self.W += value
        self.Q = self.W / self.N

    def get_ucb_score(self, c_puct):
        """Calculate the Upper Confidence Bound for Trees (UCT) score."""
        # Use a small epsilon to prevent division by zero if parent.N is 0
        parent_n = self.parent.N if self.parent else 1
        U = c_puct * self.P * math.sqrt(parent_n) / (1 + self.N)
        return self.Q + U

    def is_leaf(self):
        """Check if the node is a leaf node (has no children)."""
        return len(self.children) == 0

class MCTS:
    """The main class for the Monte Carlo Tree Search."""
    def __init__(self, model, move_map, c_puct=1.0):
        self.root = None
        self.model = model
        self.move_map = move_map # Store the move map
        self.c_puct = c_puct
        self.device = next(model.parameters()).device

    def _playout(self, board, history):
        """
        Run a single simulation from the root to a leaf, getting a value,
        and backpropagating it.
        """
        node = self.root
        # 1. Select
        while not node.is_leaf():
            move, node = node.select(self.c_puct)
            board.push(move)
            history.append(board.fen()) # Keep history in sync

        # 2. Evaluate & Expand
        with torch.no_grad():
            # The model now expects the history for repetition checks
            input_tensor = torch.tensor(
                board_to_input(board, history), dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            
            # Model returns raw logits for the policy
            value_tensor, policy_logits = self.model(input_tensor)
            value = value_tensor.item()
            
            # Apply softmax to get probabilities
            policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]

        if not board.is_game_over():
            node.expand(board, policy_probs, self.move_map)

        # 3. Backpropagate
        while node is not None:
            # The value is from the perspective of the current player at the node.
            # We must flip the sign for the parent.
            node.update(-value)
            node = node.parent
            value = -value

    def get_move(self, board, history, num_simulations, temperature=1.0):
        """
        Runs a number of simulations and returns the best move.
        """
        self.root = Node()
        for _ in range(num_simulations):
            board_copy = board.copy()
            history_copy = list(history) # Make a copy of the history
            self._playout(board_copy, history_copy)
        
        if not self.root.children:
            return None # Should not happen in a normal game

        if temperature == 0:
            # Deterministic: choose the move with the highest visit count
            return max(self.root.children.items(), key=lambda item: item[1].N)[0]
        else:
            # Probabilistic: sample from a distribution based on visit counts
            moves_visits = [(move, node.N) for move, node in self.root.children.items()]
            moves, visits = zip(*moves_visits)
            visit_counts = np.array(visits)
            move_probs = visit_counts**(1/temperature) / np.sum(visit_counts**(1/temperature))
            return np.random.choice(moves, p=move_probs)

    def get_move_analysis(self, board, history, num_simulations):
        """
        Runs simulations and returns the best move and a list of the top 5 moves for analysis.
        """
        self.root = Node()
        for _ in range(num_simulations):
            board_copy = board.copy()
            history_copy = list(history)
            self._playout(board_copy, history_copy)
        
        moves_visits = [(move, node.N) for move, node in self.root.children.items()]
        
        if not moves_visits:
            return None, []

        moves_visits.sort(key=lambda item: item[1], reverse=True)
        
        best_move = moves_visits[0][0]
        top_moves_analysis = moves_visits[:5]
        
        return best_move, top_moves_analysis
