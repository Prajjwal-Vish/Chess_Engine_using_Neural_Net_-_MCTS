# codes/mcts.py

import chess
import torch
import numpy as np
import math
import os
import sys

# We need to include the ResNet definition here so we can load the model
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class ChessResNet(nn.Module):
    def __init__(self, num_residual_blocks, policy_size):
        super(ChessResNet, self).__init__()
        self.initial_conv = nn.Conv2d(25, 128, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(128)
        self.residual_blocks = nn.ModuleList([ResidualBlock(128) for _ in range(num_residual_blocks)])
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc = nn.Linear(8*8, 1)
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, policy_size)
    def forward(self, x):
        out = F.relu(self.initial_bn(self.initial_conv(x)))
        for block in self.residual_blocks: out = block(out)
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(-1, 8*8)
        value = torch.tanh(self.value_fc(value))
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(-1, 2 * 8 * 8)
        policy = F.log_softmax(self.policy_fc(policy), dim=1)
        return value, policy

# --- Helper Functions ---
def create_move_map():
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
    return {move: i for i, move in enumerate(unique_uci_moves)}

MOVE_TO_INDEX = create_move_map()
POLICY_SIZE = len(MOVE_TO_INDEX)

def board_to_input(board):
    input_data = np.zeros((25, 8, 8), dtype=np.float32)
    for sq, p in board.piece_map().items():
        r, c = chess.square_rank(sq), chess.square_file(sq)
        p_idx = p.piece_type - 1 + (6 if p.color == chess.BLACK else 0)
        input_data[p_idx, r, c] = 1
    input_data[12, :, :] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
    input_data[13, :, :] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
    input_data[14, :, :] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
    input_data[15, :, :] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0
    input_data[16, :, :] = 1 if board.turn == chess.WHITE else 0
    return input_data

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

# --- The MCTS Implementation ---

class Node:
    def __init__(self, parent=None, prior_p=1.0):
        self.parent = parent
        self.children = {}
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = prior_p

    def select(self, c_puct):
        return max(self.children.items(), key=lambda item: item[1].get_ucb_score(c_puct))

    def expand(self, board, policy_probs):
        for move in board.legal_moves:
            if move not in self.children:
                move_uci = move.uci()
                if move_uci in MOVE_TO_INDEX:
                    move_idx = MOVE_TO_INDEX[move_uci]
                    self.children[move] = Node(parent=self, prior_p=policy_probs[move_idx])

    def update(self, value):
        self.N += 1
        self.W += value
        self.Q = self.W / self.N

    def get_ucb_score(self, c_puct):
        U = c_puct * self.P * math.sqrt(self.parent.N) / (1 + self.N)
        return self.Q + U

    def is_leaf(self):
        return len(self.children) == 0

class MCTS:
    def __init__(self, model, c_puct=1.0):
        self.root = Node()
        self.model = model
        self.c_puct = c_puct
        self.device = next(model.parameters()).device

    def _playout(self, board):
        node = self.root
        while not node.is_leaf():
            move, node = node.select(self.c_puct)
            board.push(move)

        with torch.no_grad():
            input_tensor = torch.tensor(board_to_input(board), dtype=torch.float32).unsqueeze(0).to(self.device)
            value_tensor, policy_log_softmax = self.model(input_tensor)
            value = value_tensor.item()
            policy_probs = torch.exp(policy_log_softmax).cpu().numpy()[0]

        if not board.is_game_over():
            node.expand(board, policy_probs)

        while node is not None:
            node.update(-value)
            node = node.parent
            value = -value

    def get_best_move(self, board, num_simulations):
        print(f"Thinking for {num_simulations} simulations...")
        for n in range(num_simulations):
            board_copy = board.copy()
            self._playout(board_copy)
        return max(self.root.children.items(), key=lambda item: item[1].N)[0]

# --- Example of how to use the MCTS on the first 20 positions of the dataset ---
if __name__ == '__main__':
    # Define paths
    PROJECT_PATH = "C:/Users/GFG0645/Desktop/chess_engine_project" 
    MODEL_PATH = os.path.join(PROJECT_PATH, 'models/chess_resnet_model.pth')
    DATA_DIR = os.path.join(PROJECT_PATH, 'prepared_data_d12_multipv')
    
    # Load the trained model
    print("Loading trained ResNet model...")
    device = torch.device("cpu")
    model = ChessResNet(num_residual_blocks=8, policy_size=POLICY_SIZE).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Model loaded successfully.")

    # Load the dataset inputs to test on
    inputs = np.load(os.path.join(DATA_DIR, "inputs.npy"))
    print(f"✅ Dataset inputs loaded. Total positions: {len(inputs)}")

    # --- Main Testing Loop ---
    POSITIONS_TO_TEST = 20
    SIMULATIONS_PER_MOVE = 2000 # A small number for a quick test

    for i in range(POSITIONS_TO_TEST):
        print("\n" + "="*60)
        print(f"          TESTING POSITION #{i}")
        print("="*60)
        
        # Reconstruct the board from the dataset
        board = board_from_input(inputs[i])
        print("Board to analyze:")
        print(board)
        
        # Create a FRESH MCTS for each position
        mcts = MCTS(model)
        
        # Run the search to find the best move
        best_move = mcts.get_best_move(board, num_simulations=SIMULATIONS_PER_MOVE)

        print("\n--- MCTS Result ---")
        print(f"Best move found: {best_move.uci()}")
        print("="*60)
