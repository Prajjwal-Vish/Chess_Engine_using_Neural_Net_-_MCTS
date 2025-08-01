# codes/engine_match.py (Final Corrected Version)

import os
import sys
import chess
import chess.pgn
import torch
import numpy as np
import math
import time

# --- Part 1: Re-define the necessary classes and functions ---
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
        for block in self.residual_blocks:
            out = block(out)
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(-1, 8*8)
        value = torch.tanh(self.value_fc(value))
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        # Corrected a typo here from --1 to -1
        policy = policy.view(-1, 2 * 8 * 8)
        policy = F.log_softmax(self.policy_fc(policy), dim=1)
        return value, policy

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

    def get_move(self, board, num_simulations, temperature=1.0):
        self.root = Node()
        for _ in range(num_simulations):
            board_copy = board.copy()
            self._playout(board_copy)
        moves_visits = [(move, node.N) for move, node in self.root.children.items()]
        if not moves_visits:
            return None
        moves, visit_counts = zip(*moves_visits)
        if temperature == 0:
            best_move_idx = np.argmax(visit_counts)
            return moves[best_move_idx]
        visit_probs = np.array(visit_counts)**(1/temperature)
        visit_probs /= np.sum(visit_probs)
        return np.random.choice(moves, p=visit_probs)

# --- Part 2: Define the Two Player Personalities ---
def get_resnet_only_move(board, model, device):
    with torch.no_grad():
        input_tensor = torch.tensor(board_to_input(board), dtype=torch.float32).unsqueeze(0).to(device)
        _, policy_log_softmax = model(input_tensor)
        policy_probs = torch.exp(policy_log_softmax).cpu().numpy()[0]
    legal_moves = list(board.legal_moves)
    move_probs = []
    for move in legal_moves:
        if move.uci() in MOVE_TO_INDEX:
            move_idx = MOVE_TO_INDEX[move.uci()]
            move_probs.append((move, policy_probs[move_idx]))
    if not move_probs: return None
    return max(move_probs, key=lambda item: item[1])[0]

def get_mcts_move(mcts_instance, board, num_simulations, move_count):
    temperature = 1.0 if move_count < 15 else 0.0
    return mcts_instance.get_move(board, num_simulations, temperature)

# --- Part 3: The Game Loop (Corrected) ---
def play_game(white_player_fn, black_player_fn, model, device, mcts_sims):
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Engine Match"
    game.headers["White"] = "MCTS Player" if white_player_fn == get_mcts_move else "ResNet Only Player"
    game.headers["Black"] = "MCTS Player" if black_player_fn == get_mcts_move else "ResNet Only Player"
    node = game
    mcts_instance = MCTS(model)
    full_move_count = 1
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = white_player_fn(mcts_instance, board, mcts_sims, full_move_count) if white_player_fn == get_mcts_move else white_player_fn(board, model, device)
        else:
            move = black_player_fn(mcts_instance, board, mcts_sims, full_move_count) if black_player_fn == get_mcts_move else black_player_fn(board, model, device)
        if move is None or move not in board.legal_moves:
            print("!!! Invalid move returned by engine. Ending game. !!!")
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            break
        board.push(move)
        node = node.add_variation(move)
        if board.turn == chess.WHITE:
            full_move_count += 1
    game.headers["Result"] = board.result()
    return str(game), board.result()

# --- Part 4: Main Execution ---
if __name__ == '__main__':
    PROJECT_PATH = "C:/Users/GFG0645/Desktop/chess_engine_project" 
    MODEL_PATH = os.path.join(PROJECT_PATH, 'models/chess_resnet_model.pth')
    
    print("Loading trained ResNet model...")
    device = torch.device("cpu")
    model = ChessResNet(num_residual_blocks=8, policy_size=POLICY_SIZE).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Model loaded successfully.")

    NUM_GAMES = 1
    SIMS_PER_MOVE = 1000
    
    mcts_wins, resnet_wins, draws = 0, 0, 0
    pgn_outputs = []

    for i in range(NUM_GAMES):
        print(f"\nPlaying Game {i+1}/{NUM_GAMES}...")
        
        if i % 2 == 0:
            pgn_string, result = play_game(get_mcts_move, get_resnet_only_move, model, device, mcts_sims=SIMS_PER_MOVE)
            if result == "1-0": mcts_wins += 1
            elif result == "0-1": resnet_wins += 1
            else: draws += 1
        else:
            pgn_string, result = play_game(get_resnet_only_move, get_mcts_move, model, device, mcts_sims=SIMS_PER_MOVE)
            if result == "1-0": resnet_wins += 1
            elif result == "0-1": mcts_wins += 1
            else: draws += 1
        
        pgn_outputs.append(pgn_string)
        print(f"Game {i+1} finished.")
            
    print("\n\n" + "="*50); print("             FINAL MATCH SCORE"); print("="*50)
    print(f"MCTS Player Wins: {mcts_wins}"); print(f"ResNet Only Player Wins: {resnet_wins}"); print(f"Draws: {draws}"); print("="*50)

    print("\n\n" + "="*50); print("               GAME PGNs"); print("="*50)
    for i, pgn in enumerate(pgn_outputs):
        print(f"\n--- PGN for Game {i+1} ---"); print(pgn)