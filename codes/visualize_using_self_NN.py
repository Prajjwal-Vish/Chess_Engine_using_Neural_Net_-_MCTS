import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess

# --- Part 1: Define the ResNet Architecture ---
# We include the model's structure here so the script is self-contained
# and can load the saved weights correctly.

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
        policy = policy.view(-1, 2 * 8 * 8)
        policy = F.log_softmax(self.policy_fc(policy), dim=1)
        return value, policy

# --- Part 2: Setup Paths and Load Data/Model ---
# Use your local project path
PROJECT_PATH = "C:/Users/GFG0645/Desktop/chess_engine_project" 
DATA_DIR = os.path.join(PROJECT_PATH, 'prepared_data_d12_multipv')
MODEL_PATH = os.path.join(PROJECT_PATH, 'models/chess_resnet_model.pth')
device = torch.device("cpu") # We'll use the CPU for local testing

print(f"Using device: {device}")

# Load the dataset files
try:
    inputs = np.load(os.path.join(DATA_DIR, "inputs.npy"))
    targets = np.load(os.path.join(DATA_DIR, "targets.npy"))
    policies = np.load(os.path.join(DATA_DIR, "policies.npy"))
    POLICY_SIZE = policies.shape[1]

    # Load the trained model
    model = ChessResNet(num_residual_blocks=8, policy_size=POLICY_SIZE).to(device)
    # map_location=torch.device('cpu') is important to load a GPU-trained model onto a CPU
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Set the model to evaluation mode
    print("✅ Trained model and dataset loaded successfully.")

    # --- Part 3: Helper Functions for Interpretation ---
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
        return {i: move for i, move in enumerate(unique_uci_moves)}, {move: i for i, move in enumerate(unique_uci_moves)}

    INDEX_TO_MOVE, MOVE_TO_INDEX = create_move_map()

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

    # --- Part 4: Main Testing Loop ---
    POSITIONS_TO_TEST = 20

    for i in range(POSITIONS_TO_TEST):
        input_data = inputs[i]
        target_value = targets[i]
        target_policy = policies[i]
        board = board_from_input(input_data)
        
        with torch.no_grad():
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
            pred_value_tensor, pred_policy_log_softmax = model(input_tensor)
            predicted_value = pred_value_tensor.item()
            predicted_policy_probs = torch.exp(pred_policy_log_softmax).cpu().numpy()[0]

        print("\n" + "="*60)
        print(f"          VIEWING POSITION #{i}")
        print("="*60)
        print(board)
        print(f"\n--- Turn: {'White' if board.turn == chess.WHITE else 'Black'} ---")

        print("\n--- Ground Truth (from Dataset) ---")
        print(f"Value: {target_value:.4f}")
        gt_move_indices = np.where(target_policy > 0)[0]
        gt_move_probs = sorted([(INDEX_TO_MOVE[idx], target_policy[idx]) for idx in gt_move_indices], key=lambda item: item[1], reverse=True)
        print("Policy (Top 5 Moves):")
        for move, prob in gt_move_probs:
            print(f"  - {move}: {prob:.2%}")

        print("\n--- Model's Prediction ---")
        print(f"Predicted Value: {predicted_value:.4f}")
        legal_moves = list(board.legal_moves)
        pred_move_probs = []
        for move in legal_moves:
            if move.uci() in MOVE_TO_INDEX:
                move_idx = MOVE_TO_INDEX[move.uci()]
                pred_move_probs.append((move.uci(), predicted_policy_probs[move_idx]))
        pred_move_probs.sort(key=lambda item: item[1], reverse=True)
        print("Predicted Policy (Top 5 Legal Moves):")
        for move, prob in pred_move_probs[:5]:
            print(f"  - {move}: {prob:.2%}")
        print("="*60)

except FileNotFoundError:
    print(f"❌ ERROR: Could not find the model or dataset files. Please check your paths.")
    print(f"  - Looking for model at: {MODEL_PATH}")
    print(f"  - Looking for data in: {DATA_DIR}")
