# codes/prepare_blunder_dataset.py

import os
import sys
import time
import csv
import numpy as np
import chess
import subprocess
from tqdm import tqdm

# ----------------- CONFIG -----------------
PROJECT_PATH = "C:/Users/GFG0645/Desktop/chess_engine_project"
# Use the data from your original model's training
DATA_DIR = os.path.join(PROJECT_PATH, "prepared_puzzle_data") 
INPUTS_F = os.path.join(DATA_DIR, "inputs.npy")
POLICIES_F = os.path.join(DATA_DIR, "policies.npy")

# Output files
OUT_BLUNDER_NPY = os.path.join(DATA_DIR, "blunder_labels.npy")
OUT_INFO_CSV = os.path.join(DATA_DIR, "blunder_info.csv")

STOCKFISH_PATH = os.path.join(PROJECT_PATH, "stockfish/stockfish-windows-x86-64-avx2.exe")
DEPTH = 14  # A deep search is needed to reliably identify blunders
BLUNDER_THRESHOLD_CP = 150 # A 1.5 pawn advantage loss is a clear blunder
# ------------------------------------------

# --- Custom Stockfish Engine Controller ---
# This robust class uses direct communication to avoid hangs.
class StockfishEngine:
    def __init__(self, path):
        self.engine = subprocess.Popen(
            path, universal_newlines=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self._put("uci")
        self._read_until("uciok")
        self._put("setoption name MultiPV value 5")

    def _put(self, command):
        if self.engine.stdin:
            self.engine.stdin.write(f"{command}\n")
            self.engine.stdin.flush()

    def _read_until(self, wait_for):
        if self.engine.stdout:
            lines = []
            while True:
                line = self.engine.stdout.readline().strip()
                lines.append(line)
                if wait_for in line:
                    return lines
    
    def analyse(self, board, depth):
        fen = board.fen()
        self._put(f"position fen {fen}")
        self._put(f"go depth {depth}")
        lines = self._read_until("bestmove")
        results = {}
        for line in lines:
            if "multipv" in line and "cp" in line:
                try:
                    score_cp = int(line.split("cp ")[1].split(" ")[0])
                    move_uci = line.split(" pv ")[1].split(" ")[0]
                    results[move_uci] = score_cp
                except (IndexError, ValueError): continue
        return results

    def quit(self):
        self._put("quit")
        self.engine.kill()

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
    return {i: uci for i, uci in enumerate(unique_uci_moves)}

INDEX_TO_MOVE = create_move_map()

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

# --- Main Processing Logic ---
def main():
    print("Loading inputs and policies...")
    inputs = np.load(INPUTS_F)
    policies = np.load(POLICIES_F)
    n = inputs.shape[0]
    print(f"Loaded {n} positions.")

    blunder_labels = np.full((n,), -1, dtype=np.int8)
    info_rows = []
    engine = None

    try:
        engine = StockfishEngine(STOCKFISH_PATH)
        print("✅ Stockfish engine initialized.")

        for i in tqdm(range(n), desc="Labeling Blunders"):
            board = board_from_input(inputs[i])
            
            # Get the model's predicted move
            top_policy_idx = np.argmax(policies[i])
            predicted_move_uci = INDEX_TO_MOVE.get(top_policy_idx)

            if not predicted_move_uci:
                continue # Skip if move index is invalid

            # Get a dictionary of {move: score} from Stockfish
            move_scores = engine.analyse(board, depth=DEPTH)
            
            if not move_scores:
                continue # Skip if engine fails

            # Find the actual best move and its score
            best_move_uci = max(move_scores, key=move_scores.get)
            best_move_score = move_scores[best_move_uci]

            # Get the score for the model's predicted move
            predicted_move_score = move_scores.get(predicted_move_uci)

            # If the predicted move is illegal, Stockfish won't return a score for it.
            # This is a clear blunder.
            if predicted_move_score is None:
                is_blunder = 1
                cp_loss = 9999 # Assign a large penalty
            else:
                # Calculate centipawn loss from the current player's perspective
                cp_loss = best_move_score - predicted_move_score
                if board.turn == chess.BLACK:
                    cp_loss = -cp_loss # Invert for black

                is_blunder = 1 if cp_loss >= BLUNDER_THRESHOLD_CP else 0

            blunder_labels[i] = is_blunder
            info_rows.append((i, board.fen(), predicted_move_uci, best_move_uci, cp_loss, is_blunder))

    except Exception as e:
        print(f"\n❌ A critical error occurred: {e}")
    finally:
        if engine:
            print("Closing engine...")
            engine.quit()

    # Save the final outputs
    np.save(OUT_BLUNDER_NPY, blunder_labels)
    with open(OUT_INFO_CSV, "w", newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(["index", "fen", "predicted_move", "best_move", "cp_loss", "is_blunder"])
        writer.writerows(info_rows)
    
    print(f"\n✅ Blunder labeling complete. Saved labels to '{OUT_BLUNDER_NPY}'")

if __name__ == "__main__":
    main()
