# codes/prepare_puzzles.py

import os
import sys
import chess
import chess.pgn
import numpy as np
import time
import subprocess
import math
import pandas as pd

# --- Part 1: Configuration ---
STOCKFISH_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"
# This should be the path to the puzzle file you downloaded
PUZZLE_CSV_PATH = "data/lichess_db_puzzle.csv" 
OUTPUT_DIR = "prepared_puzzle_data"
MAX_POSITIONS_TO_GENERATE = 150000 # Target size for our new dataset
PUZZLE_RATING_RANGE = (1500, 2500) # Elo range for puzzles to include
SEARCH_DEPTH = 18 # Use a high depth for "perfect" ground truth data

# --- Custom Stockfish Engine Controller ---
# (This is the same robust class from our previous script)
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
        results = []
        for line in lines:
            if "multipv" in line:
                try:
                    move_uci = line.split(" pv ")[1].split(" ")[0]
                    if "cp" in line:
                        score = int(line.split("cp ")[1].split(" ")[0])
                    elif "mate" in line:
                        mate_in = int(line.split("mate ")[1].split(" ")[0])
                        score = 10000 * (1 if mate_in > 0 else -1)
                    else: continue
                    results.append({"score": score, "pv": [move_uci]})
                except (IndexError, ValueError): continue
        return results

    def quit(self):
        self._put("quit")
        self.engine.kill()

# --- Main Logic ---
if not os.path.exists(STOCKFISH_PATH): print(f"❌ ERROR: Stockfish engine not found at '{STOCKFISH_PATH}'")
elif not os.path.exists(PUZZLE_CSV_PATH): print(f"❌ ERROR: Puzzle CSV file not found at '{PUZZLE_CSV_PATH}'")
else:
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
        
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
        return {move_string: i for i, move_string in enumerate(unique_uci_moves)}

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

    engine = None
    try:
        print("Loading and filtering puzzle database... (This may take a minute)")
        # Use pandas to efficiently read and filter the large CSV
        df = pd.read_csv(PUZZLE_CSV_PATH)
        
        # Filter for puzzles with the desired themes and rating
        themes = ['mate', 'crushing', 'advantage', 'crushing', 'endgame', 'hangingPiece', 'long', 'quietMove']
        filtered_df = df[
            df['Themes'].str.contains('|'.join(themes)) &
            df['Rating'].between(PUZZLE_RATING_RANGE[0], PUZZLE_RATING_RANGE[1])
        ].sample(frac=1) # Shuffle the puzzles
        
        print(f"✅ Found {len(filtered_df)} high-quality puzzles. Starting generation...")

        print("Initializing Stockfish engine controller...")
        engine = StockfishEngine(path=STOCKFISH_PATH)
        print("✅ Engine initialized.")
        
        inputs, targets, policies = [], [], []
        position_count = 0
        start_time = time.time()

        # Iterate through the filtered puzzles
        for index, row in filtered_df.iterrows():
            if position_count >= MAX_POSITIONS_TO_GENERATE: break
            
            fen = row['FEN']           
            board = chess.Board(fen)
            
            # Process each move in the puzzle's solution
            for move_uci in solution_moves_uci:
                if position_count >= MAX_POSITIONS_TO_GENERATE: break
                
                # Analyze the position *before* the correct move is made
                info = engine.analyse(board, depth=SEARCH_DEPTH)
                
                if info:
                    inputs.append(board_to_input(board))
                    
                    top_move_score = info[0]["score"]
                    if abs(top_move_score) >= 9999: # Mate score
                        targets.append(1.0 if top_move_score > 0 else -1.0)
                    else:
                        targets.append(np.tanh(top_move_score / 300.0))
                    
                    policy = np.zeros(POLICY_SIZE, dtype=np.float32)
                    move_scores = [p['score'] for p in info]
                    exp_scores = [math.exp(s / 100.0) for s in move_scores]
                    sum_exp_scores = sum(exp_scores)
                    probabilities = [s / sum_exp_scores for s in exp_scores]
                    
                    for i, p in enumerate(info):
                        move_idx = MOVE_TO_INDEX.get(p["pv"][0], -1)
                        if move_idx != -1:
                            policy[move_idx] = probabilities[i]
                    policies.append(policy)
                    position_count += 1
                    
                    if position_count % 100 == 0: 
                        elapsed_time = time.time() - start_time
                        print(f"✅ Processed {position_count}/{MAX_POSITIONS_TO_GENERATE} positions... (Total Time: {elapsed_time:.2f}s)")
                
                # Make the correct puzzle move to get to the next position in the solution
                try:
                    board.push_uci(move_uci)
                except ValueError:
                    # Break if a move in the puzzle data is illegal for some reason
                    break
                        
    except Exception as e: print(f"\n❌ A critical error occurred: {e}")
    finally:
        if engine: print("Closing engine..."); engine.quit()

    if position_count > 0:
        np.save(os.path.join(OUTPUT_DIR, "inputs.npy"), np.array(inputs))
        np.save(os.path.join(OUTPUT_DIR, "targets.npy"), np.array(targets))
        np.save(os.path.join(OUTPUT_DIR, "policies.npy"), np.array(policies))
        print(f"\n✅ Puzzle data generation complete. Saved {position_count} positions to '{OUTPUT_DIR}'.")










































































































































































































































































































































































































































































































































































































































































































































































































































