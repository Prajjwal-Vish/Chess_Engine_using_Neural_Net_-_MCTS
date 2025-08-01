
import os
import sys
import chess
import chess.pgn
import numpy as np
import time
import subprocess
import math

# --- Part 1: Configuration ---
STOCKFISH_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"
PGN_FILE_PATH = "lichess_data.pgn"
OUTPUT_DIR = "prepared_data_d12_multipv" # Saving to a new, specific directory
MAX_POSITIONS = 150000
SEARCH_DEPTH = 12

# --- Custom Stockfish Engine Controller ---
class StockfishEngine:
    """A robust class to handle direct communication with the Stockfish engine."""
    def __init__(self, path):
        self.engine = subprocess.Popen(
            path,
            universal_newlines=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._put("uci")
        self._read_until("uciok")
        # CRITICAL: Tell Stockfish to find the top 5 best moves
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
        """Analyzes a position and returns the top 5 moves and their scores."""
        fen = board.fen()
        self._put(f"position fen {fen}")
        self._put(f"go depth {depth}")
        lines = self._read_until("bestmove")
        
        results = []
        for line in lines:
            if "multipv" in line and "cp" in line:
                try:
                    score_cp = int(line.split("cp ")[1].split(" ")[0])
                    move_uci = line.split(" pv ")[1].split(" ")[0]
                    results.append({"score": score_cp, "pv": [move_uci]})
                except (IndexError, ValueError):
                    continue # Skip lines that are not parsable
        return results

    def quit(self):
        self._put("quit")
        self.engine.kill()

# --- Part 2: Main Logic ---
if not os.path.exists(STOCKFISH_PATH):
    print(f"❌ ERROR: Stockfish engine not found at '{STOCKFISH_PATH}'")
elif not os.path.exists(PGN_FILE_PATH):
    print(f"❌ ERROR: PGN file not found at '{PGN_FILE_PATH}'")
else:
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Helper functions
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

    # --- Main Loop using the robust engine ---
    engine = None
    try:
        print("Initializing new Stockfish engine controller...")
        engine = StockfishEngine(path=STOCKFISH_PATH)
        print("✅ Engine initialized and set to find top 5 moves.")
        
        inputs, targets, policies = [], [], []
        position_count = 0

        print(f"\n🚀 Starting data generation for {MAX_POSITIONS} positions...")
        start_time = time.time()
        with open(PGN_FILE_PATH, "r", encoding="utf-8") as pgn_file:
            while position_count < MAX_POSITIONS:
                game = chess.pgn.read_game(pgn_file)
                if game is None: 
                    print("Reached the end of the PGN file.")
                    break
                board = game.board()
                for move in game.mainline_moves():
                    if position_count >= MAX_POSITIONS: break
                    board.push(move)
                    
                    # This call will now return a list of up to 5 moves
                    info = engine.analyse(board, depth=SEARCH_DEPTH)
                    
                    if info:
                        inputs.append(board_to_input(board))
                        # The 'value' is still based on the single best move
                        targets.append(np.tanh(info[0]["score"] / 300.0))
                        
                        # --- NEW MULTI-MOVE POLICY LOGIC ---
                        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
                        move_scores = [p['score'] for p in info]
                        
                        # Use the softmax function to convert scores to probabilities
                        # Scaling by 100 is a common practice to prevent numbers from becoming too large or small
                        exp_scores = [math.exp(s / 100.0) for s in move_scores]
                        sum_exp_scores = sum(exp_scores)
                        probabilities = [s / sum_exp_scores for s in exp_scores]
                        
                        # Assign the calculated probabilities to the correct move indices
                        for i, p in enumerate(info):
                            move_idx = MOVE_TO_INDEX.get(p["pv"][0], -1)
                            if move_idx != -1:
                                policy[move_idx] = probabilities[i]
                        
                        policies.append(policy)
                        position_count += 1
                        
                        if position_count % 100 == 0: 
                            elapsed_time = time.time() - start_time
                            print(f"✅ Processed {position_count}/{MAX_POSITIONS} positions... (Total Time: {elapsed_time:.2f}s)")
                            
    except Exception as e:
        print(f"\n❌ A critical error occurred: {e}")
    finally:
        if engine:
            print("Closing engine...")
            engine.quit()

    if position_count > 0:
        np.save(os.path.join(OUTPUT_DIR, "inputs.npy"), np.array(inputs))
        np.save(os.path.join(OUTPUT_DIR, "targets.npy"), np.array(targets))
        np.save(os.path.join(OUTPUT_DIR, "policies.npy"), np.array(policies))
        print(f"\n✅ Data generation complete. Saved {position_count} positions to '{OUTPUT_DIR}'.")
