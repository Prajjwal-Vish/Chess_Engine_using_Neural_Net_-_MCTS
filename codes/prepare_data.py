# codes/prepare_data.py (Final Version with Corrected Broadcasting)

import os
import sys
import chess
import chess.pgn
import numpy as np
import time
import subprocess

# --- Part 1: Configuration ---
STOCKFISH_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"
PGN_FILE_PATH = "lichess_data.pgn"
OUTPUT_DIR = "prepared_data_d12"
MAX_POSITIONS = 150000
SEARCH_DEPTH = 12

# --- NEW: A more robust way to talk to Stockfish ---
class StockfishEngine:
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

    def _put(self, command):
        if self.engine.stdin:
            self.engine.stdin.write(f"{command}\n")
            self.engine.stdin.flush()

    def _read_until(self, wait_for):
        if self.engine.stdout:
            while True:
                line = self.engine.stdout.readline().strip()
                if wait_for in line:
                    return

    def analyse(self, board, depth):
        fen = board.fen()
        self._put(f"position fen {fen}")
        self._put(f"go depth {depth}")
        
        if self.engine.stdout:
            last_line = ""
            while True:
                line = self.engine.stdout.readline().strip()
                if "bestmove" in line:
                    try:
                        score_cp = int(last_line.split("cp ")[1].split(" ")[0])
                        best_move = line.split("bestmove ")[1].split(" ")[0]
                        return {"score": score_cp, "pv": [best_move]}
                    except (IndexError, ValueError):
                        return None
                last_line = line
        return None

    def quit(self):
        self._put("quit")
        self.engine.kill()

# --- Part 2: Main Logic ---
sys.path.append(os.path.dirname(os.getcwd()))

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
        """Converts a board state into a 25x8x8 numpy array for the neural network."""
        input_data = np.zeros((25, 8, 8), dtype=np.float32)
        for sq, p in board.piece_map().items():
            r, c = chess.square_rank(sq), chess.square_file(sq)
            p_idx = p.piece_type - 1 + (6 if p.color == chess.BLACK else 0)
            input_data[p_idx, r, c] = 1
        
        # --- THE FIX IS HERE ---
        # We explicitly set each of the four 8x8 planes.
        input_data[12, :, :] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
        input_data[13, :, :] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
        input_data[14, :, :] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
        input_data[15, :, :] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0
        # --- End of Fix ---

        input_data[16, :, :] = 1 if board.turn == chess.WHITE else 0
        return input_data

    # --- Main Loop using the new, robust engine ---
    engine = None
    try:
        print("Initializing new Stockfish engine controller...")
        engine = StockfishEngine(path=STOCKFISH_PATH)
        print("✅ Engine initialized.")
        
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
                    
                    info = engine.analyse(board, depth=SEARCH_DEPTH)
                    
                    if info and "score" in info:
                        inputs.append(board_to_input(board))
                        targets.append(np.tanh(info["score"] / 300.0))
                        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
                        best_move_uci = info["pv"][0]
                        move_idx = MOVE_TO_INDEX.get(best_move_uci, -1)
                        if move_idx != -1: policy[move_idx] = 1.0
                        policies.append(policy)
                        position_count += 1
                        
                        if position_count % 1000 == 0: 
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
        print(f"\n✅ Data generation complete. Saved {position_count} positions.")