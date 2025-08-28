# codes/prepare_data.py (Definitive Version with 24 Planes)

import os
import sys
import chess
import chess.pgn
import numpy as np
import time
import subprocess
import math

# --- Configuration ---
STOCKFISH_PATH = r"C:\Users\GFG0645\Desktop\chess_engine_project\stockfish\stockfish-windows-x86-64-avx2.exe"
PGN_FILE_PATH = r"C:\Users\GFG0645\Desktop\chess_engine_project\lichess_data.pgn"
OUTPUT_DIR = "prepared_data_24_layers" # Saving to a new, definitive directory
MAX_POSITIONS = 250000
SEARCH_DEPTH = 15
THREADS = 1

# --- Custom Stockfish Engine Controller ---
class StockfishEngine:
    def __init__(self, path, threads=1):
        self.engine = subprocess.Popen(
            path, universal_newlines=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self._put("uci")
        self._read_until("uciok")
        self._put(f"setoption name Threads value {threads}")
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
elif not os.path.exists(PGN_FILE_PATH): print(f"❌ ERROR: PGN file not found at '{PGN_FILE_PATH}'")
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

    # --- DEFINITIVE 24-PLANE INPUT REPRESENTATION ---
    def board_to_input(board, history):
        input_data = np.zeros((24, 8, 8), dtype=np.float32)
        
        piece_values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9}
        white_material, black_material = 0, 0

        # 12 planes for piece positions & calculate material
        for sq, p in board.piece_map().items():
            r, c = chess.square_rank(sq), chess.square_file(sq)
            p_idx = p.piece_type - 1 + (6 if p.color == chess.BLACK else 0)
            input_data[p_idx, r, c] = 1
            
            if p.color == chess.WHITE:
                white_material += piece_values.get(p.symbol().lower(), 0)
            else:
                black_material += piece_values.get(p.symbol().lower(), 0)
        
        # 4 planes for castling rights
        input_data[12, :, :] = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
        input_data[13, :, :] = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
        input_data[14, :, :] = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
        input_data[15, :, :] = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0
        
        # 1 plane for whose turn it is
        input_data[16, :, :] = 1 if board.turn == chess.WHITE else 0
        
        # 1 plane for the total move count
        input_data[17, :, :] = board.fullmove_number / 100.0 
        
        # 1 plane for the fifty-move rule counter
        input_data[18, :, :] = board.halfmove_clock / 100.0

        # 1 plane for the en passant square
        if board.ep_square:
            r, c = chess.square_rank(board.ep_square), chess.square_file(board.ep_square)
            input_data[19, r, c] = 1

        # 2 planes for repetition count
        current_fen = board.fen()
        rep_count = history.count(current_fen)
        if rep_count >= 1:
            input_data[20, :, :] = 1.0
        if rep_count >= 2:
            input_data[21, :, :] = 1.0
            
        # 2 planes for material advantage
        material_diff = white_material - black_material
        # Make the advantage relative to the current player
        if board.turn == chess.BLACK:
            material_diff = -material_diff
        input_data[22, :, :] = np.tanh(material_diff / 10.0) # Scaled between -1 and 1
        
        total_material = white_material + black_material
        input_data[23, :, :] = total_material / 78.0 # Scaled by total possible material
            
        return input_data

    engine = None
    try:
        print("Initializing Stockfish engine controller...")
        engine = StockfishEngine(path=STOCKFISH_PATH, threads=THREADS)
        print(f"✅ Engine initialized with {THREADS} threads.")
        inputs, targets, policies = [], [], []
        position_count = 0
        print(f"\n🚀 Starting data generation for {MAX_POSITIONS} positions...")
        start_time = time.time()
        with open(PGN_FILE_PATH, "r", encoding="utf-8") as pgn_file:
            while position_count < MAX_POSITIONS:
                game = chess.pgn.read_game(pgn_file)
                if game is None: print("Reached the end of the PGN file."); break
                
                board = game.board()
                game_history = [board.fen()]

                for move in game.mainline_moves():
                    if position_count >= MAX_POSITIONS: break
                    board.push(move)
                    
                    info = engine.analyse(board, depth=SEARCH_DEPTH)
                    if info:
                        inputs.append(board_to_input(board, game_history))
                        
                        top_move_score = info[0]["score"]
                        if abs(top_move_score) >= 9998:
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
                            print(f"✅ Processed {position_count}/{MAX_POSITIONS} positions... (Total Time: {elapsed_time:.2f}s)")
                    
                    game_history.append(board.fen())

    except Exception as e: print(f"\n❌ A critical error occurred: {e}")
    finally:
        if engine: print("Closing engine..."); engine.quit()
    if position_count > 0:
        np.save(os.path.join(OUTPUT_DIR, "inputs.npy"), np.array(inputs))
        np.save(os.path.join(OUTPUT_DIR, "targets.npy"), np.array(targets))
        np.save(os.path.join(OUTPUT_DIR, "policies.npy"), np.array(policies))
        print(f"\n✅ Data generation complete. Saved {position_count} positions to '{OUTPUT_DIR}'.")
