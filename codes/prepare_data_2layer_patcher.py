# codes/patch_material_features.py

import os
import numpy as np
import chess
from tqdm import tqdm
import shutil

# --- Configuration ---
PROJECT_PATH = "C:/Users/GFG0645/Desktop/chess_engine_project" 
# The folder with your existing 22-plane dataset
DATA_DIR = os.path.join(PROJECT_PATH, "prepared_data_final_move_hist") 

# --- Main Patching Logic ---
def main():
    print(f"Starting dataset patch process for directory: {DATA_DIR}")

    # Define file paths
    inputs_old_f = os.path.join(DATA_DIR, "inputs.npy")
    targets_old_f = os.path.join(DATA_DIR, "targets_patched.npy")
    policies_old_f = os.path.join(DATA_DIR, "policies.npy")

    # Define new file paths for the upgraded dataset
    output_dir_v2 = os.path.join(PROJECT_PATH, "prepared_data_2layer_patched")
    if not os.path.exists(output_dir_v2):
        os.makedirs(output_dir_v2)
        print(f"Created new directory for patched data: {output_dir_v2}")

    inputs_new_f = os.path.join(output_dir_v2, "inputs.npy")
    targets_new_f = os.path.join(output_dir_v2, "targets.npy")
    policies_new_f = os.path.join(output_dir_v2, "policies.npy")

    try:
        print("Loading existing 22-plane inputs...")
        inputs_old = np.load(inputs_old_f)
        num_positions = inputs_old.shape[0]
        print(f"✅ Loaded {num_positions} positions.")

        # Create a new, larger array for the 24-plane data
        inputs_new = np.zeros((num_positions, 24, 8, 8), dtype=np.float32)

        # Copy the existing 22 planes
        print("Copying existing feature planes...")
        inputs_new[:, :22, :, :] = inputs_old
        print("✅ Existing data copied.")

        print("Calculating and adding new material feature planes...")
        piece_values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}

        for i in tqdm(range(num_positions), desc="Patching positions"):
            position_data = inputs_old[i]
            
            white_material = 0
            black_material = 0

            # Calculate material from the first 12 planes
            for plane_idx in range(12):
                piece_type_char = 'pnbrqk'[plane_idx % 6]
                piece_value = piece_values[piece_type_char]
                
                # Sum the number of pieces on this plane
                num_pieces = np.sum(position_data[plane_idx])
                
                if plane_idx < 6: # White pieces
                    white_material += num_pieces * piece_value
                else: # Black pieces
                    black_material += num_pieces * piece_value
            
            # Get the current player's turn from plane 16
            is_white_turn = np.any(position_data[16])

            # Plane 22: Material difference from the current player's perspective
            material_diff = white_material - black_material
            if not is_white_turn:
                material_diff = -material_diff
            inputs_new[i, 22, :, :] = np.tanh(material_diff / 10.0)

            # Plane 23: Total material on the board
            total_material = white_material + black_material
            inputs_new[i, 23, :, :] = total_material / 78.0

        print("✅ New feature planes calculated.")

        # Save the new inputs file
        print(f"Saving new 24-plane inputs to {inputs_new_f}...")
        np.save(inputs_new_f, inputs_new)

        # Copy the targets and policies files to the new directory
        print("Copying targets and policies files...")
        shutil.copy(targets_old_f, targets_new_f)
        shutil.copy(policies_old_f, policies_new_f)
        
        print("\n" + "="*50)
        print("🎉 Dataset Patching Complete! 🎉")
        print(f"New dataset saved in: '{output_dir_v2}'")
        print("You can now use this new directory for training.")
        print("="*50)

    except FileNotFoundError:
        print(f"❌ ERROR: Could not find 'inputs.npy' in {DATA_DIR}. Please check the path.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
