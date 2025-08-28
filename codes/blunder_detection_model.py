# --- Blunder Detection Model Training Script ---
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score

# --- Part 1: Define the Blunder Detection Network ---
# This is a simpler version of our ResNet, as it only has one job:
# to output a single number (the probability of a blunder).

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

class BlunderNet(nn.Module):
    def __init__(self, num_residual_blocks=6):
        super(BlunderNet, self).__init__()
        self.initial_conv = nn.Conv2d(25, 128, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(128)
        self.residual_blocks = nn.ModuleList([ResidualBlock(128) for _ in range(num_residual_blocks)])
        
        # A single, simple head for binary classification
        self.classifier_fc1 = nn.Linear(128 * 8 * 8, 512)
        self.classifier_fc2 = nn.Linear(512, 1) # Single output neuron

    def forward(self, x):
        out = F.relu(self.initial_bn(self.initial_conv(x)))
        for block in self.residual_blocks:
            out = block(out)
        
        # Flatten the output from the convolutional body
        out = out.view(-1, 128 * 8 * 8)
        
        out = F.relu(self.classifier_fc1(out))
        out = self.classifier_fc2(out) # No sigmoid here, as BCEWithLogitsLoss is more stable
        return out

# --- Part 2: Setup and Data Loading ---
drive.mount('/content/drive')
DRIVE_PROJECT_PATH = '/content/drive/MyDrive/chess'
# Path to the folder containing your inputs.npy and the new blunder_labels.npy
DATA_DIR_DRIVE = os.path.join(DRIVE_PROJECT_PATH, 'prepared_puzzle_data') 
MODEL_SAVE_PATH = os.path.join(DRIVE_PROJECT_PATH, 'blunder_detector_model.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
print("Copying dataset from Google Drive to local storage...")
!cp -r "{DATA_DIR_DRIVE}" /content/prepared_data
DATA_DIR_LOCAL = "/content/prepared_data"
print("✅ Dataset copied locally.")

try:
    inputs = np.load(os.path.join(DATA_DIR_LOCAL, "inputs.npy"))
    # Load the new blunder labels
    blunder_labels = np.load(os.path.join(DATA_DIR_LOCAL, "blunder_labels.npy"))
    
    # --- Data Cleaning: Remove skipped positions ---
    # Your script correctly marks skipped/invalid positions as -1.
    # We must remove these so the model doesn't train on them.
    valid_indices = np.where(blunder_labels != -1)[0]
    inputs = inputs[valid_indices]
    blunder_labels = blunder_labels[valid_indices]
    
    print(f"✅ Dataset loaded and cleaned. Found {len(inputs)} valid positions.")
except FileNotFoundError:
    print(f"❌ ERROR: inputs.npy or blunder_labels.npy not found in {DATA_DIR_DRIVE}.")
    raise

# --- Part 3: Create DataLoaders ---
inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
labels_tensor = torch.tensor(blunder_labels, dtype=torch.float32).view(-1, 1)
dataset = TensorDataset(inputs_tensor, labels_tensor)

train_size = int(0.85 * len(dataset)); val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(f"Training set: {len(train_dataset)} | Validation set: {len(val_dataset)}")
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

# --- Part 4: Initialize Model and Training Loop ---
model = BlunderNet().to(device)
# This loss function is the standard for binary classification and is numerically stable
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

num_epochs = 40
best_val_loss = float('inf')
patience = 7
epochs_no_improve = 0

print("\n🚀 Starting blunder detector model training...")
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for batch_inputs, batch_labels in train_loader:
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    # --- Validation with Detailed Metrics ---
    model.eval()
    total_val_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_inputs, batch_labels in val_loader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            total_val_loss += loss.item()
            
            # Convert logits to probabilities (0 or 1) for metrics
            preds = torch.round(torch.sigmoid(outputs))
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    
    # Calculate metrics using scikit-learn
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
          f"Accuracy: {accuracy:.2%} | Precision: {precision:.2%} | Recall: {recall:.2%}")
    
    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"🎉 Validation loss improved! Model saved to {MODEL_SAVE_PATH}")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")
    
    if epochs_no_improve >= patience:
        print(f"\nStopping early. Validation loss has not improved for {patience} epochs.")
        break

print("\n✅ --- Blunder Detector Training Complete --- ✅")
