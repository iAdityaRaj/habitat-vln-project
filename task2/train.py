"""
Training pipeline for CLIP-VLN model.
Uses behavior cloning (imitation learning) — same approach as VLN-CE.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
from model import VLNModel, ACTION_NAMES
import random
from tqdm import tqdm

# ── Synthetic dataset (replace with R2R in Task 3) ───────────────────────────
class SyntheticVLNDataset(Dataset):
    """
    Fake (rgb, instruction, action) triples for testing pipeline.
    Will be replaced with real R2R episodes in Task 3.
    """
    SAMPLES = [
        ("Go straight to the kitchen",              1),  # MOVE_FORWARD
        ("Turn left at the corridor",               2),  # TURN_LEFT
        ("Turn right near the elevator",            3),  # TURN_RIGHT
        ("Stop here at the reception desk",         0),  # STOP
        ("Move forward down the hallway",           1),  # MOVE_FORWARD
        ("Take a left at the end of the hall",      2),  # TURN_LEFT
        ("Go right past the door",                  3),  # TURN_RIGHT
        ("You have reached the destination, stop",  0),  # STOP
        ("Walk forward toward the living room",     1),  # MOVE_FORWARD
        ("Rotate left to face the window",          2),  # TURN_LEFT
    ]

    def __init__(self, size=500):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        instruction, action = random.choice(self.SAMPLES)
        # Random RGB image — replace with real habitat obs later
        rgb = torch.randint(0, 255, (3, 256, 256), dtype=torch.float32)
        return rgb, instruction, torch.tensor(action, dtype=torch.long)


# ── Training loop ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for rgb, instructions, actions in tqdm(loader, desc="Training"):
        rgb     = rgb.to(device)
        actions = actions.to(device)

        optimizer.zero_grad()

        # No hidden state between batches in behavior cloning
        logits, _ = model(rgb, list(instructions), hidden_state=None)
        loss = criterion(logits, actions)
        loss.backward()

        # Gradient clipping (same as VLN-CE)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        total_loss += loss.item()
        preds   = torch.argmax(logits, dim=1)
        correct += (preds == actions).sum().item()
        total   += actions.size(0)

    return total_loss / len(loader), correct / total


# ── Validation loop ───────────────────────────────────────────────────────────
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for rgb, instructions, actions in tqdm(loader, desc="Validating"):
            rgb     = rgb.to(device)
            actions = actions.to(device)

            logits, _ = model(rgb, list(instructions), hidden_state=None)
            loss = criterion(logits, actions)

            total_loss += loss.item()
            preds   = torch.argmax(logits, dim=1)
            correct += (preds == actions).sum().item()
            total   += actions.size(0)

    return total_loss / len(loader), correct / total


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Device — use M1 GPU if available
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device : {device}")

    # Dataset
    train_data   = SyntheticVLNDataset(size=400)
    val_data     = SyntheticVLNDataset(size=100)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=16, shuffle=False)

    # Model
    model     = VLNModel(feature_dim=512, num_actions=4).to(device)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-5
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print("\n" + "=" * 55)
    print("       TRAINING CLIP-VLN MODEL")
    print("=" * 55)

    NUM_EPOCHS   = 10
    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = val_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()

        print(f"Train → Loss: {train_loss:.4f} | Acc: {train_acc:.3f}")
        print(f"Val   → Loss: {val_loss:.4f} | Acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc': val_acc,
            }, "best_model.pth")
            print(f"✅ Best model saved! val_acc={val_acc:.3f}")

    print("\n" + "=" * 55)
    print(f"Training complete!")
    print(f"Best Validation Accuracy : {best_val_acc:.3f}")
    print(f"Model saved to           : best_model.pth")
