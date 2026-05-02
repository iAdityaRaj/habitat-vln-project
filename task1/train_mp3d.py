"""
Train CLIP-VLN on REAL MP3D RGB observations.
This is the proper training we've been working towards.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from model import VLNModel, ACTION_NAMES
from dataset_mp3d import MP3DDataset
from metrics import compute_metrics


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for rgb, instructions, actions in tqdm(loader, desc="  Train", leave=False):
        rgb     = rgb.to(device)
        actions = actions.to(device)
        optimizer.zero_grad()
        logits, _ = model(rgb, list(instructions), hidden_state=None)
        loss      = criterion(logits, actions)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        correct    += (torch.argmax(logits,1)==actions).sum().item()
        total      += actions.size(0)
    return total_loss/len(loader), correct/total


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss       = 0
    all_preds, all_t = [], []
    with torch.no_grad():
        for rgb, instructions, actions in tqdm(loader, desc="  Val", leave=False):
            rgb     = rgb.to(device)
            actions = actions.to(device)
            logits, _ = model(rgb, list(instructions), hidden_state=None)
            total_loss += criterion(logits, actions).item()
            all_preds.append(torch.argmax(logits,1).cpu())
            all_t.append(actions.cpu())
    all_preds = torch.cat(all_preds)
    all_t     = torch.cat(all_t)
    acc       = (all_preds==all_t).float().mean().item()
    m         = compute_metrics(all_preds, all_t)
    return total_loss/len(loader), acc, m


def plot_curves(history):
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    epochs = range(1, len(history['train_loss'])+1)

    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train')
    axes[0].plot(epochs, history['val_loss'],   'r-o', label='Val')
    axes[0].set_title('Loss over Epochs', fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train')
    axes[1].plot(epochs, history['val_acc'],   'r-o', label='Val')
    axes[1].set_title('Accuracy over Epochs', fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history['sr'],  'g-o', label='SR')
    axes[2].plot(epochs, history['spl'], 'm-o', label='SPL')
    axes[2].plot(epochs, history['ne'],  'c-o', label='NE (lower=better)')
    axes[2].set_title('SR / SPL / NE', fontweight='bold')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Score')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle('CLIP-VLN — Real MP3D Training (17DRP5sb8fy)', fontsize=13)
    plt.tight_layout()
    plt.savefig("learning_curves_mp3d.png", dpi=150, bbox_inches='tight')
    print("Saved: learning_curves_mp3d.png")
    plt.close()


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {device}")

    # Collect real RGB data
    print("\nBuilding real MP3D dataset...")
    full_ds  = MP3DDataset(num_samples=800)

    n_train  = int(0.8 * len(full_ds))
    n_val    = len(full_ds) - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=16,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=16,
                              shuffle=False, num_workers=0)

    print(f"Train: {n_train} | Val: {n_val}")

    model     = VLNModel(feature_dim=512, num_actions=4).to(device)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-5
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history  = {'train_loss':[], 'val_loss':[],
                'train_acc':[], 'val_acc':[],
                'sr':[], 'spl':[], 'ne':[]}
    best_sr  = 0.0
    NUM_EPOCHS = 20

    print("\n" + "="*55)
    print("   TRAINING ON REAL MP3D RGB — 17DRP5sb8fy")
    print("="*55)

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        tl, ta    = train_epoch(model, train_loader,
                                optimizer, criterion, device)
        vl, va, m = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        history['train_acc'].append(ta)
        history['val_acc'].append(va)
        history['sr'].append(m['sr'])
        history['spl'].append(m['spl'])
        history['ne'].append(m['ne'])

        print(f"  Train → Loss:{tl:.4f} | Acc:{ta:.3f}")
        print(f"  Val   → Loss:{vl:.4f} | Acc:{va:.3f}")
        print(f"  SR={m['sr']:.3f} | SPL={m['spl']:.3f} | NE={m['ne']:.3f}")
        print(f"  Per-action: ", end="")
        for k, v in m['action_acc'].items():
            print(f"{k}={v:.2f}", end=" ")
        print()

        if m['sr'] >= best_sr:
            best_sr = m['sr']
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'sr':          m['sr'],
                'spl':         m['spl'],
                'ne':          m['ne'],
                'scene':       '17DRP5sb8fy'
            }, "best_model_mp3d.pth")
            print(f"  ✅ Best model saved! SR={m['sr']:.3f}")

    plot_curves(history)

    with open("history_mp3d.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*55)
    print("        TRAINING COMPLETE")
    print("="*55)
    print(f"Best SR          : {best_sr:.4f}")
    print(f"Model saved      : best_model_mp3d.pth")
    print(f"Learning curves  : learning_curves_mp3d.png")
    print("\nNext: run evaluate.py with MODEL_PATH = 'best_model_mp3d.pth'")
