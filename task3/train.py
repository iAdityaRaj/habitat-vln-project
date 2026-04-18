import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os
from tqdm import tqdm

from model import VLNModel, ACTION_NAMES
from dataset import R2RDataset
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
        preds   = torch.argmax(logits, dim=1)
        correct += (preds == actions).sum().item()
        total   += actions.size(0)
    return total_loss / len(loader), correct / total


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss       = 0
    all_preds, all_t = [], []
    with torch.no_grad():
        for rgb, instructions, actions in tqdm(loader, desc="  Val", leave=False):
            rgb     = rgb.to(device)
            actions = actions.to(device)
            logits, _ = model(rgb, list(instructions), hidden_state=None)
            loss      = criterion(logits, actions)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_t.append(actions.cpu())

    all_preds = torch.cat(all_preds)
    all_t     = torch.cat(all_t)
    acc       = (all_preds == all_t).float().mean().item()
    m         = compute_metrics(all_preds, all_t)
    return total_loss / len(loader), acc, m


def plot_curves(history, name):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train')
    axes[0].plot(epochs, history['val_loss'],   'r-o', label='Val')
    axes[0].set_title('Loss over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train Acc')
    axes[1].plot(epochs, history['val_acc'],   'r-o', label='Val Acc')
    axes[1].set_title('Accuracy over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history['sr'],  'g-o', label='SR')
    axes[2].plot(epochs, history['spl'], 'm-o', label='SPL')
    axes[2].plot(epochs, history['ne'],  'c-o', label='NE (lower=better)')
    axes[2].set_title('SR / SPL / NE over Epochs')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Score')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'CLIP-VLN Training — {name}', fontsize=13)
    plt.tight_layout()
    fname = f"learning_curves_{name}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved: {fname}")
    plt.close()


CONFIGS = [
    {'lr': 1e-4, 'batch_size': 16, 'name': 'lr1e-4_bs16'},
    {'lr': 5e-4, 'batch_size': 16, 'name': 'lr5e-4_bs16'},
    {'lr': 1e-4, 'batch_size': 32, 'name': 'lr1e-4_bs32'},
]


def train_config(config, train_path, val_path, device, num_epochs=10):
    print(f"\n{'='*55}")
    print(f"Config: lr={config['lr']} | batch_size={config['batch_size']}")
    print(f"{'='*55}")

    train_ds = R2RDataset(train_path, max_episodes=500)
    val_ds   = R2RDataset(val_path,   max_episodes=100)
    tr_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    vl_loader = DataLoader(val_ds,   batch_size=config['batch_size'], shuffle=False)

    model     = VLNModel(feature_dim=512, num_actions=4).to(device)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'], weight_decay=1e-5
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history  = {'train_loss':[], 'val_loss':[],
                'train_acc':[], 'val_acc':[],
                'sr':[], 'spl':[], 'ne':[]}
    best_sr  = 0.0

    for epoch in range(num_epochs):
        print(f"\n  Epoch {epoch+1}/{num_epochs}")
        tr_loss, tr_acc   = train_epoch(model, tr_loader, optimizer, criterion, device)
        vl_loss, vl_acc, m = val_epoch(model, vl_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)
        history['sr'].append(m['sr'])
        history['spl'].append(m['spl'])
        history['ne'].append(m['ne'])

        print(f"  Train → Loss: {tr_loss:.4f} | Acc: {tr_acc:.3f}")
        print(f"  Val   → Loss: {vl_loss:.4f} | Acc: {vl_acc:.3f}")
        print(f"  SR={m['sr']:.3f} | SPL={m['spl']:.3f} | NE={m['ne']:.3f}")
        print(f"  Per-action: ", end="")
        for k, v in m['action_acc'].items():
            print(f"{k}={v:.2f}", end=" ")
        print()

        if m['sr'] >= best_sr:
            best_sr = m['sr']
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'config': config, 'metrics': m},
                       f"best_model_{config['name']}.pth")
            print(f"  ✅ Best model saved!")

    plot_curves(history, config['name'])
    with open(f"history_{config['name']}.json", 'w') as f:
        json.dump(history, f, indent=2)
    return history, best_sr


if __name__ == "__main__":
    device = torch.device("cpu")
    print(f"Device: {device}")

    train_path = "data/R2R_train.json"
    val_path   = "data/R2R_val_seen.json"

    if not os.path.exists(train_path):
        print("❌ R2R data not found!")
        exit()

    results = {}
    for config in CONFIGS:
        history, best_sr = train_config(
            config, train_path, val_path, device, num_epochs=10
        )
        results[config['name']] = {
            'best_sr':       best_sr,
            'best_spl':      max(history['spl']),
            'best_ne':       min(history['ne']),
            'final_val_acc': history['val_acc'][-1]
        }

    print("\n" + "=" * 60)
    print("        HYPERPARAMETER TUNING RESULTS")
    print("=" * 60)
    print(f"{'Config':<20} {'SR':>8} {'SPL':>8} {'NE':>8} {'Val Acc':>10}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<20} {r['best_sr']:>8.3f} {r['best_spl']:>8.3f} "
              f"{r['best_ne']:>8.3f} {r['final_val_acc']:>10.3f}")

    best = max(results, key=lambda x: results[x]['best_sr'])
    print(f"\n🏆 Best config : {best}")
    print(f"   SR          = {results[best]['best_sr']:.3f}")
    print(f"   SPL         = {results[best]['best_spl']:.3f}")
    print(f"   NE          = {results[best]['best_ne']:.3f}")
    print(f"\nRun: open learning_curves_{best}.png")
