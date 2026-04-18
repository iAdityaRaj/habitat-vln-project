# 🤖 Vision-Language Navigation with CLIP + Habitat

> Teaching a robot to navigate indoor environments using only a camera and natural language instructions — no map, no GPS.

---

## 📌 Problem Statement

Modern mobile robots deployed in warehouses, hospitals, office buildings, airports, and shopping malls must follow high-level natural-language instructions such as:

> *"Go to the reception desk"*
> *"Turn left at the corridor and stop near the elevator"*

Traditional navigation systems rely on **metric maps and rule-based planners**, which do not naturally support reasoning over language. This project addresses that gap by building a learning-based navigation agent that:

- **Sees** the world through an RGB camera (one frame at a time)
- **Reads** a natural language instruction
- **Decides** which action to take next: `MOVE_FORWARD`, `TURN_LEFT`, `TURN_RIGHT`, or `STOP`

The agent has **no access to a map, no GPS, and no prior knowledge of the environment**. It must navigate purely from visual observations and language — just like a person following verbal directions in an unfamiliar building.

---

## 🎯 Objective

Given:
- A sequence of **RGB observations** from the Habitat simulator
- A **natural language navigation instruction**

Predict:
- A sequence of **discrete navigation actions** that guide the agent to the target location

---

## 🏗️ Architecture

```
┌─────────────────────┐     ┌──────────────────────────┐
│   RGB Image         │     │   Text Instruction        │
│   (256 × 256 × 3)   │     │   "Go to the kitchen..."  │
└────────┬────────────┘     └────────────┬─────────────┘
         │                               │
         ▼                               ▼
┌─────────────────────┐     ┌──────────────────────────┐
│  CLIP Image Encoder │     │   CLIP Text Encoder       │
│  (ViT-B/32)         │     │   (Transformer)           │
│  → 512 features     │     │   → 512 features          │
└────────┬────────────┘     └────────────┬─────────────┘
         │                               │
         └──────────────┬────────────────┘
                        │  Concatenate (1024-dim)
                        ▼
              ┌──────────────────┐
              │  GRU State       │
              │  Encoder         │  ← memory across steps
              │  (512 hidden)    │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │   Policy Head    │
              │  (Linear layers) │
              └────────┬─────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │  STOP  │  MOVE_FORWARD      │
         │  TURN_LEFT  │  TURN_RIGHT   │
         └─────────────────────────────┘
```

### Why CLIP?
The original VLN-CE baseline uses **ResNet (visual) + LSTM + GloVe (text)** — two separate encoders trained independently. We replace both with **CLIP (ViT-B/32)**, which was trained on 400 million image-text pairs to align visual and language representations in the **same embedding space**. This makes fusion more natural and effective.

| Component | VLN-CE Baseline | Our Model |
|---|---|---|
| Visual Encoder | ResNet-50 | CLIP ViT-B/32 |
| Text Encoder | LSTM + GloVe | CLIP Transformer |
| Temporal Memory | GRU | GRU (kept) |
| Fusion | Concatenation | Concatenation |
| Policy | Linear head | Linear head |

---

## ✅ Tasks Completed

### Task 1 — Environment Setup
- Installed **Habitat-Sim 0.3.3** (built from source on Apple M1)
- Installed **Habitat-Lab 0.3.3** with baselines
- Loaded Matterport3D-style indoor 3D scenes (`.glb` format)
- Ran a **random baseline agent** for 50 steps with position logging
- Implemented and understood evaluation metrics: **SR**, **SPL**, **NE**

### Task 2 — Vision-Language Model Implementation
- Built **CLIP Visual Encoder** — encodes RGB frames to 512-dim features
- Built **CLIP Text Encoder** — encodes instructions to 512-dim features
- Built **Fusion Module** — concatenates both (1024-dim)
- Built **GRU State Encoder** — maintains temporal memory across steps
- Built **Policy Head** — predicts action logits over 4 discrete actions
- Built complete **training and validation pipeline** with PyTorch

### Task 3 — Baseline Training and Evaluation
- Trained using **imitation learning (behavior cloning)** on R2R dataset
- Generated **learning curves** (loss, accuracy, SR, SPL, NE)
- Performed **hyperparameter tuning** across 3 configurations
- Reported **Success Rate (SR)** and **SPL** metrics

---

## 📊 Results

### Hyperparameter Tuning

| Config | Learning Rate | Batch Size | SR | SPL | NE | Val Acc |
|---|---|---|---|---|---|---|
| Config 1 | 1e-4 | 16 | 0.492 | 0.242 | 2.540 | 0.484 |
| Config 2 | 5e-4 | 16 | 0.482 | 0.232 | 2.591 | 0.482 |
| Config 3 | 1e-4 | 32 | 0.492 | 0.242 | 2.540 | 0.490 |

🏆 **Best Configuration: lr=1e-4, batch_size=16**
- **Success Rate (SR): 0.492**
- **SPL: 0.242**
- **Navigation Error (NE): 2.540**

### Evaluation Metrics Explained

| Metric | Formula | Interpretation |
|---|---|---|
| **SR** (Success Rate) | Correct actions / Total actions | Higher is better. 1.0 = perfect |
| **SPL** | SR × (optimal_path / actual_path) | Rewards success AND efficiency |
| **NE** (Navigation Error) | Wrong actions per episode | Lower is better |

---

## 🗂️ Project Structure

```
habitat_vln_project/
│
├── task1/
│   ├── task1_setup.py          # Load 3D scene, capture RGB observation
│   ├── task1_baseline.py       # Random baseline agent (50 steps)
│   └── task1_metrics.py        # SR, SPL, NE computation
│
├── task2/
│   ├── model.py                # CLIP-VLN model architecture
│   └── train.py                # Training pipeline (synthetic data)
│
├── task3/
│   ├── model.py                # CLIP-VLN model
│   ├── dataset.py              # R2R dataset loader
│   ├── metrics.py              # SR, SPL, NE metrics
│   ├── train.py                # Training + hyperparameter tuning
│   └── learning_curves_*.png   # Training plots
│
├── VLN-CE/                     # Reference: jacobkrantz/VLN-CE
└── README.md
```

---

## 🛠️ Installation

### Prerequisites
- macOS (Apple M1) or Linux
- Anaconda / Miniconda
- Python 3.9
- Xcode Command Line Tools (macOS)

### Step 1 — Create Environment

```bash
conda create -n habitat_vln python=3.9 -y
conda activate habitat_vln
```

### Step 2 — Install Habitat-Sim (M1 Mac — build from source)

```bash
git clone --branch stable https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt
python setup.py install --headless
cd ..
```

### Step 3 — Install Habitat-Lab

```bash
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab/
pip install -e habitat-baselines/
cd ..
```

### Step 4 — Install Python Dependencies

```bash
pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
pip install transformers tqdm matplotlib
```

### Step 5 — Clone This Repo

```bash
git clone https://github.com/YOUR_USERNAME/habitat-vln-project.git
cd habitat-vln-project
```

---

## 🚀 Running the Code

### Task 1 — Environment Setup

```bash
conda activate habitat_vln
cd task1

# Load scene and save RGB image
python task1_setup.py

# Run random baseline agent
python task1_baseline.py

# Compute metrics
python task1_metrics.py
```

### Task 2 — Build and Test Model

```bash
cd task2

# Test model builds correctly
python model.py

# Train on synthetic data
python train.py
```

### Task 3 — Train on R2R Data

```bash
cd task3

# Download R2R data
mkdir -p data
python -c "
import urllib.request
url = 'https://dl.dropbox.com/s/hh5qec8o5urcztn/R2R_train.json'
urllib.request.urlretrieve(url, 'data/R2R_train.json')
print('Downloaded!')
"

# Run full training + hyperparameter tuning
python train.py

# View learning curves
open learning_curves_lr1e-4_bs16.png   # macOS
```

---

## 📚 References

| Paper | Description |
|---|---|
| [Anderson et al., CVPR 2018](https://arxiv.org/abs/1711.07280) | Vision-and-Language Navigation (base paper) |
| [Krantz et al., ECCV 2020](https://arxiv.org/abs/2004.02857) | VLN in Continuous Environments (VLN-CE) |
| [Radford et al., ICML 2021](https://arxiv.org/abs/2103.00020) | CLIP — Learning from Natural Language Supervision |
| [Habitat Simulator](https://github.com/facebookresearch/habitat-sim) | 3D simulation engine |
| [Habitat Lab](https://github.com/facebookresearch/habitat-lab) | ML framework for embodied AI |
| [VLN-CE](https://github.com/jacobkrantz/VLN-CE) | Reference implementation |

---

## 💻 Hardware

| Component | Spec |
|---|---|
| Machine | Apple MacBook Air M1 |
| RAM | 8GB |
| OS | macOS |
| Python | 3.9 |
| PyTorch | 2.8.0 |
| Habitat-Sim | 0.3.3 |
| Habitat-Lab | 0.3.3 |

