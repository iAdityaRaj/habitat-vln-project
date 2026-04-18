"""
VLN Model with CLIP encoders + GRU state encoder
Based on VLN-CE (jacobkrantz) architecture but replacing:
  - InstructionEncoder (GRU+GloVe) → CLIP Text Encoder
  - RGB ResNet Encoder             → CLIP Image Encoder
"""

import torch
import torch.nn as nn
import clip
from PIL import Image
import torchvision.transforms as T

# ── Action space ─────────────────────────────────────────────────────────────
# 0=STOP, 1=MOVE_FORWARD, 2=TURN_LEFT, 3=TURN_RIGHT
NUM_ACTIONS = 4
ACTION_NAMES = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]


# ── 1. CLIP VISUAL ENCODER ───────────────────────────────────────────────────
class CLIPVisualEncoder(nn.Module):
    """
    Replaces VLN-CE's TorchVisionResNet50.
    Uses CLIP ViT-B/32 image encoder.
    Input  : RGB image tensor (batch, 3, 224, 224)
    Output : visual features  (batch, 512)
    """
    def __init__(self, output_size=512, trainable=False):
        super().__init__()
        self.clip_model, self.preprocess = clip.load(
            "ViT-B/32", device="cpu"
        )
        # Freeze CLIP weights (pretrained)
        if not trainable:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # Project to output_size if needed
        self.projector = nn.Linear(512, output_size)
        self.relu = nn.ReLU()

    @property
    def output_size(self):
        return 512

    def forward(self, rgb):
        """
        rgb: tensor (batch, 3, H, W) — raw habitat observation
        """
        # Resize to 224x224 for CLIP
        rgb = T.functional.resize(rgb, [224, 224])

        # Normalize to CLIP's expected range
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                             device=rgb.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                             device=rgb.device).view(1, 3, 1, 1)
        rgb  = (rgb.float() / 255.0 - mean) / std

        #with torch.no_grad():
         #   feats = self.clip_model.encode_image(rgb)  # (batch, 512)

        self.clip_model = self.clip_model.to(rgb.device)
        with torch.no_grad():
    	    feats = self.clip_model.encode_image(rgb) 

        feats = feats.float()
        
        return self.relu(self.projector(feats))         # (batch, 512)


# ── 2. CLIP TEXT ENCODER ─────────────────────────────────────────────────────
class CLIPTextEncoder(nn.Module):
    """
    Replaces VLN-CE's InstructionEncoder (GRU + GloVe).
    Uses CLIP text transformer.
    Input  : list of instruction strings
    Output : text features (batch, 512)
    """
    def __init__(self, output_size=512, trainable=False):
        super().__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device="cpu")

        if not trainable:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        self.projector = nn.Linear(512, output_size)
        self.relu = nn.ReLU()

    @property
    def output_size(self):
        return 512

    def forward(self, instructions):
        """
        instructions: list of strings
        e.g. ["Go to the kitchen", "Turn left at corridor"]
        """
        device = next(self.projector.parameters()).device
        tokens = clip.tokenize(instructions, truncate=True).to(device)

        with torch.no_grad():
            feats = self.clip_model.encode_text(tokens)  # (batch, 512)

        feats = feats.float()
        return self.relu(self.projector(feats))           # (batch, 512)


# ── 3. FUSION + GRU STATE ENCODER ────────────────────────────────────────────
class VLNStateEncoder(nn.Module):
    """
    Replaces VLN-CE's build_rnn_state_encoder.
    Fuses visual + text features and maintains temporal state via GRU.
    
    Input  : visual (batch, 512) + text (batch, 512)
    Output : hidden state (batch, hidden_size)
    """
    def __init__(self, input_size=1024, hidden_size=512):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, hidden_state=None):
        """
        x           : (batch, input_size) — fused features for this step
        hidden_state: (1, batch, hidden_size) — memory from previous step
        """
        x = x.unsqueeze(1)  # (batch, 1, input_size) — single timestep
        out, hidden_state = self.gru(x, hidden_state)
        out = self.layer_norm(out.squeeze(1))  # (batch, hidden_size)
        return out, hidden_state


# ── 4. POLICY HEAD ────────────────────────────────────────────────────────────
class PolicyHead(nn.Module):
    """
    Same role as VLN-CE's ILPolicy action distribution.
    Input  : hidden state (batch, hidden_size)
    Output : action logits (batch, num_actions)
    """
    def __init__(self, hidden_size=512, num_actions=4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        return self.head(x)  # (batch, 4)


# ── 5. COMPLETE VLN MODEL ────────────────────────────────────────────────────
class VLNModel(nn.Module):
    """
    Full Vision-Language Navigation model.
    
    Architecture:
        RGB image   → CLIP Image Encoder → visual_feats (512)
        instruction → CLIP Text Encoder  → text_feats   (512)
        [visual_feats; text_feats]        → concat (1024)
                                          → GRU state encoder (512)
                                          → Policy Head
                                          → action logits (4)
    """
    def __init__(self, feature_dim=512, num_actions=4):
        super().__init__()
        self.visual_encoder = CLIPVisualEncoder(output_size=feature_dim)
        self.text_encoder   = CLIPTextEncoder(output_size=feature_dim)
        self.state_encoder  = VLNStateEncoder(
            input_size=feature_dim * 2,
            hidden_size=feature_dim
        )
        self.policy_head    = PolicyHead(
            hidden_size=feature_dim,
            num_actions=num_actions
        )

    def forward(self, rgb, instructions, hidden_state=None):
        """
        rgb          : tensor (batch, 3, H, W)
        instructions : list of strings
        hidden_state : GRU state from previous step (or None)

        returns:
            action_logits : (batch, 4)
            hidden_state  : updated GRU state
        """
        visual_feats = self.visual_encoder(rgb)           # (batch, 512)
        text_feats   = self.text_encoder(instructions)    # (batch, 512)

        # Concatenate — same as VLN-CE's torch.cat([inst, depth, rgb])
        fused = torch.cat([visual_feats, text_feats], dim=1)  # (batch, 1024)

        # GRU maintains memory across steps — key for navigation
        state, hidden_state = self.state_encoder(fused, hidden_state)

        action_logits = self.policy_head(state)           # (batch, 4)
        return action_logits, hidden_state

    def predict_action(self, rgb, instruction, hidden_state=None):
        """
        Single step inference during navigation.
        Returns action index + updated hidden state.
        """
        self.eval()
        with torch.no_grad():
            logits, hidden_state = self.forward(
                rgb.unsqueeze(0), [instruction], hidden_state
            )
            action = torch.argmax(logits, dim=1).item()
        return action, hidden_state


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading CLIP + building VLN model...\n")
    model = VLNModel(feature_dim=512, num_actions=4)

    # Fake inputs (same shape as habitat observations)
    dummy_rgb = torch.randint(0, 255, (2, 3, 256, 256), dtype=torch.uint8)
    dummy_instructions = [
        "Go to the kitchen and stop near the counter",
        "Turn left at the corridor and stop by the elevator"
    ]

    logits, hidden = model(dummy_rgb.float(), dummy_instructions)

    print("=" * 50)
    print("         CLIP-VLN MODEL TEST")
    print("=" * 50)
    print(f"Input RGB shape      : {dummy_rgb.shape}")
    print(f"Instructions         : {len(dummy_instructions)} sentences")
    print(f"Output logits shape  : {logits.shape}")
    print(f"Hidden state shape   : {hidden.shape}")
    print()
    actions = torch.argmax(logits, dim=1)
    for i, a in enumerate(actions):
        print(f"Instruction {i+1} → {ACTION_NAMES[a]}")
    print("=" * 50)
    print("✅ CLIP-VLN Model built successfully!")
