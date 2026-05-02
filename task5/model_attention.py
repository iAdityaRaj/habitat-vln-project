"""
Task 5: Attention-based Fusion Model
Improvement over baseline concatenation fusion.

Baseline: visual_feats + text_feats → concat → linear → GRU
Improved: Cross-attention(visual, text) → weighted fusion → GRU

Key insight: attention lets the model focus on RELEVANT parts
of the visual features given the instruction context.
"""

import torch
import torch.nn as nn
import clip


ACTION_NAMES = ['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT']


# ── Shared CLIP Encoders (same as baseline) ───────────────────────────────────
class CLIPVisualEncoder(nn.Module):
    def __init__(self, output_size=512, trainable=False):
        super().__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device="cpu")
        if not trainable:
            for p in self.clip_model.parameters():
                p.requires_grad = False
        self.projector = nn.Linear(512, output_size)
        self.relu      = nn.ReLU()

    def forward(self, x):
        import torchvision.transforms.functional as TF
        x = TF.resize(x.float(), [224, 224]) if x.shape[-1] != 224 else x.float()
        mean = torch.tensor([0.48145466,0.4578275,0.40821073],
                             device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.26862954,0.26130258,0.27577711],
                             device=x.device).view(1,3,1,1)
        x    = (x/255.0 - mean) / std
        with torch.no_grad():
            feats = self.clip_model.encode_image(x)
        return self.relu(self.projector(feats.float()))


class CLIPTextEncoder(nn.Module):
    def __init__(self, output_size=512, trainable=False):
        super().__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device="cpu")
        if not trainable:
            for p in self.clip_model.parameters():
                p.requires_grad = False
        self.projector = nn.Linear(512, output_size)
        self.relu      = nn.ReLU()

    def forward(self, instructions):
        device = next(self.projector.parameters()).device
        tokens = clip.tokenize(instructions, truncate=True).to(device)
        with torch.no_grad():
            feats = self.clip_model.encode_text(tokens)
        return self.relu(self.projector(feats.float()))


# ── BASELINE: Simple Concatenation Fusion ─────────────────────────────────────
class ConcatFusion(nn.Module):
    """Baseline: just concatenate visual + text features."""
    def __init__(self, feature_dim=512):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
    def forward(self, visual, text):
        return self.fusion(torch.cat([visual, text], dim=1))


# ── IMPROVED: Cross-Attention Fusion ─────────────────────────────────────────
class CrossAttentionFusion(nn.Module):
    """
    Task 5 Improvement: Cross-attention fusion.

    The text instruction acts as a QUERY.
    The visual features act as KEY and VALUE.
    Attention weights decide which visual aspects matter for the instruction.

    Example:
    - Instruction: "Turn left" → attention focuses on left-side visual features
    - Instruction: "Walk forward" → attention focuses on forward-path features
    """
    def __init__(self, feature_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim

        # Multi-head cross attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Self-attention on text for context
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward after attention
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.norm3 = nn.LayerNorm(feature_dim)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )

        # Final projection
        self.output_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, visual_feats, text_feats):
        """
        visual_feats: (batch, feature_dim)
        text_feats:   (batch, feature_dim)
        """
        # Add sequence dimension for attention
        v = visual_feats.unsqueeze(1)  # (batch, 1, dim)
        t = text_feats.unsqueeze(1)    # (batch, 1, dim)

        # Cross-attention: text queries visual
        # "Given this instruction, what visual features are relevant?"
        attended, attn_weights = self.cross_attention(
            query=t,   # text asks question
            key=v,     # visual provides context
            value=v    # visual provides content
        )
        attended = self.norm1(attended + t)  # residual

        # Feed-forward
        out = self.norm2(attended + self.ffn(attended))
        out = out.squeeze(1)  # (batch, dim)

        # Gating: blend attended features with original visual
        gate_input = torch.cat([out, visual_feats], dim=1)
        gate_val   = self.gate(gate_input)
        fused      = gate_val * out + (1 - gate_val) * visual_feats

        return self.norm3(self.output_proj(fused))


# ── GRU State Encoder (same for both) ────────────────────────────────────────
class GRUStateEncoder(nn.Module):
    def __init__(self, input_size=512, hidden_size=512):
        super().__init__()
        self.gru        = nn.GRU(input_size, hidden_size, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, hidden=None):
        out, hidden = self.gru(x.unsqueeze(1), hidden)
        return self.layer_norm(out.squeeze(1)), hidden


# ── Policy Head (same for both) ───────────────────────────────────────────────
class PolicyHead(nn.Module):
    def __init__(self, hidden_size=512, num_actions=4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_actions)
        )
    def forward(self, x):
        return self.head(x)


# ── BASELINE VLN Model ────────────────────────────────────────────────────────
class VLNBaseline(nn.Module):
    """Original model with concatenation fusion."""
    def __init__(self, feature_dim=512, num_actions=4):
        super().__init__()
        self.visual_encoder = CLIPVisualEncoder(output_size=feature_dim)
        self.text_encoder   = CLIPTextEncoder(output_size=feature_dim)
        self.fusion         = ConcatFusion(feature_dim)
        self.state_encoder  = GRUStateEncoder(feature_dim, feature_dim)
        self.policy_head    = PolicyHead(feature_dim, num_actions)

    def forward(self, rgb, instructions, hidden_state=None):
        v      = self.visual_encoder(rgb)
        t      = self.text_encoder(instructions)
        fused  = self.fusion(v, t)
        state, hidden = self.state_encoder(fused, hidden_state)
        return self.policy_head(state), hidden

    def predict_action(self, rgb, instruction, hidden_state=None):
        self.eval()
        with torch.no_grad():
            logits, hidden = self.forward(
                rgb.unsqueeze(0), [instruction], hidden_state)
            action = torch.argmax(logits, dim=1).item()
        return action, hidden


# ── IMPROVED VLN Model with Attention ────────────────────────────────────────
class VLNAttention(nn.Module):
    """Improved model with cross-attention fusion."""
    def __init__(self, feature_dim=512, num_actions=4, num_heads=8):
        super().__init__()
        self.visual_encoder = CLIPVisualEncoder(output_size=feature_dim)
        self.text_encoder   = CLIPTextEncoder(output_size=feature_dim)
        self.fusion         = CrossAttentionFusion(feature_dim, num_heads)
        self.state_encoder  = GRUStateEncoder(feature_dim, feature_dim)
        self.policy_head    = PolicyHead(feature_dim, num_actions)

    def forward(self, rgb, instructions, hidden_state=None):
        v      = self.visual_encoder(rgb)
        t      = self.text_encoder(instructions)
        fused  = self.fusion(v, t)
        state, hidden = self.state_encoder(fused, hidden_state)
        return self.policy_head(state), hidden

    def predict_action(self, rgb, instruction, hidden_state=None):
        self.eval()
        with torch.no_grad():
            logits, hidden = self.forward(
                rgb.unsqueeze(0), [instruction], hidden_state)
            action = torch.argmax(logits, dim=1).item()
        return action, hidden


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing both models...\n")
    dummy_rgb  = torch.randint(0, 255, (2, 3, 256, 256), dtype=torch.float32)
    dummy_inst = ["Walk forward", "Turn left"]

    baseline   = VLNBaseline()
    attention  = VLNAttention()

    b_params = sum(p.numel() for p in baseline.parameters()  if p.requires_grad)
    a_params = sum(p.numel() for p in attention.parameters() if p.requires_grad)

    logits_b, _ = baseline(dummy_rgb, dummy_inst)
    logits_a, _ = attention(dummy_rgb, dummy_inst)

    print("=" * 50)
    print("  MODEL COMPARISON")
    print("=" * 50)
    print(f"Baseline  (concat)    output: {logits_b.shape} "
          f"trainable: {b_params:,}")
    print(f"Attention (cross-att) output: {logits_a.shape} "
          f"trainable: {a_params:,}")
    print(f"Extra params in attention: {a_params-b_params:,}")
    print("✅ Both models built successfully!")
