"""
models.py — All model architectures for the Text Emotion Recognition project.

Contains:
    1. TextCNN          — Kim (2014) baseline with GloVe embeddings
    2. BiLSTMAttention  — BiLSTM + self-attention baseline
    3. VanillaBERT      — Standard BERT [CLS] → Linear fine-tuning
    4. BERTCNNLocal     — BERT token embeddings → CNN (local branch only)
    5. BERTCLSGlobal    — BERT [CLS] → MLP (global branch only)
    6. DualBranchModel  — Full proposed model: local + global + gated fusion

All models share a common interface:
    forward(input_ids, attention_mask) -> logits  (shape: [batch, num_classes])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, List, Optional, Tuple


# ============================================================================
#  1. TextCNN Baseline (Kim 2014)
# ============================================================================

class TextCNN(nn.Module):
    """
    Multi-kernel CNN for text classification (Kim, 2014).

    Architecture:
        Input (word indices) → Embedding (GloVe) → Parallel Conv1D filters
        with different kernel sizes → Max-over-time pooling → Concat → Dense → Softmax

    This captures local n-gram patterns but has NO mechanism for sentence-level
    global semantics — exactly the limitation identified in the project handout.

    Args:
        num_classes: Number of emotion categories.
        vocab_size: Vocabulary size.
        embedding_dim: Word embedding dimension (300 for GloVe).
        pretrained_embeddings: Optional pretrained weight matrix.
        num_filters: Number of output channels per kernel size.
        kernel_sizes: Tuple of convolutional kernel widths.
        dropout: Dropout probability.
    """
    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        embedding_dim: int = 300,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        num_filters: int = 128,
        kernel_sizes: Tuple[int, ...] = (2, 3, 4),
        dropout: float = 0.3
    ):
        super().__init__()

        # Embedding layer (optionally initialized with GloVe)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True  # Allow fine-tuning

        # Parallel convolution filters with different kernel sizes
        # Each captures n-grams of different lengths
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=0
            )
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        feature_dim = num_filters * len(kernel_sizes)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len] word indices.
            attention_mask: [batch, seq_len] (unused here, for interface compat).
        Returns:
            logits: [batch, num_classes].
        """
        # [batch, seq_len, embed_dim]
        x = self.embedding(input_ids)

        # Conv1d expects [batch, channels, length], so transpose
        x = x.transpose(1, 2)  # [batch, embed_dim, seq_len]

        # Apply each kernel and max-pool over time
        conv_outputs = []
        for conv in self.convs:
            c = F.relu(conv(x))       # [batch, num_filters, seq_len - k + 1]
            c = c.max(dim=2).values   # [batch, num_filters] — max-over-time
            conv_outputs.append(c)

        # Concatenate all kernel outputs
        out = torch.cat(conv_outputs, dim=1)  # [batch, num_filters * num_kernels]
        out = self.dropout(out)
        logits = self.classifier(out)

        return logits


# ============================================================================
#  2. BiLSTM + Attention Baseline
# ============================================================================

class BiLSTMAttention(nn.Module):
    """
    Bidirectional LSTM with self-attention for text classification.

    Architecture:
        Input → Embedding → BiLSTM → Self-Attention → Weighted sum → Dense → Softmax

    The attention mechanism lets the model focus on emotionally relevant
    words, but the sequential processing creates an information bottleneck
    for long-range dependencies (as noted in the project handout).

    Args:
        num_classes: Number of emotion categories.
        vocab_size: Vocabulary size.
        embedding_dim: Word embedding dimension.
        pretrained_embeddings: Optional pretrained weight matrix.
        hidden_size: LSTM hidden state dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability.
        bidirectional: Use bidirectional LSTM.
    """
    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        embedding_dim: int = 300,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # Self-attention: learn which timesteps to attend to
        self.attention_w = nn.Linear(lstm_output_size, 1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_output_size, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        x = self.embedding(input_ids)             # [batch, seq_len, embed_dim]
        lstm_out, _ = self.lstm(x)                 # [batch, seq_len, 2*hidden]

        # Self-attention scores
        attn_scores = self.attention_w(lstm_out).squeeze(-1)  # [batch, seq_len]

        # Mask padding positions (set to -inf so softmax gives them ~0 weight)
        attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=1)          # [batch, seq_len]

        # Weighted sum of LSTM outputs
        context = torch.bmm(
            attn_weights.unsqueeze(1),  # [batch, 1, seq_len]
            lstm_out                     # [batch, seq_len, 2*hidden]
        ).squeeze(1)                     # [batch, 2*hidden]

        context = self.dropout(context)
        logits = self.classifier(context)

        return logits


# ============================================================================
#  3. Vanilla BERT Baseline
# ============================================================================

class VanillaBERT(nn.Module):
    """
    Standard BERT fine-tuning: [CLS] → Linear → Softmax.

    This is the most common approach and serves as the primary Transformer
    baseline. It uses only the [CLS] token for classification, which
    captures global semantics but doesn't explicitly model local word-level
    emotion signals separately — the gap our dual-branch model addresses.

    Args:
        num_classes: Number of emotion categories.
        encoder_name: HuggingFace model identifier.
        hidden_size: BERT hidden dimension (768 for base).
        dropout: Dropout probability.
        freeze_n_layers: Number of transformer layers to freeze (from bottom).
    """
    def __init__(
        self,
        num_classes: int,
        encoder_name: str = "bert-base-uncased",
        hidden_size: int = 768,
        dropout: float = 0.3,
        freeze_n_layers: int = 0
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name)
        self._freeze_layers(freeze_n_layers)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def _freeze_layers(self, n_layers: int) -> None:
        """Freeze embedding layer and first n transformer layers."""
        if n_layers <= 0:
            return

        # Freeze embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False

        # Freeze transformer layers
        if hasattr(self.encoder, "encoder"):
            layers = self.encoder.encoder.layer
        elif hasattr(self.encoder, "transformer"):
            layers = self.encoder.transformer.layer
        else:
            return

        for i in range(min(n_layers, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits


# ============================================================================
#  4. BERT + CNN (Local Branch Only)
# ============================================================================

class BERTCNNLocal(nn.Module):
    """
    BERT token embeddings → Multi-kernel CNN → Classifier.

    Tests the local branch in isolation: uses BERT's contextualized
    token embeddings as input to a CNN that captures n-gram emotion
    patterns. No [CLS]-based global features.

    This isolates the contribution of the local branch for the ablation study.
    """
    def __init__(
        self,
        num_classes: int,
        encoder_name: str = "bert-base-uncased",
        hidden_size: int = 768,
        cnn_num_filters: int = 128,
        cnn_kernel_sizes: Tuple[int, ...] = (2, 3, 4),
        dropout: float = 0.3,
        freeze_n_layers: int = 8
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name)
        self._freeze_layers(freeze_n_layers)

        # Multi-kernel 1D CNN over token embeddings
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_size, cnn_num_filters, k)
            for k in cnn_kernel_sizes
        ])

        feature_dim = cnn_num_filters * len(cnn_kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def _freeze_layers(self, n_layers: int) -> None:
        if n_layers <= 0:
            return
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        if hasattr(self.encoder, "encoder"):
            layers = self.encoder.encoder.layer
        elif hasattr(self.encoder, "transformer"):
            layers = self.encoder.transformer.layer
        else:
            return
        for i in range(min(n_layers, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeds = outputs.last_hidden_state  # [batch, seq_len, hidden]

        # Conv1d: [batch, hidden, seq_len]
        x = token_embeds.transpose(1, 2)

        conv_outputs = []
        for conv in self.convs:
            c = F.relu(conv(x))         # [batch, filters, seq-k+1]
            c = c.max(dim=2).values     # [batch, filters]
            conv_outputs.append(c)

        local_features = torch.cat(conv_outputs, dim=1)  # [batch, filters*num_kernels]
        local_features = self.dropout(local_features)
        logits = self.classifier(local_features)
        return logits


# ============================================================================
#  5. BERT + [CLS] Global (Global Branch Only)
# ============================================================================

class BERTCLSGlobal(nn.Module):
    """
    BERT [CLS] → 2-layer MLP → Classifier.

    Tests the global branch in isolation: uses only the [CLS] token
    representation processed through a two-layer MLP. No CNN-based
    local features.

    This isolates the contribution of the global branch for the ablation study.
    """
    def __init__(
        self,
        num_classes: int,
        encoder_name: str = "bert-base-uncased",
        hidden_size: int = 768,
        global_hidden_dim: int = 384,
        dropout: float = 0.3,
        freeze_n_layers: int = 8
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name)
        self._freeze_layers(freeze_n_layers)

        # Two-layer MLP on [CLS]
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, global_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(global_hidden_dim, global_hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(global_hidden_dim, num_classes)

    def _freeze_layers(self, n_layers: int) -> None:
        if n_layers <= 0:
            return
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        if hasattr(self.encoder, "encoder"):
            layers = self.encoder.encoder.layer
        elif hasattr(self.encoder, "transformer"):
            layers = self.encoder.transformer.layer
        else:
            return
        for i in range(min(n_layers, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
        global_features = self.mlp(cls_output)             # [batch, global_dim]
        global_features = self.dropout(global_features)
        logits = self.classifier(global_features)
        return logits


# ============================================================================
#  6. Dual-Branch Local–Global Fusion Model (PROPOSED)
# ============================================================================

class GatedFusion(nn.Module):
    """
    Learned gating mechanism for fusing local and global features.

    Computes a gate vector g ∈ [0,1]^d that dynamically weights
    the contribution of each branch per sample:

        g = σ(W_g · [local ⊕ global] + b_g)
        fused = g ⊙ local + (1 - g) ⊙ global

    Intuition: For sarcastic text, the gate should lean toward global
    context (sentence meaning). For straightforward emotional text
    ("I'm furious!"), the gate should lean toward local word cues.
    The gate learns this automatically from data.
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        # Gate network: takes concatenated features, outputs per-dim weights
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )

    def forward(
        self,
        local_feat: torch.Tensor,
        global_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            local_feat:  [batch, feature_dim] from CNN branch.
            global_feat: [batch, feature_dim] from [CLS] branch.
        Returns:
            fused: [batch, feature_dim] gated combination.
        """
        combined = torch.cat([local_feat, global_feat], dim=1)
        g = self.gate(combined)               # [batch, feature_dim], values in [0,1]
        fused = g * local_feat + (1 - g) * global_feat
        return fused


class AttentionFusion(nn.Module):
    """
    Attention-based fusion as an alternative to gating.

    Computes attention weights over [local, global] feature vectors
    to produce a weighted combination.
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.scale = feature_dim ** 0.5

    def forward(
        self,
        local_feat: torch.Tensor,
        global_feat: torch.Tensor
    ) -> torch.Tensor:
        # Stack: [batch, 2, feature_dim]
        features = torch.stack([local_feat, global_feat], dim=1)

        # Compute attention: use mean as query
        q = self.query(features.mean(dim=1, keepdim=True))  # [batch, 1, dim]
        k = self.key(features)                                # [batch, 2, dim]

        attn = torch.bmm(q, k.transpose(1, 2)) / self.scale  # [batch, 1, 2]
        attn = F.softmax(attn, dim=-1)

        fused = torch.bmm(attn, features).squeeze(1)          # [batch, dim]
        return fused


class BilinearFusion(nn.Module):
    """
    Bilinear fusion: captures multiplicative interactions between branches.
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        self.bilinear = nn.Bilinear(feature_dim, feature_dim, feature_dim)

    def forward(
        self,
        local_feat: torch.Tensor,
        global_feat: torch.Tensor
    ) -> torch.Tensor:
        return self.bilinear(local_feat, global_feat)


class DualBranchModel(nn.Module):
    """
    ╔══════════════════════════════════════════════════════════════════╗
    ║  PROPOSED MODEL: Dual-Branch Local–Global Fusion for TER       ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  Input text → BERT Encoder → Token embeddings + [CLS]           ║
    ║                                                                  ║
    ║  ┌─────────────────┐     ┌─────────────────────┐               ║
    ║  │  LOCAL BRANCH    │     │  GLOBAL BRANCH       │               ║
    ║  │  Token embeds    │     │  [CLS] embedding     │               ║
    ║  │  → CNN(2,3,4)    │     │  → MLP(768→384→384)  │               ║
    ║  │  → MaxPool       │     │                       │               ║
    ║  │  → 384-dim       │     │  → 384-dim            │               ║
    ║  └────────┬────────┘     └──────────┬────────────┘               ║
    ║           │                          │                           ║
    ║           └──────────┬───────────────┘                           ║
    ║                      ▼                                           ║
    ║            ┌─────────────────┐                                   ║
    ║            │  GATED FUSION   │                                   ║
    ║            │  g = σ(W·[l⊕g]) │                                   ║
    ║            │  f = g⊙l+(1-g)⊙g│                                   ║
    ║            └────────┬────────┘                                   ║
    ║                     ▼                                            ║
    ║            ┌─────────────────┐                                   ║
    ║            │  CLASSIFIER     │                                   ║
    ║            │  Linear → Softmax│                                  ║
    ║            └─────────────────┘                                   ║
    ╚══════════════════════════════════════════════════════════════════╝

    Args:
        num_classes: Number of emotion categories.
        encoder_name: HuggingFace model name for the shared encoder.
        hidden_size: Encoder hidden dimension.
        cnn_num_filters: Conv1D output channels per kernel.
        cnn_kernel_sizes: Tuple of kernel widths for local branch.
        global_hidden_dim: Hidden dimension in global branch MLP.
        fusion_type: "gated", "concat", "average", "bilinear", "attention".
        dropout: Dropout probability.
        freeze_embeddings: Whether to freeze encoder embeddings.
        freeze_n_layers: Number of encoder layers to freeze.
    """
    def __init__(
        self,
        num_classes: int,
        encoder_name: str = "bert-base-uncased",
        hidden_size: int = 768,
        cnn_num_filters: int = 128,
        cnn_kernel_sizes: Tuple[int, ...] = (2, 3, 4),
        global_hidden_dim: int = 384,
        fusion_type: str = "gated",
        dropout: float = 0.3,
        freeze_embeddings: bool = True,
        freeze_n_layers: int = 8
    ):
        super().__init__()

        self.fusion_type = fusion_type
        self.encoder_name = encoder_name

        # ---------- Shared Encoder ----------
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self._freeze_layers(freeze_embeddings, freeze_n_layers)

        # ---------- Local Branch (CNN on token embeddings) ----------
        self.local_convs = nn.ModuleList([
            nn.Conv1d(hidden_size, cnn_num_filters, k)
            for k in cnn_kernel_sizes
        ])
        local_feature_dim = cnn_num_filters * len(cnn_kernel_sizes)
        # Project to common dimension for fusion
        self.local_projection = nn.Linear(local_feature_dim, global_hidden_dim)
        self.local_dropout = nn.Dropout(dropout)

        # ---------- Global Branch ([CLS] → MLP) ----------
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_size, global_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(global_hidden_dim, global_hidden_dim),
        )
        self.global_dropout = nn.Dropout(dropout)

        # ---------- Fusion Layer ----------
        feature_dim = global_hidden_dim  # Both branches output this dim

        if fusion_type == "gated":
            self.fusion = GatedFusion(feature_dim)
            classifier_input_dim = feature_dim
        elif fusion_type == "concat":
            self.fusion = None  # Simple concatenation
            classifier_input_dim = feature_dim * 2
        elif fusion_type == "average":
            self.fusion = None  # Simple average
            classifier_input_dim = feature_dim
        elif fusion_type == "bilinear":
            self.fusion = BilinearFusion(feature_dim)
            classifier_input_dim = feature_dim
        elif fusion_type == "attention":
            self.fusion = AttentionFusion(feature_dim)
            classifier_input_dim = feature_dim
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # ---------- Classifier ----------
        self.classifier_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(classifier_input_dim, num_classes)

    def _freeze_layers(self, freeze_embeddings: bool, n_layers: int) -> None:
        """Freeze encoder embeddings and first n transformer layers."""
        if freeze_embeddings:
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False

        if n_layers <= 0:
            return

        # Handle different encoder architectures (BERT vs RoBERTa vs DistilBERT)
        if hasattr(self.encoder, "encoder"):
            layers = self.encoder.encoder.layer
        elif hasattr(self.encoder, "transformer"):
            layers = self.encoder.transformer.layer
        else:
            return

        for i in range(min(n_layers, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_features: bool = False,
        return_gate: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through the dual-branch model.

        Args:
            input_ids: [batch, seq_len] token indices.
            attention_mask: [batch, seq_len] padding mask.
            return_features: If True, also return local/global feature vectors
                             (used for t-SNE visualization).
            return_gate: If True, also return gate values
                         (used for gate activation analysis).

        Returns:
            logits: [batch, num_classes] raw scores.
            (optional) dict with 'local_feat', 'global_feat', 'gate_values', 'fused_feat'.
        """
        # ---- Shared encoder ----
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeds = outputs.last_hidden_state  # [batch, seq_len, hidden]
        cls_embed = token_embeds[:, 0, :]          # [batch, hidden]

        # ---- Local branch: CNN on token embeddings ----
        x = token_embeds.transpose(1, 2)  # [batch, hidden, seq_len]

        conv_outputs = []
        for conv in self.local_convs:
            c = F.relu(conv(x))
            c = c.max(dim=2).values
            conv_outputs.append(c)

        local_concat = torch.cat(conv_outputs, dim=1)  # [batch, filters*kernels]
        local_feat = self.local_projection(local_concat)  # [batch, feature_dim]
        local_feat = self.local_dropout(local_feat)

        # ---- Global branch: [CLS] → MLP ----
        global_feat = self.global_mlp(cls_embed)   # [batch, feature_dim]
        global_feat = self.global_dropout(global_feat)

        # ---- Fusion ----
        extras = {}

        if self.fusion_type == "gated":
            # Compute gate values for analysis
            combined = torch.cat([local_feat, global_feat], dim=1)
            gate_values = self.fusion.gate(combined)  # [batch, feature_dim]
            fused = gate_values * local_feat + (1 - gate_values) * global_feat
            if return_gate:
                extras["gate_values"] = gate_values.detach()
        elif self.fusion_type == "concat":
            fused = torch.cat([local_feat, global_feat], dim=1)
        elif self.fusion_type == "average":
            fused = (local_feat + global_feat) / 2.0
        elif self.fusion_type in ["bilinear", "attention"]:
            fused = self.fusion(local_feat, global_feat)
        else:
            fused = torch.cat([local_feat, global_feat], dim=1)

        # ---- Classifier ----
        fused = self.classifier_dropout(fused)
        logits = self.classifier(fused)

        if return_features or return_gate:
            extras["local_feat"] = local_feat.detach()
            extras["global_feat"] = global_feat.detach()
            extras["fused_feat"] = fused.detach()
            return logits, extras

        return logits


# ============================================================================
#  7. Model Factory
# ============================================================================

def build_model(
    model_name: str,
    num_classes: int,
    cfg,
    vocab_size: Optional[int] = None,
    pretrained_embeddings: Optional[torch.Tensor] = None,
    encoder_name: Optional[str] = None,
    cnn_kernel_sizes: Optional[Tuple[int, ...]] = None,
    fusion_type: Optional[str] = None,
) -> nn.Module:
    """
    Factory function to create any model by name.

    This is the single function called by the training pipeline.
    Model-specific params are pulled from cfg, with optional overrides.

    Args:
        model_name: One of "textcnn", "bilstm_attention", "vanilla_bert",
                    "bert_cnn_local", "bert_cls_global", "dual_branch".
        num_classes: Number of emotion classes.
        cfg: ProjectConfig with model hyperparameters.
        vocab_size: For non-BERT models.
        pretrained_embeddings: For non-BERT models (GloVe).
        encoder_name: Override encoder (for encoder comparison experiment).
        cnn_kernel_sizes: Override kernels (for kernel analysis experiment).
        fusion_type: Override fusion (for ablation experiment).

    Returns:
        Instantiated model (nn.Module).
    """
    mcfg = cfg.model
    enc = encoder_name or mcfg.encoder_name
    kernels = cnn_kernel_sizes or mcfg.cnn_kernel_sizes
    fusion = fusion_type or mcfg.fusion_type

    # Determine hidden size based on encoder
    if "distilbert" in enc:
        hidden_size = 768
        freeze_n = min(mcfg.freeze_n_layers, 6)  # DistilBERT has 6 layers
    elif "base" in enc:
        hidden_size = 768
        freeze_n = mcfg.freeze_n_layers
    elif "large" in enc:
        hidden_size = 1024
        freeze_n = mcfg.freeze_n_layers
    else:
        hidden_size = mcfg.hidden_size
        freeze_n = mcfg.freeze_n_layers

    if model_name == "textcnn":
        return TextCNN(
            num_classes=num_classes,
            vocab_size=vocab_size or mcfg.vocab_size,
            embedding_dim=mcfg.embedding_dim,
            pretrained_embeddings=pretrained_embeddings,
            num_filters=mcfg.textcnn_num_filters,
            kernel_sizes=mcfg.textcnn_kernel_sizes,
            dropout=mcfg.dropout,
        )

    elif model_name == "bilstm_attention":
        return BiLSTMAttention(
            num_classes=num_classes,
            vocab_size=vocab_size or mcfg.vocab_size,
            embedding_dim=mcfg.embedding_dim,
            pretrained_embeddings=pretrained_embeddings,
            hidden_size=mcfg.lstm_hidden_size,
            num_layers=mcfg.lstm_num_layers,
            dropout=mcfg.dropout,
            bidirectional=mcfg.lstm_bidirectional,
        )

    elif model_name == "vanilla_bert":
        return VanillaBERT(
            num_classes=num_classes,
            encoder_name=enc,
            hidden_size=hidden_size,
            dropout=mcfg.dropout,
            freeze_n_layers=freeze_n,
        )

    elif model_name == "bert_cnn_local":
        return BERTCNNLocal(
            num_classes=num_classes,
            encoder_name=enc,
            hidden_size=hidden_size,
            cnn_num_filters=mcfg.cnn_num_filters,
            cnn_kernel_sizes=kernels,
            dropout=mcfg.dropout,
            freeze_n_layers=freeze_n,
        )

    elif model_name == "bert_cls_global":
        return BERTCLSGlobal(
            num_classes=num_classes,
            encoder_name=enc,
            hidden_size=hidden_size,
            global_hidden_dim=mcfg.global_hidden_dim,
            dropout=mcfg.dropout,
            freeze_n_layers=freeze_n,
        )

    elif model_name == "dual_branch":
        return DualBranchModel(
            num_classes=num_classes,
            encoder_name=enc,
            hidden_size=hidden_size,
            cnn_num_filters=mcfg.cnn_num_filters,
            cnn_kernel_sizes=kernels,
            global_hidden_dim=mcfg.global_hidden_dim,
            fusion_type=fusion,
            dropout=mcfg.dropout,
            freeze_embeddings=mcfg.freeze_embeddings,
            freeze_n_layers=freeze_n,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")
