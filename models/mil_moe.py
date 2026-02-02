import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.multimodal_resnet import MultiModalResNet1D
except ImportError:
    from .multimodal_resnet import MultiModalResNet1D


class MoEEmotionClassifier(nn.Module):
    def __init__(self, encoder, segment_length, expert_hidden_dims, 
                 feature_dim=128, num_classes=4):
        super(MoEEmotionClassifier, self).__init__()
        self.encoder = encoder
        self.segment_length = segment_length
        self.feature_dim = feature_dim
        self.num_experts = len(expert_hidden_dims)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hid),
                nn.ReLU(inplace=True),
                nn.Linear(hid, feature_dim)
            ) for hid in expert_hidden_dims
        ])

        self.gating = nn.Linear(feature_dim, self.num_experts)
        self.segment_classifier = nn.Linear(feature_dim, num_classes)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        B, total_channels, T = x.shape
        L = self.segment_length
        N = math.ceil(T / L)
        pad_len = N * L - T
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))
        x_seg = x.unfold(2, L, L)
        x_seg = x_seg.permute(0, 2, 1, 3).contiguous()
        x_batch = x_seg.view(B * N, total_channels, L)

        logits_seg_enc, feats_flat, tokens_enc = self.encoder(x_batch)
        feats = feats_flat.view(B, N, self.feature_dim)

        gate_weights = F.softmax(self.gating(feats), dim=-1)
        expert_outs = torch.stack([expert(feats) for expert in self.experts], dim=-1)
        combined = (expert_outs * gate_weights.unsqueeze(2)).sum(dim=-1)

        segment_logits = self.segment_classifier(combined)
        trial_rep = combined.mean(dim=1)
        trial_logits = self.classifier(trial_rep)

        return trial_logits, segment_logits, trial_rep
