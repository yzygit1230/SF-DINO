# -*- coding: utf-8 -*-
"""
DINOv3-only Baseline
For ablation study
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from model.dinov3.models.vision_transformer import vit_large, vit_base
from model.FSHA import FrequencySelectiveHashingAttention


class PrototypeContrastHead(nn.Module):
    def __init__(self, feat_dim, num_classes, scale=30.0, ema_m=0.9):
        super().__init__()
        self.num_classes = num_classes
        self.scale = scale
        self.ema_m = ema_m
        self.register_buffer("prototypes", F.normalize(torch.randn(num_classes, feat_dim), dim=-1))
        self.fc = nn.Linear(feat_dim, num_classes, bias=False)

    @torch.no_grad()
    def update_prototypes(self, feats, labels):
        for c in labels.unique():
            idx = (labels == c)
            if idx.any():
                mean_feat = F.normalize(feats[idx].mean(dim=0, keepdim=True), dim=-1)
                new_val = F.normalize(
                    self.ema_m * self.prototypes[c] + (1 - self.ema_m) * mean_feat.squeeze(0),
                    dim=-1
                )
                self.prototypes[c].copy_(new_val)

    def forward(self, feats, labels=None, mix_lambda=0.5):
        if labels is not None and labels.ndim == 2:
            labels = torch.argmax(labels, dim=1)

        z = F.normalize(feats, dim=-1)

        with torch.no_grad():
            proto = self.prototypes.detach().clone()

        logits_proto = self.scale * (z @ proto.t())

        logits_linear = self.fc(feats)

        logits = mix_lambda * logits_proto + (1 - mix_lambda) * logits_linear

        if self.training and labels is not None:
            with torch.no_grad():
                self.update_prototypes(z.detach(), labels)

        return logits, logits_proto, logits_linear

class SMAdapter(nn.Module):
    def __init__(self, dim=768, bottleneck_dim=64, kernel_size=3):
        super().__init__()
        
        self.down_proj = nn.Linear(dim, bottleneck_dim)
        
        self.conv = nn.Conv2d(
            bottleneck_dim, bottleneck_dim, 
            kernel_size=kernel_size, 
            padding=kernel_size//2, 
            groups=bottleneck_dim # Depthwise
        )
        self.act = nn.GELU()
        
        self.up_proj = nn.Linear(bottleneck_dim, dim)
        
        self.scale = nn.Parameter(torch.zeros(1) + 1e-4)

    def forward(self, x):
        residual = x
        B, N, C = x.shape
        H = W = int(N ** 0.5) 
        
        x = self.down_proj(x) # [B, N, bottleneck]
        x = x.permute(0, 2, 1).view(B, -1, H, W) # [B, bottleneck, H, W]
        x = self.conv(x)
        x = self.act(x)
        x = x.flatten(2).transpose(1, 2) # [B, N, bottleneck]
        x = self.up_proj(x) # [B, N, C]
        
        return x * self.scale

class SFDINO(nn.Module):
    def __init__(self, num_classes=5, pretrained=False):
        super().__init__()
        self.backbone = vit_base(patch_size=16, pretrained=pretrained)
        self.embed_dim = 768

        # Forzen DINO
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False

        self.adapters = nn.ModuleList([
            SMAdapter(dim=self.embed_dim, bottleneck_dim=64) 
            for _ in range(len(self.backbone.blocks))
        ])

        self.norm = nn.LayerNorm(self.embed_dim)
        self.pch_head = PrototypeContrastHead(self.embed_dim, num_classes)
        self.att = FrequencySelectiveHashingAttention(n_hashes = 2, embed_dim = self.embed_dim)

    def forward(self, x, labels=None):
        feats = self.backbone.patch_embed(x)      
        B, H, W, C = feats.shape
        feats = feats.reshape(B, H * W, C)       
        
        for i, blk in enumerate(self.backbone.blocks):
            feats_trans = blk(feats) 
            feats_conv = self.adapters[i](feats)
            feats = feats_trans + feats_conv

        feats = self.backbone.norm(feats)        

        feats = feats.permute(0, 2, 1).contiguous().view(B, C, H, W)
        feats = self.att(feats)
        feats = feats.flatten(2).transpose(1, 2).contiguous()
        feats = feats.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()


        feat_vec = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        feat_vec = self.norm(feat_vec)
        
        logits, logits_proto, logits_linear = self.pch_head(feat_vec, labels=labels)
        return {"logits": logits, "logits_proto": logits_proto, "logits_linear": logits_linear, "feat": feat_vec}

    