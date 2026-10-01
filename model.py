"""
Vision-language model for radiology report generation.

Architecture:
    CNNEncoder   - ResNet backbone extracts a spatial feature map per X-ray
                   view, projected to d_model and flattened into a sequence
                   of "memory" tokens (one or more views are concatenated
                   into a single memory sequence, similar to how multi-view
                   studies are handled in the R2Gen / CoAtt literature).
    ReportDecoder - a standard Transformer decoder that cross-attends to the
                   image memory and generates the report autoregressively,
                   the same way a captioning transformer generates a caption.

This is intentionally a from-scratch, readable implementation rather than
wrapping a pretrained vision-language model - it is meant to be trained on
your own report corpus and vocabulary.
"""

import math

import torch
import torch.nn as nn
from torchvision import models

from vocab import PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN


class CNNEncoder(nn.Module):
    def __init__(self, d_model: int = 256, backbone: str = "resnet50", pretrained: bool = True):
        super().__init__()
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            net = models.resnet50(weights=weights)
            feat_dim = 2048
        elif backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            net = models.resnet18(weights=weights)
            feat_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Drop the final avgpool + fc; keep the spatial feature map so the
        # decoder can attend to different regions of the X-ray.
        self.backbone = nn.Sequential(*list(net.children())[:-2])
        self.project = nn.Conv2d(feat_dim, d_model, kernel_size=1)

    def forward(self, images):
        """
        images: (B, V, 3, H, W) - V views per study (e.g. frontal + lateral)
        returns: (B, V * h' * w', d_model) memory sequence for the decoder
        """
        b, v, c, h, w = images.shape
        images = images.view(b * v, c, h, w)
        feats = self.backbone(images)                      # (B*V, feat_dim, h', w')
        feats = self.project(feats)                         # (B*V, d_model, h', w')
        _, d, fh, fw = feats.shape
        feats = feats.view(b, v, d, fh, fw)
        feats = feats.permute(0, 1, 3, 4, 2).reshape(b, v * fh * fw, d)
        return feats


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class ReportDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 1024,
                 dropout: float = 0.1, max_len: int = 100, pad_idx: int = 0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    @staticmethod
    def _causal_mask(size: int, device):
        # Bool mask (True = blocked) - kept the same dtype as the padding
        # mask below since PyTorch's MultiheadAttention warns (and will
        # eventually error) on mixed float/bool masks.
        return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, tgt_ids, memory):
        """
        tgt_ids: (B, T) teacher-forced token ids
        memory:  (B, S, d_model) encoder output
        returns: (B, T, vocab_size) logits
        """
        tgt_mask = self._causal_mask(tgt_ids.size(1), tgt_ids.device)
        pad_mask = tgt_ids == self.pad_idx

        x = self.embedding(tgt_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        out = self.decoder(tgt=x, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=pad_mask)
        return self.output_proj(out)


class ReportGenerator(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, backbone: str = "resnet50",
                 pretrained: bool = True, nhead: int = 8, num_layers: int = 4,
                 dropout: float = 0.2, max_len: int = 100, pad_idx: int = 0):
        super().__init__()
        self.encoder = CNNEncoder(d_model=d_model, backbone=backbone, pretrained=pretrained)
        self.decoder = ReportDecoder(vocab_size, d_model=d_model, nhead=nhead,
                                      num_layers=num_layers, dropout=dropout,
                                      max_len=max_len, pad_idx=pad_idx)

    def forward(self, images, tgt_ids):
        memory = self.encoder(images)
        return self.decoder(tgt_ids, memory)

    @torch.inference_mode()
    def generate(self, images, vocab, max_len: int = 100, repetition_penalty: float = 1.35,
                 no_repeat_ngram: int = 3, min_len: int = 4, device="cpu"):
        """Autoregressive report generation with repetition penalty & special token masking."""
        self.eval()
        memory = self.encoder(images.to(device))
        b = images.size(0)
        start_idx = vocab.word2idx.get(START_TOKEN, 1)
        end_idx = vocab.word2idx.get(END_TOKEN, 2)
        pad_idx = vocab.word2idx.get(PAD_TOKEN, 0)
        unk_idx = vocab.word2idx.get(UNK_TOKEN, 3)

        generated = torch.full((b, 1), start_idx, dtype=torch.long, device=device)
        finished = torch.zeros(b, dtype=torch.bool, device=device)

        for step in range(max_len - 1):
            logits = self.decoder(generated, memory)[:, -1, :].clone()

            # Prevent generation of special and unk tokens
            logits[:, pad_idx] = -float("inf")
            logits[:, start_idx] = -float("inf")
            logits[:, unk_idx] = -float("inf")

            if step < min_len:
                logits[:, end_idx] = -float("inf")

            # Trigram / n-gram blocking: prevent loops
            if no_repeat_ngram > 0 and generated.size(1) >= no_repeat_ngram:
                for batch_idx in range(b):
                    curr_tokens = generated[batch_idx].tolist()
                    ngram_prefix = tuple(curr_tokens[-(no_repeat_ngram - 1):])
                    for i in range(len(curr_tokens) - no_repeat_ngram + 1):
                        if tuple(curr_tokens[i:i + no_repeat_ngram - 1]) == ngram_prefix:
                            banned = curr_tokens[i + no_repeat_ngram - 1]
                            if banned not in (pad_idx, start_idx, end_idx):
                                logits[batch_idx, banned] = -float("inf")

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for batch_idx in range(b):
                    for token_id in set(generated[batch_idx].tolist()):
                        if token_id in (pad_idx, start_idx, end_idx):
                            continue
                        if logits[batch_idx, token_id] > 0:
                            logits[batch_idx, token_id] /= repetition_penalty
                        else:
                            logits[batch_idx, token_id] *= repetition_penalty

            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            finished |= next_token.squeeze(1) == end_idx
            if finished.all():
                break

        return generated

