"""
Training script for the radiology report generation model.

Usage:
    python train.py --annotation_path annotation.json --image_dir images/ --epochs 30

See README.md for how to obtain annotation.json and images/.
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import IUXrayDataset
from model import ReportGenerator
from vocab import PAD_TOKEN, Vocabulary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--annotation_path", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--vocab_path", default="vocab.json")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--max_len", type=int, default=100)
    p.add_argument("--max_images", type=int, default=2)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--backbone", default="resnet50", choices=["resnet18", "resnet50"])
    p.add_argument("--no_pretrained", action="store_true",
                    help="Train the CNN backbone from scratch. Avoids downloading ImageNet "
                         "weights but needs much more data/epochs to converge.")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--min_word_freq", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--backbone_lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=7)
    return p.parse_args()


def get_device():
    if torch.cuda.is_available():
        try:
            _ = (torch.zeros(1, device="cuda") + 1).cpu()
            return torch.device("cuda")
        except Exception:
            return torch.device("cpu")
    return torch.device("cpu")


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    with open(args.annotation_path) as f:
        ann = json.load(f)

    if os.path.exists(args.vocab_path):
        vocab = Vocabulary().load(args.vocab_path)
        print(f"Loaded existing vocab ({len(vocab)} tokens) from {args.vocab_path}")
    else:
        reports = [item["report"] for item in ann["train"]]
        vocab = Vocabulary(min_freq=args.min_word_freq).build(reports)
        vocab.save(args.vocab_path)
        print(f"Built new vocab ({len(vocab)} tokens) from {len(reports)} training reports, "
              f"saved to {args.vocab_path}")

    train_ds = IUXrayDataset(args.annotation_path, args.image_dir, vocab, split="train",
                              image_size=args.image_size, max_len=args.max_len, max_images=args.max_images)
    val_ds = IUXrayDataset(args.annotation_path, args.image_dir, vocab, split="val",
                            image_size=args.image_size, max_len=args.max_len, max_images=args.max_images)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ReportGenerator(
        vocab_size=len(vocab), d_model=args.d_model, backbone=args.backbone,
        pretrained=not args.no_pretrained, nhead=args.nhead, num_layers=args.num_layers,
        dropout=args.dropout, max_len=args.max_len, pad_idx=vocab.word2idx[PAD_TOKEN],
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.word2idx[PAD_TOKEN], label_smoothing=0.1)
    
    # Differential LR: fine-tune pretrained backbone with lower lr
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.backbone.parameters(), "lr": args.backbone_lr, "weight_decay": args.weight_decay},
        {"params": model.encoder.project.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": model.decoder.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, reports in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            images, reports = images.to(device), reports.to(device)
            input_ids = reports[:, :-1]
            target_ids = reports[:, 1:]

            optimizer.zero_grad()
            logits = model(images, input_ids)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, reports in val_loader:
                images, reports = images.to(device), reports.to(device)
                input_ids = reports[:, :-1]
                target_ids = reports[:, 1:]
                logits = model(images, input_ids)
                loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
            }, ckpt_path)
            print(f"  -> New best model saved (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered at epoch {epoch} (no val improvement for {args.patience} epochs).")
                break

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
