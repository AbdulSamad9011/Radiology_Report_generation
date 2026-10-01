"""
Generate reports on a held-out split and score them against the reference
reports with BLEU and ROUGE-L.

Usage:
    python generate.py --annotation_path annotation.json --image_dir images/ \
        --checkpoint checkpoints/best_model.pt --vocab_path vocab.json --split test
"""

import argparse

import torch
from torch.utils.data import DataLoader

from dataset import IUXrayDataset
from model import ReportGenerator
from vocab import PAD_TOKEN, Vocabulary

try:
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False


def _lcs_length(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(reference_tokens, hypothesis_tokens, beta: float = 1.2):
    """Simple ROUGE-L F-score, no external dependency needed."""
    if not reference_tokens or not hypothesis_tokens:
        return 0.0
    lcs = _lcs_length(reference_tokens, hypothesis_tokens)
    precision = lcs / len(hypothesis_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return ((1 + beta**2) * precision * recall) / (recall + beta**2 * precision)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--annotation_path", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--vocab_path", default="vocab.json")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--split", default="test")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_samples_to_print", type=int, default=5)
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

    vocab = Vocabulary().load(args.vocab_path)
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})

    ds = IUXrayDataset(
        args.annotation_path, args.image_dir, vocab, split=args.split,
        image_size=ckpt_args.get("image_size", 224),
        max_len=ckpt_args.get("max_len", 100),
        max_images=ckpt_args.get("max_images", 2),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = ReportGenerator(
        vocab_size=len(vocab),
        d_model=ckpt_args.get("d_model", 256),
        backbone=ckpt_args.get("backbone", "resnet50"),
        pretrained=False,  # weights are restored from the checkpoint below, not ImageNet
        nhead=ckpt_args.get("nhead", 8),
        num_layers=ckpt_args.get("num_layers", 4),
        dropout=ckpt_args.get("dropout", 0.2),
        max_len=ckpt_args.get("max_len", 100),
        pad_idx=vocab.word2idx[PAD_TOKEN],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    references, hypotheses, rouge_scores = [], [], []
    printed = 0

    for images, reports in loader:
        images = images.to(device)
        generated_ids = model.generate(
            images, vocab, max_len=ckpt_args.get("max_len", 100),
            repetition_penalty=1.3, device=device
        )

        for i in range(images.size(0)):
            hyp = vocab.decode(generated_ids[i].tolist())
            ref = vocab.decode(reports[i].tolist())
            hyp_tokens, ref_tokens = hyp.split(), ref.split()

            hypotheses.append(hyp_tokens)
            references.append([ref_tokens])
            rouge_scores.append(rouge_l(ref_tokens, hyp_tokens))

            if printed < args.num_samples_to_print:
                print("-" * 60)
                print("REFERENCE :", ref)
                print("GENERATED :", hyp)
                printed += 1

    print("-" * 60)
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    print(f"Mean ROUGE-L: {avg_rouge:.4f}")

    if _HAS_NLTK:
        smoothie = SmoothingFunction().method4
        bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        print(f"BLEU-1: {bleu1:.4f} | BLEU-4: {bleu4:.4f}")
    else:
        print("Install nltk (`pip install nltk`) to also compute BLEU scores.")


if __name__ == "__main__":
    main()
