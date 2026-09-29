"""
Streamlit demo: upload chest X-ray view(s), generate a draft findings report.

Run with:
    streamlit run app.py
"""

import streamlit as st
import torch
from PIL import Image

from dataset import build_transform
from model import ReportGenerator
from vocab import PAD_TOKEN, Vocabulary

st.set_page_config(page_title="Radiology Report Generation", layout="centered")


@st.cache_resource
def load_model(checkpoint_path, vocab_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocabulary().load(vocab_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})

    model = ReportGenerator(
        vocab_size=len(vocab),
        d_model=ckpt_args.get("d_model", 256),
        backbone=ckpt_args.get("backbone", "resnet50"),
        pretrained=False,  # weights come from the checkpoint below, not ImageNet
        nhead=ckpt_args.get("nhead", 8),
        num_layers=ckpt_args.get("num_layers", 4),
        max_len=ckpt_args.get("max_len", 100),
        pad_idx=vocab.word2idx[PAD_TOKEN],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return {
        "model": model,
        "vocab": vocab,
        "device": device,
        "image_size": ckpt_args.get("image_size", 224),
        "max_images": ckpt_args.get("max_images", 2),
        "max_len": ckpt_args.get("max_len", 100),
    }


def main():
    st.title("Radiology Report Generation")
    st.caption("Vision-language model: upload chest X-ray view(s) and generate a draft findings report.")

    checkpoint_path = st.sidebar.text_input("Checkpoint path", "checkpoints/best_model.pt")
    vocab_path = st.sidebar.text_input("Vocab path", "vocab.json")

    frontal = st.file_uploader("Frontal view", type=["png", "jpg", "jpeg"])
    lateral = st.file_uploader("Lateral view (optional)", type=["png", "jpg", "jpeg"])

    if frontal is None:
        st.info("Upload at least a frontal chest X-ray to generate a report.")
        return

    try:
        ctx = load_model(checkpoint_path, vocab_path)
    except FileNotFoundError:
        st.error("Checkpoint or vocab file not found. Train a model first with train.py.")
        return

    transform = build_transform(ctx["image_size"], train=False)
    views = [Image.open(frontal).convert("RGB")]
    if lateral is not None:
        views.append(Image.open(lateral).convert("RGB"))

    tensors = [transform(v) for v in views]
    while len(tensors) < ctx["max_images"]:
        tensors.append(tensors[-1].clone())
    tensors = tensors[: ctx["max_images"]]
    image_tensor = torch.stack(tensors, dim=0).unsqueeze(0)  # (1, V, 3, H, W)

    st.image(views, caption=["Frontal", "Lateral"][: len(views)], width=250)

    if st.button("Generate report"):
        with st.spinner("Generating..."):
            generated_ids = ctx["model"].generate(
                image_tensor, ctx["vocab"], max_len=ctx["max_len"], device=ctx["device"]
            )
            report = ctx["vocab"].decode(generated_ids[0].tolist())
        st.subheader("Generated findings")
        st.write(report if report.strip() else "(model produced an empty report — needs more training)")

    st.warning("Research / portfolio demo only — outputs are not validated for clinical use.")


if __name__ == "__main__":
    main()
