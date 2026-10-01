"""
Streamlit app — Chest X-Ray Radiology Report Generation
Run:  streamlit run app.py
"""

import io
import os
import re

# Force CPU — avoids driver crashes on unsupported GPUs (e.g. GeForce MX350)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.cuda.is_available = lambda: False
torch.set_num_threads(1)

import streamlit as st
from PIL import Image

from dataset import build_transform
from model import ReportGenerator
from vocab import PAD_TOKEN, Vocabulary

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Radiology Report Generation",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
    }
    /* Main background */
    .stApp {
        background-color: #0f172a;
        color: #f1f5f9;
    }
    /* Headers */
    h1, h2, h3 { color: #f8fafc !important; }
    p, label, .stMarkdown { color: #cbd5e1 !important; }

    /* Upload card */
    .upload-box {
        border: 2px dashed #334155;
        border-radius: 12px;
        padding: 22px;
        background: #1e293b;
        text-align: center;
        margin-bottom: 12px;
        transition: border-color 0.2s;
    }
    /* Report box */
    .report-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px 28px;
        line-height: 1.8;
        font-size: 16px;
        color: #e2e8f0;
        margin-top: 8px;
    }
    .report-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #38bdf8;
        margin-bottom: 10px;
    }
    /* Disclaimer */
    .disclaimer {
        font-size: 12px;
        color: #64748b;
        border-top: 1px solid #1e293b;
        padding-top: 14px;
        margin-top: 32px;
    }
    /* Metric cards */
    .stat-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    .stat-value { font-size: 22px; font-weight: 700; color: #38bdf8; }
    .stat-label { font-size: 12px; color: #94a3b8; margin-top: 2px; }
    /* Divider */
    hr { border-color: #1e293b !important; }
    /* Button styling */
    .stButton > button {
        background-color: #0284c7;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.4rem;
        font-weight: 600;
        font-size: 15px;
        width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover { background-color: #0369a1; }
</style>
""", unsafe_allow_html=True)


# ── Helper: post-process raw model output ──────────────────────────────────────
def format_report(raw_text: str) -> str:
    """
    Clean up the model's raw token sequence:
      1. Remove consecutive duplicate tokens  (model repetition)
      2. Drop tokens that appear more than 3 times total
      3. Fix punctuation spacing, capitalise sentences
    """
    if not raw_text or not raw_text.strip():
        return ""

    tokens = raw_text.split()

    # Step 1 — consecutive de-dup
    deduped = []
    for tok in tokens:
        if not deduped or tok != deduped[-1]:
            deduped.append(tok)

    # Step 2 — global frequency cap
    seen: dict = {}
    filtered = []
    for tok in deduped:
        seen[tok] = seen.get(tok, 0) + 1
        if seen[tok] <= 3:
            filtered.append(tok)

    if not filtered:
        return ""

    # Step 3 — punctuation & capitalisation
    text = " ".join(filtered)
    text = re.sub(r"\s+([.,])", r"\1", text)
    sentences = [s.strip().capitalize() for s in text.split(". ") if s.strip()]
    formatted = ". ".join(sentences)
    if formatted and not formatted.endswith("."):
        formatted += "."
    return formatted


# ── Cached model loader ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model into memory — this takes ~30 s on first run…")
def load_model(checkpoint_path: str, vocab_path: str):
    device = torch.device("cpu")
    vocab = Vocabulary().load(vocab_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    args = ckpt.get("args", {})

    model = ReportGenerator(
        vocab_size=len(vocab),
        d_model=args.get("d_model", 256),
        backbone=args.get("backbone", "resnet50"),
        pretrained=False,
        nhead=args.get("nhead", 8),
        num_layers=args.get("num_layers", 4),
        dropout=args.get("dropout", 0.2),
        max_len=args.get("max_len", 100),
        pad_idx=vocab.word2idx[PAD_TOKEN],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return {
        "model": model,
        "vocab": vocab,
        "device": device,
        "image_size": args.get("image_size", 224),
        "max_images": args.get("max_images", 2),
        "max_len": args.get("max_len", 100),
    }


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.markdown("---")

        checkpoint_path = st.text_input(
            "Checkpoint path",
            value="checkpoints/best_model.pt",
            help="Path to your trained model checkpoint (.pt file)",
        )
        vocab_path = st.text_input(
            "Vocab path",
            value="vocab.json",
            help="Path to the vocabulary JSON built during training",
        )

        st.markdown("---")
        st.markdown("### About the Model")
        st.markdown("""
- **Encoder:** ResNet-50 (ImageNet-pretrained)
- **Decoder:** Transformer (4 layers, 8 heads)
- **Dataset:** IU Chest X-Ray (~3 300 studies)
- **Training:** Teacher-forcing + cross-entropy
- **Inference:** Greedy decoding with repetition penalty
        """)

        st.markdown("---")
        st.markdown(
            "<div class='disclaimer'>⚠️ <b>Research use only.</b><br>"
            "Not for clinical use.</div>",
            unsafe_allow_html=True,
        )

    return checkpoint_path, vocab_path


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    checkpoint_path, vocab_path = render_sidebar()

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown("# 🩻 Radiology Report Generation")
    st.markdown(
        "Upload a chest X-ray and the model will generate a draft **Findings** "
        "section — the same way a radiologist would describe what they see."
    )
    st.markdown("---")

    # ── Load model ─────────────────────────────────────────────────────────────
    try:
        ctx = load_model(checkpoint_path, vocab_path)
    except FileNotFoundError as e:
        st.error(f"**File not found:** {e}\n\nMake sure `checkpoints/best_model.pt` and `vocab.json` exist in the project root.")
        return
    except Exception as e:
        st.error(f"**Error loading model:** {e}")
        return

    # ── Two-column layout ──────────────────────────────────────────────────────
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("### 📤 Upload X-Ray(s)")

        use_sample = st.checkbox("Use built-in sample X-ray instead", value=False)

        views = []
        view_labels = []

        if use_sample:
            sample_path = "images/sample_xray.jpeg"
            if os.path.exists(sample_path):
                img = Image.open(sample_path).convert("RGB")
                views.append(img)
                view_labels.append("Sample — Frontal View")
                st.success("Sample X-ray loaded ✓")
            else:
                st.warning("Sample image not found at `images/sample_xray.jpeg`.")
        else:
            frontal_file = st.file_uploader(
                "Frontal view (PA / AP) — **required**",
                type=["png", "jpg", "jpeg"],
                key="frontal",
            )
            lateral_file = st.file_uploader(
                "Lateral view — optional (improves accuracy)",
                type=["png", "jpg", "jpeg"],
                key="lateral",
            )

            if frontal_file:
                views.append(Image.open(io.BytesIO(frontal_file.getvalue())).convert("RGB"))
                view_labels.append("Frontal View (PA/AP)")
            if lateral_file:
                views.append(Image.open(io.BytesIO(lateral_file.getvalue())).convert("RGB"))
                view_labels.append("Lateral View")

        # Preview uploaded images
        if views:
            st.markdown("#### Preview")
            preview_cols = st.columns(len(views))
            for i, (img, label) in enumerate(zip(views, view_labels)):
                preview_cols[i].image(img, caption=label, use_container_width=True)
        else:
            st.info("Upload a frontal chest X-ray above to get started.")

    with col_result:
        st.markdown("### 📋 Generated Findings")

        generate_disabled = len(views) == 0
        generate_clicked = st.button(
            "🔍 Generate Findings Report",
            disabled=generate_disabled,
            use_container_width=True,
        )

        # Stats row
        st.markdown("&nbsp;", unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        s1.markdown(
            "<div class='stat-card'><div class='stat-value'>ResNet50</div>"
            "<div class='stat-label'>Vision Encoder</div></div>",
            unsafe_allow_html=True,
        )
        s2.markdown(
            "<div class='stat-card'><div class='stat-value'>4L · 8H</div>"
            "<div class='stat-label'>Transformer Decoder</div></div>",
            unsafe_allow_html=True,
        )
        s3.markdown(
            "<div class='stat-card'><div class='stat-value'>CPU</div>"
            "<div class='stat-label'>Inference Device</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("&nbsp;", unsafe_allow_html=True)

        report_placeholder = st.empty()

        if not views:
            report_placeholder.markdown(
                "<div class='report-card'>"
                "<div class='report-label'>Findings</div>"
                "<span style='color:#475569'>No X-ray uploaded yet. "
                "Upload an image on the left and click the button above.</span>"
                "</div>",
                unsafe_allow_html=True,
            )

        if generate_clicked and views:
            with st.spinner("Analyzing the X-ray and composing findings…"):
                try:
                    # Preprocess
                    transform = build_transform(ctx["image_size"], train=False)
                    tensors = [transform(v) for v in views]

                    # Pad to max_images if only one view
                    while len(tensors) < ctx["max_images"]:
                        tensors.append(tensors[-1].clone())
                    tensors = tensors[: ctx["max_images"]]
                    image_tensor = (
                        torch.stack(tensors, dim=0).unsqueeze(0).to(ctx["device"])
                    )

                    # Generate
                    generated_ids = ctx["model"].generate(
                        image_tensor,
                        ctx["vocab"],
                        max_len=ctx["max_len"],
                        repetition_penalty=1.35,
                        no_repeat_ngram=3,
                        device=ctx["device"],
                    )
                    raw = ctx["vocab"].decode(generated_ids[0].tolist())
                    report = format_report(raw)

                    if report:
                        report_placeholder.markdown(
                            f"<div class='report-card'>"
                            f"<div class='report-label'>Generated Findings</div>"
                            f"{report}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        # Copy-to-clipboard workaround via text area
                        with st.expander("📋 Copy report text"):
                            st.text_area("", value=report, height=120, label_visibility="collapsed")
                    else:
                        report_placeholder.warning(
                            "The model produced an empty report. "
                            "Try a different image or check your checkpoint."
                        )

                except Exception as e:
                    report_placeholder.error(f"**Generation failed:** {e}")

    # ── Footer disclaimer ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div class='disclaimer'>"
        "⚠️ <b>Research use only.</b> "
        "Outputs are generated by a deep learning model and have not been clinically validated. "
        "Do not use for medical decision-making."
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
