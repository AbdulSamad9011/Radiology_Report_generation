# Radiology Report Generation (Vision-Language)

A from-scratch vision-language model: given one or two chest X-ray views
(frontal + lateral), it generates a free-text findings report. This is the
"AI looks at an image and writes a report" family of problems — a step up
from classification/segmentation into multimodal generation.

## Why this project

- **Two skill trees at once.** A CNN vision encoder plus a Transformer
  language decoder, trained jointly — this is the core recipe behind every
  modern vision-language model, just at a scale you can actually train and
  understand end to end.
- **Different eval methodology than a classifier.** You're scored with
  NLG metrics (BLEU, ROUGE-L) instead of accuracy/Dice, which is worth
  understanding if you want multimodal roles.
- **A demo that reads, not just labels.** "Upload an X-ray, get a draft
  report back" is a much stronger interview story than a bar chart of
  class probabilities.
- **No gated/credentialed data required.** Built around the IU X-Ray
  dataset (public, no PhysioNet-style credentialing, unlike MIMIC-CXR).

## Architecture

```
Frontal + lateral X-ray
        │
        ▼
  CNNEncoder (ResNet, ImageNet-pretrained)
        │  spatial feature maps, projected to d_model,
        │  flattened into a sequence of "memory" tokens
        ▼
  ReportDecoder (Transformer decoder, cross-attends to memory)
        │  autoregressive, causal self-attention + cross-attention
        ▼
   Generated findings text
```

```
radreport/
├── vocab.py       # word-level vocabulary built from your training reports
├── dataset.py      # IU X-Ray style dataset loader (frontal+lateral pairs)
├── model.py         # CNNEncoder + ReportDecoder + ReportGenerator wrapper
├── train.py           # training loop (teacher forcing, cross-entropy)
├── generate.py          # greedy decoding + BLEU / ROUGE-L evaluation
├── app.py                 # Streamlit demo
└── requirements.txt
```

Every module has been smoke-tested here (forward-pass shape checks, a real
multi-epoch training run on synthetic data, checkpoint save/load, greedy
generation terminating correctly, and the Streamlit app booting) — you're
starting from code that runs, not a sketch.

## 1. Setup

```bash
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## 2. Get the dataset

This targets the **IU Chest X-Ray** collection (Indiana University /
Open-i), preprocessed into the annotation format popularized by the R2Gen
paper (Chen et al., *"Generating Radiology Reports via Memory-driven
Transformer"*, EMNLP 2020):

```json
{
  "train": [
    {"id": "CXR1000_IM-0003",
     "report": "the heart is normal in size...",
     "image_path": ["CXR1000_IM-0003-1001.png", "CXR1000_IM-0003-2001.png"]},
    ...
  ],
  "val": [...],
  "test": [...]
}
```

Search for **"IU X-Ray R2Gen annotation.json"** — this preprocessed
split (images + annotation.json) is widely mirrored alongside several
open-source report-generation repos on GitHub, since it's the standard
benchmark split for this task. Put the images in one folder and point
`--image_dir` at it, and pass the annotation file as `--annotation_path`.

If you instead start from the *raw* Open-i XML reports, you'll need to pull
the `FINDINGS`/`IMPRESSION` text and the paired image filenames out of each
XML file yourself — the exact tag names vary slightly between raw download
batches, so inspect one sample file before writing that parser.

## 3. Train

```bash
python train.py --annotation_path annotation.json --image_dir images/ \
    --epochs 40 --batch_size 8 --backbone resnet50
```

First run builds a word-level vocabulary from the **training split only**
(saved to `vocab.json`) and trains with teacher forcing + cross-entropy.
Best checkpoint (by validation loss) is saved to `checkpoints/best_model.pt`.

Notes:
- The ResNet backbone downloads ImageNet-pretrained weights by default
  (needs internet the first time). Pass `--no_pretrained` to train from
  scratch instead — works, but expect slower convergence and weaker
  results given how small IU X-Ray is (~3,300 studies).
- `--max_images 2` assumes frontal+lateral; studies with only one image
  are handled automatically (the view is duplicated).
- Chest X-rays are **not** horizontally flipped during augmentation —
  left/right laterality is clinically meaningful, unlike in most natural-
  image pipelines.

## 4. Evaluate

```bash
python generate.py --annotation_path annotation.json --image_dir images/ \
    --checkpoint checkpoints/best_model.pt --split test
```

Greedy-decodes reports for the split, prints a handful of side-by-side
reference/generated examples, and reports **BLEU-1/BLEU-4** and
**ROUGE-L**. Treat early-training BLEU-4 scores in the 0.05–0.15 range as
normal for this task — published papers on this exact dataset report
roughly this order of magnitude, since free-text generation is a much
harder metric target than classification accuracy.

## 5. Run the interactive demo

```bash
streamlit run app.py
```

Upload a frontal (and optionally lateral) X-ray and generate a draft
report live.

## Extending it further (good "next steps" to mention in an interview)

- Add a **cross-entropy + reinforcement learning (CIDEr-optimized) fine-tuning
  stage**, the approach several published report-generation papers use to
  close the gap between teacher-forced training and free-running generation.
- Swap greedy decoding for **beam search** in `generate.py`.
- Add a **relational memory / knowledge-graph module** that conditions
  generation on common findings co-occurrence patterns (the "R2Gen" idea).
- Try a **frozen pretrained vision backbone (e.g. a CLIP-style image
  encoder) + a small trainable decoder**, and compare sample-efficiency
  against training the CNN from scratch.
- Report **per-abnormality precision/recall** (e.g. via CheXpert's rule-
  based labeler on the generated text) in addition to BLEU/ROUGE — clinical
  correctness matters more than fluency for this task, and pointing that
  out unprompted is a good signal in an interview.

## Putting it on your resume / portfolio

> Built an encoder-decoder vision-language model (ResNet + Transformer
> decoder) that generates chest X-ray findings reports from frontal/lateral
> views, achieving X.XX BLEU-4 / X.XX ROUGE-L on held-out studies; shipped
> as an interactive Streamlit demo.

Fill in your real numbers after training on the actual dataset.

## Disclaimer

This is a research/portfolio project, not a medical device. Generated
reports are not validated for, and must not be used for, clinical
decision-making.
