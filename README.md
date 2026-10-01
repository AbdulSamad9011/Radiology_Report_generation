# Radiology Report Generation

A vision-language model that generates free-text radiology findings from chest X-ray images. Given a frontal (and optionally lateral) view, the model produces a draft Findings section describing what it observes — cardiac size, lung fields, pleural space, bony structures, and so on.

Built with PyTorch from scratch. Served with Streamlit.

---

## Demo

Upload a chest X-ray and click **Generate Findings Report**:

![Sample chest X-ray](images/sample_xray.jpeg)

---

## Architecture

```
Frontal X-ray  ──┐
                 ├──► ResNet-50 Encoder
Lateral X-ray  ──┘         │
                    Spatial feature map (7×7×2048)
                    Projected → 256-d, flattened to sequence
                            │
                            ▼
                  Transformer Decoder
                  4 layers · 8 attention heads · d_model=256
                  Cross-attends to image memory at each step
                            │
                            ▼
                   Generated findings text
```

The encoder extracts spatial features from each X-ray view. These are projected and concatenated into a single memory sequence. The Transformer decoder attends to this memory autoregressively, generating one token at a time until it produces an `<end>` token.

Multi-view input is handled by concatenating the feature sequences from each view — studies with only one image automatically duplicate the single view to keep the input shape consistent.

---

## Project structure

```
.
├── app.py               # Streamlit web interface
├── model.py             # CNNEncoder + ReportDecoder + ReportGenerator
├── dataset.py           # IU X-Ray dataset loader
├── vocab.py             # Word-level vocabulary builder / encoder / decoder
├── train.py             # Training loop (teacher forcing, cross-entropy)
├── generate.py          # Greedy decoding + BLEU / ROUGE-L evaluation
├── checkpoints/
│   └── best_model.pt    # Best checkpoint saved by validation loss
├── images/
│   └── sample_xray.jpeg
├── vocab.json           # Auto-generated during training
├── requirements.txt
├── run_app.sh           # One-command launcher
└── .streamlit/
    └── config.toml
```

---

## Setup

```bash
git clone https://github.com/your-username/Radiology_Report_generation.git
cd Radiology_Report_generation

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Dataset

Trained on the **IU Chest X-Ray** (Indiana University / Open-i) collection, using the preprocessed annotation format from the [R2Gen paper](https://arxiv.org/abs/2010.16056) (Chen et al., EMNLP 2020).

The annotation file structure:

```json
{
  "train": [
    {
      "id": "CXR1000_IM-0003",
      "report": "the heart is normal in size. the lungs are clear.",
      "image_path": ["CXR1000_IM-0003-1001.png", "CXR1000_IM-0003-2001.png"]
    }
  ],
  "val": [...],
  "test": [...]
}
```

This preprocessed split (`annotation.json` + images) is mirrored alongside several open-source report-generation repositories since it's the standard benchmark. Place images in a folder and pass its path to `--image_dir`.

---

## Training

```bash
python train.py \
    --annotation_path annotation.json \
    --image_dir path/to/images/ \
    --epochs 40 \
    --batch_size 8 \
    --backbone resnet50
```

On the first run, `vocab.json` is built from the training split and saved to the project root. The best checkpoint (lowest validation loss) is saved to `checkpoints/best_model.pt`.

**Notes:**
- The ResNet-50 backbone downloads ImageNet weights on first use — requires internet. Pass `--no_pretrained` to skip.
- Horizontal flip augmentation is deliberately excluded — left/right orientation is clinically meaningful in chest X-rays.
- Studies with only one view are handled automatically (the view is duplicated to match `--max_images`).

---

## Evaluation

```bash
python generate.py \
    --annotation_path annotation.json \
    --image_dir path/to/images/ \
    --checkpoint checkpoints/best_model.pt \
    --split test
```

Decodes the full test split and reports **BLEU-1, BLEU-4**, and **ROUGE-L** alongside a handful of reference vs. generated example pairs.

Expected score range on IU X-Ray: BLEU-4 ~0.08–0.15, ROUGE-L ~0.30–0.38. Free-text generation is a significantly harder metric target than classification — published results on this dataset are in the same range.

---

## Running the app

```bash
bash run_app.sh
```

Or directly:

```bash
streamlit run app.py
```

Open **http://localhost:8501**. Upload a frontal chest X-ray (JPEG or PNG), optionally add a lateral view, and click **Generate Findings Report**. A sample image is bundled if you want to test without your own data.

---

## Model configuration

| Parameter | Default | Description |
|---|---|---|
| `backbone` | `resnet50` | Vision encoder. Also supports `resnet18`. |
| `d_model` | `256` | Embedding dimension throughout encoder + decoder |
| `nhead` | `8` | Transformer attention heads |
| `num_layers` | `4` | Transformer decoder layers |
| `max_len` | `100` | Maximum generated report length (tokens) |
| `max_images` | `2` | Views per study (frontal + lateral) |
| `image_size` | `224` | Input resolution after resize |

All hyperparameters are saved inside the checkpoint and restored automatically at inference time.

---

## Disclaimer

This project is for research purposes only. Generated reports have not been clinically validated and must not be used for diagnostic or treatment decisions.

---

## License

MIT
