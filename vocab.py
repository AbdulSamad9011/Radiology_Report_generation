"""
Word-level vocabulary for radiology report text.

We build a vocab from the training corpus rather than using a pretrained
tokenizer, so the whole project runs without needing to download anything
from the Hugging Face Hub — only the corpus you already have.
"""

import json
import re
from collections import Counter

PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"


class Vocabulary:
    def __init__(self, min_freq: int = 3):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}

    @staticmethod
    def tokenize(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z0-9.,\s]", " ", text)
        return text.split()

    def build(self, reports):
        """reports: iterable of raw report strings (training split only!)."""
        counter = Counter()
        for r in reports:
            counter.update(self.tokenize(r))

        specials = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]
        words = specials + sorted(w for w, c in counter.items() if c >= self.min_freq)
        self.word2idx = {w: i for i, w in enumerate(words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        return self

    def encode(self, text: str, max_len: int = None):
        tokens = [START_TOKEN] + self.tokenize(text) + [END_TOKEN]
        unk_idx = self.word2idx[UNK_TOKEN]
        ids = [self.word2idx.get(t, unk_idx) for t in tokens]
        if max_len is not None:
            ids = ids[:max_len]
            ids = ids + [self.word2idx[PAD_TOKEN]] * (max_len - len(ids))
        return ids

    def decode(self, ids, stop_at_end: bool = True):
        words = []
        for i in ids:
            w = self.idx2word.get(int(i), UNK_TOKEN)
            if w in (START_TOKEN, PAD_TOKEN):
                continue
            if w == END_TOKEN:
                if stop_at_end:
                    break
                continue
            words.append(w)
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.word2idx, f)

    def load(self, path: str):
        with open(path) as f:
            self.word2idx = json.load(f)
        self.idx2word = {int(i): w for w, i in self.word2idx.items()}
        return self
