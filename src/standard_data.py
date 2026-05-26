"""
standard_data.py — Dataset loaders for CIFAR-10 and TinyShakespeare.

Both datasets are loaded from disk; see download instructions below.

CIFAR-10:
  The standard 60k-image benchmark (50k train / 10k test) across 10 classes.
  This module expects the dataset as individual JPEG files on disk, organized
  as data/cifar10/{train,test}/{class_name}/*.jpg. Download:

      # Option A: torchvision (auto-download, used by train_cifar.py)
      import torchvision
      torchvision.datasets.CIFAR10("data/cifar10", download=True)

      # Option B: raw JPEGs (used by this module)
      git clone https://github.com/YoongiKim/CIFAR-10-images data/cifar10

  The JPEG layout matches the YoongiKim repo and the torchvision auto-extract.
  Images are loaded once per process and cached in memory (~150 MB for full
  train split; negligible startup overhead on repeat calls).

TinyShakespeare:
  1,115,394 characters of Shakespeare plays. Vocabulary: 65 printable chars.
  Standard 90/10 train/val split following nanoGPT convention.
  Download:

      mkdir -p data/tinyshakespeare
      curl -o data/tinyshakespeare/input.txt \\
        https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

  Or run reproduce.sh, which downloads it automatically.
"""

import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


# ---------------------------------------------------------------------------
# Paths — override via environment variables for custom data locations.
# ---------------------------------------------------------------------------

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CIFAR10_DIR = os.environ.get(
    "CIFAR10_DIR", os.path.join(_ROOT, "data", "cifar10")
)
TINY_SHAKESPEARE = os.environ.get(
    "TINY_SHAKESPEARE", os.path.join(_ROOT, "data", "tinyshakespeare", "input.txt")
)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


# ---------------------------------------------------------------------------
# CIFAR-10
# ---------------------------------------------------------------------------

class CIFAR10Dataset(Dataset):
    """CIFAR-10 from individual JPEG files.

    On first call per (split, subset) combination the images are read from
    disk and cached as a uint8 tensor in memory. Subsequent calls reuse the
    cache, so multiple DataLoader workers share one copy.

    Args:
        split: "train" or "test".
        subset: if given, randomly subsample this many images (for fast runs).
        augment: if True, apply random horizontal flip + random 4-pixel crop.
        seed: random seed for subsampling.
    """

    _cache: dict = {}

    def __init__(self, split: str = "train", subset=None,
                 augment: bool = False, seed: int = 0):
        assert split in ("train", "test"), f"Unknown split: {split!r}"
        key = (split, subset)
        if key not in self._cache:
            self._cache[key] = self._load(split, subset, seed)
        self.images, self.labels = self._cache[key]
        self.augment = augment
        self.mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
        self.std  = torch.tensor(CIFAR10_STD).view(3, 1, 1)

    @staticmethod
    def _load(split: str, subset, seed: int):
        all_paths, all_labels = [], []
        for cls_idx, cls in enumerate(CIFAR10_CLASSES):
            d = os.path.join(CIFAR10_DIR, split, cls)
            paths = sorted(glob.glob(os.path.join(d, "*.jpg")))
            all_paths.extend(paths)
            all_labels.extend([cls_idx] * len(paths))
        if not all_paths:
            raise FileNotFoundError(
                f"No JPEG images found under {CIFAR10_DIR}/{split}/. "
                "See the docstring in standard_data.py for download instructions."
            )
        all_labels = np.array(all_labels, dtype=np.int64)
        if subset is not None and subset < len(all_paths):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(all_paths), size=subset, replace=False)
            all_paths = [all_paths[i] for i in idx]
            all_labels = all_labels[idx]
        N = len(all_paths)
        imgs = np.empty((N, 3, 32, 32), dtype=np.uint8)
        for i, p in enumerate(all_paths):
            arr = np.array(Image.open(p).convert("RGB"))  # (32, 32, 3)
            imgs[i] = arr.transpose(2, 0, 1)
        return torch.from_numpy(imgs), torch.from_numpy(all_labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i].float() / 255.0  # (3, 32, 32)
        if self.augment:
            if torch.rand(1).item() < 0.5:
                img = img.flip(-1)           # random horizontal flip
            pad = 4
            padded = torch.nn.functional.pad(
                img.unsqueeze(0), (pad, pad, pad, pad), mode="reflect"
            ).squeeze(0)
            top  = int(torch.randint(0, 2 * pad + 1, (1,)).item())
            left = int(torch.randint(0, 2 * pad + 1, (1,)).item())
            img = padded[:, top : top + 32, left : left + 32]
        img = (img - self.mean) / self.std
        return img, int(self.labels[i].item())


# ---------------------------------------------------------------------------
# TinyShakespeare
# ---------------------------------------------------------------------------

def load_shakespeare():
    """Load and split TinyShakespeare into train/val token sequences.

    Returns:
        (train_ids, val_ids, vocab_size, stoi, itos)
        where stoi and itos are char↔int mappings.
    """
    if not os.path.exists(TINY_SHAKESPEARE):
        raise FileNotFoundError(
            f"TinyShakespeare not found at {TINY_SHAKESPEARE}. "
            "Run reproduce.sh or: "
            "curl -o data/tinyshakespeare/input.txt "
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
            "data/tinyshakespeare/input.txt"
        )
    with open(TINY_SHAKESPEARE, "r") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(ids))        # 90/10 split (nanoGPT convention)
    return ids[:split], ids[split:], len(chars), stoi, itos


class ShakespeareDataset(Dataset):
    """Sample random length-`seq_len` windows from the TinyShakespeare corpus.

    Args:
        n_samples: number of random windows to pre-sample (deterministic given seed).
        seq_len: context length in characters.
        split: "train" or "val".
        seed: random seed for window selection.
    """

    _cache: dict = {}

    def __init__(self, n_samples: int = 1500, seq_len: int = 128,
                 split: str = "train", seed: int = 0):
        if "all" not in self._cache:
            self._cache["all"] = load_shakespeare()
        train_ids, val_ids, vocab, stoi, itos = self._cache["all"]
        self.vocab_size = vocab
        self.stoi = stoi
        self.itos = itos
        self.seq_len = seq_len
        self.data_ids = train_ids if split == "train" else val_ids
        g = torch.Generator().manual_seed(seed + (0 if split == "train" else 999))
        max_start = len(self.data_ids) - seq_len - 1
        self.starts = torch.randint(0, max_start, (n_samples,), generator=g)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i].item())
        chunk = self.data_ids[s : s + self.seq_len + 1]
        return chunk[:-1].clone(), chunk[1:].clone()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Module-level constant (used by train_standard.py for model construction)
# ---------------------------------------------------------------------------

# Lazy: avoid crashing at import time when data hasn't been downloaded yet.
# train_standard.py accesses this at the point of model construction, at which
# time the data must be present.  Run scripts/download_data.py first.
try:
    _, _, SHAKESPEARE_VOCAB_SIZE, _, _ = load_shakespeare()
except FileNotFoundError:
    SHAKESPEARE_VOCAB_SIZE = 65   # correct for TinyShakespeare; set properly on first use
