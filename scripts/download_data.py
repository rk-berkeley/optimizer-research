"""
scripts/download_data.py — Download and prepare both datasets.

Downloads:
  1. CIFAR-10 via torchvision, then unpacks into individual JPEG files
     organised as data/cifar10/{train,test}/{class_name}/*.jpg
     (the layout expected by src/standard_data.py).
  2. TinyShakespeare from karpathy/char-rnn via HTTP.

Usage:
    python scripts/download_data.py                # both datasets
    python scripts/download_data.py --dataset cifar
    python scripts/download_data.py --dataset shakespeare

Both steps are idempotent: if the target files already exist they are skipped.

Expected disk usage:
    CIFAR-10 (JPEGs):      ~170 MB
    TinyShakespeare:       ~1 MB
"""

import os
import sys
import argparse
import urllib.request

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# CIFAR-10
# ---------------------------------------------------------------------------

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def _already_have_cifar(cifar_dir: str) -> bool:
    """Return True if at least 10 JPEG files exist per class in train/."""
    for cls in CIFAR10_CLASSES:
        d = os.path.join(cifar_dir, "train", cls)
        if not os.path.isdir(d):
            return False
        jpegs = [f for f in os.listdir(d) if f.endswith(".jpg")]
        if len(jpegs) < 10:
            return False
    return True


def download_cifar10(root: str = None):
    """Download CIFAR-10 via torchvision and export as per-class JPEGs."""
    try:
        import torchvision
        import torchvision.transforms as T
        from PIL import Image
    except ImportError as e:
        print(f"ERROR: {e}\nInstall dependencies: pip install -r requirements.txt")
        sys.exit(1)

    cifar_dir = root or os.path.join(_ROOT, "data", "cifar10")

    if _already_have_cifar(cifar_dir):
        print(f"CIFAR-10 already present at {cifar_dir}/ — skipping.")
        return

    print("Downloading CIFAR-10 via torchvision (this may take a minute)...")
    # Use a temporary pkl directory separate from the JPEG output.
    pkl_dir = os.path.join(_ROOT, "data", "_cifar10_pkl")
    os.makedirs(pkl_dir, exist_ok=True)

    train_ds = torchvision.datasets.CIFAR10(pkl_dir, train=True,  download=True)
    test_ds  = torchvision.datasets.CIFAR10(pkl_dir, train=False, download=True)

    for split_name, ds in [("train", train_ds), ("test", test_ds)]:
        print(f"  Exporting {split_name} split ({len(ds)} images)...")
        # Pre-create class directories.
        for cls in CIFAR10_CLASSES:
            os.makedirs(os.path.join(cifar_dir, split_name, cls), exist_ok=True)
        # Per-class counters for unique filenames.
        counters = {cls: 0 for cls in CIFAR10_CLASSES}
        for img, label in ds:
            cls = CIFAR10_CLASSES[label]
            idx = counters[cls]
            out_path = os.path.join(cifar_dir, split_name, cls,
                                    f"{idx:05d}.jpg")
            if not os.path.exists(out_path):
                # img is a PIL Image from torchvision.
                img.save(out_path, format="JPEG", quality=95)
            counters[cls] = idx + 1

    print(f"  CIFAR-10 JPEGs written to {cifar_dir}/")
    print(f"  (Raw torchvision pkl files are in {pkl_dir}/ — safe to delete)")


# ---------------------------------------------------------------------------
# TinyShakespeare
# ---------------------------------------------------------------------------

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)


def download_shakespeare(root: str = None):
    """Download TinyShakespeare from GitHub."""
    shake_dir  = root or os.path.join(_ROOT, "data", "tinyshakespeare")
    shake_file = os.path.join(shake_dir, "input.txt")

    if os.path.exists(shake_file) and os.path.getsize(shake_file) > 100_000:
        print(f"TinyShakespeare already present at {shake_file} — skipping.")
        return

    os.makedirs(shake_dir, exist_ok=True)
    print(f"Downloading TinyShakespeare from {SHAKESPEARE_URL} ...")
    urllib.request.urlretrieve(SHAKESPEARE_URL, shake_file)
    size_kb = os.path.getsize(shake_file) // 1024
    print(f"  Saved {size_kb} KB to {shake_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["cifar", "shakespeare", "all"],
                   default="all", help="Which dataset to download (default: all).")
    args = p.parse_args()

    if args.dataset in ("cifar", "all"):
        download_cifar10()
    if args.dataset in ("shakespeare", "all"):
        download_shakespeare()

    print("\nAll requested datasets are ready.")


if __name__ == "__main__":
    main()
