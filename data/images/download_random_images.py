"""
Build the negative class for the basketball experiment: a few images from each of many random
ImageNet-1k synsets (excluding basketball, n02802426), saved as

  data/images/imagenet_430_binary/not_basketball/<wnid>/<original filename>

torchvision's ImageFolder walks subdirectories, so all of these become class 'not_basketball'.
Downloading from image-net.org requires an account that has accepted the ImageNet terms of access.
The exact negative images used in the paper were not recorded; with the same seed this
reproduces the synset sample, not necessarily the identical files.

Usage
-----
  python data/images/download_random_images.py
  python data/images/download_random_images.py --num_classes 110 --images_per_class 10 --seed 42
"""

import argparse
import os
import random
import tarfile
import urllib.request
from pathlib import Path

BASKETBALL_SYNSET = "n02802426"
BASE_URL = "https://image-net.org/data/winter21_whole"
IMAGES_DIR = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synset_list", type=Path, default=IMAGES_DIR / "imagenet_synsets.txt",
                   help="Text file with one ImageNet-1k synset ID per line.")
    p.add_argument("--out_dir", type=Path, default=IMAGES_DIR / "imagenet_430_binary" / "not_basketball")
    p.add_argument("--num_classes", type=int, default=110)
    p.add_argument("--images_per_class", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    with open(args.synset_list) as f:
        all_synsets = [line.strip() for line in f if line.strip() and line.strip() != BASKETBALL_SYNSET]

    random.seed(args.seed)
    chosen = random.sample(all_synsets, args.num_classes)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for synset in chosen:
        print(f"Processing synset {synset}...")
        tar_path = args.out_dir / f"{synset}.tar"
        try:
            urllib.request.urlretrieve(f"{BASE_URL}/{synset}.tar", tar_path)
            with tarfile.open(tar_path) as tar:
                members = [m for m in tar.getmembers() if m.name.lower().endswith((".jpeg", ".jpg"))]
                if len(members) < args.images_per_class:
                    print(f"  Skipping {synset}: only {len(members)} images")
                    continue
                dest_dir = args.out_dir / synset
                dest_dir.mkdir(parents=True, exist_ok=True)
                for m in members[:args.images_per_class]:
                    tar.extract(m, path=dest_dir)
        except Exception as e:
            print(f"  Failed to process {synset}: {e}")
        finally:
            if tar_path.exists():
                os.remove(tar_path)


if __name__ == "__main__":
    main()
