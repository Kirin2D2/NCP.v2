"""
Download ImageNet synset tarballs (ImageNet-21k "winter21" release) and extract every JPEG into a
flat folder per synset, named {wnid}_{index:05d}.jpeg in tarball order.

The experiment splits are defined on sorted filename order, so keep this naming.
Downloading from image-net.org requires an account that has accepted the ImageNet terms of access.

Tasks
-----
  carton_dugong  n02971356 (carton), n02074367 (dugong)  -> data/images/carton_dugong/original/<wnid>/
  crate_packet   n03127925 (crate),  n03871628 (packet)  -> data/images/crate_packet/original/<wnid>/
  basketball     n02802426 (basketball)                  -> data/images/imagenet_430_binary/basketball/
                 (negatives: data/images/download_random_images.py)

Note: the paper's crate/packet runs used images with ILSVRC-style names (n03127925_97.JPEG),
so a fresh download may not reproduce the exact crate/packet split.

Usage
-----
  python data/images/download_binary_dataset.py --task carton_dugong
  python data/images/download_binary_dataset.py --wnids n02971356 --out_dir /some/dir
"""

import argparse
import io
import ssl
import tarfile
import urllib.request
from pathlib import Path

import certifi

BASE_URL = "https://image-net.org/data/winter21_whole"  # per-synset tarballs
IMAGES_DIR = Path(__file__).resolve().parent

TASKS = {
    "carton_dugong": (["n02971356", "n02074367"], IMAGES_DIR / "carton_dugong" / "original"),
    "crate_packet":  (["n03127925", "n03871628"], IMAGES_DIR / "crate_packet" / "original"),
    "basketball":    (["n02802426"], IMAGES_DIR / "imagenet_430_binary"),
}


def _install_opener():
    ctx = ssl.create_default_context(cafile=certifi.where())
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (wget-like)")]
    urllib.request.install_opener(opener)


def download_synset(wnid: str, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_url = f"{BASE_URL}/{wnid}.tar"
    tar_bytes = urllib.request.urlopen(tar_url, timeout=60).read()
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tar:
        members = [m for m in tar.getmembers() if m.name.lower().endswith((".jpeg", ".jpg"))]
        if not members:
            raise RuntimeError(f"No JPEGs found inside {wnid}.tar (URL: {tar_url})")
        kept = 0
        for m in members:
            f = tar.extractfile(m)
            if f is None:
                continue
            with open(dest_dir / f"{wnid}_{kept:05d}.jpeg", "wb") as o:
                o.write(f.read())
            kept += 1
    print(f"[ok] {wnid}: extracted {kept} images to {dest_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=sorted(TASKS), help="Download the synsets for a paper task.")
    p.add_argument("--wnids", nargs="+", help="Explicit synset IDs (overrides --task synsets).")
    p.add_argument("--out_dir", type=Path, help="Parent directory; each synset goes to <out_dir>/<wnid>/.")
    args = p.parse_args()

    if not args.task and not args.wnids:
        p.error("pass --task or --wnids")
    wnids, out_dir = TASKS[args.task] if args.task else (args.wnids, None)
    wnids = args.wnids or wnids
    out_dir = args.out_dir or out_dir
    if out_dir is None:
        p.error("--out_dir is required with --wnids")

    _install_opener()
    for wnid in wnids:
        # The basketball ImageFolder expects the positive class folder to be named 'basketball'.
        dest = out_dir / ("basketball" if args.task == "basketball" and not args.wnids else wnid)
        try:
            download_synset(wnid, dest)
        except Exception as e:
            print(f"[fail] {wnid}: {e}")


if __name__ == "__main__":
    main()
