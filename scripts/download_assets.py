"""
Download released CNP assets (pruned checkpoints and the paper's stats.pt files) into results/.

Assets are published as release files; fill in RELEASE_BASE_URL and ASSETS when they are uploaded.
Checkpoints load on CPU or GPU with ncp.checkpoint.load_pruned_checkpoint.

Usage
-----
  python scripts/download_assets.py --list
  python scripts/download_assets.py carton_ncp_seed0 carton_vanilla_seed0
  python scripts/download_assets.py --all
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from ncp.paths import RESULTS_ROOT

# e.g. "https://github.com/KirinDanek/NCP.v2/releases/download/v1.0.0"
RELEASE_BASE_URL = None

# name -> (file name in the release, destination relative to RESULTS_ROOT, sha256 or None)
ASSETS = {
    # "carton_ncp_seed0":     ("carton_ncp_seed0.pt",     "watermark_experiment/carton/ncp/seed0/ncp.pt", None),
    # "carton_vanilla_seed0": ("carton_vanilla_seed0.pt", "watermark_experiment/carton/vanilla/seed0/vanilla.pt", None),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, force: bool = False) -> Path:
    remote, dest_rel, sha256 = ASSETS[name]
    dest = RESULTS_ROOT / dest_rel
    if dest.exists() and not force:
        print(f"[skip] {name}: {dest} exists")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RELEASE_BASE_URL}/{remote}"
    print(f"[get]  {name}: {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    if sha256 is not None and _sha256(tmp) != sha256:
        tmp.unlink()
        raise RuntimeError(f"checksum mismatch for {name}")
    tmp.replace(dest)
    print(f"[ok]   {dest}")
    return dest


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", help="Asset names to download (see --list).")
    p.add_argument("--all", action="store_true", help="Download every asset.")
    p.add_argument("--list", action="store_true", help="List available assets.")
    p.add_argument("--force", action="store_true", help="Re-download existing files.")
    args = p.parse_args()

    if RELEASE_BASE_URL is None or not ASSETS:
        sys.exit("No release assets have been published yet (RELEASE_BASE_URL / ASSETS are empty).")

    if args.list or not (args.names or args.all):
        for name, (_, dest, _) in ASSETS.items():
            print(f"{name:30s} -> results/{dest}")
        return

    for name in (ASSETS if args.all else args.names):
        if name not in ASSETS:
            sys.exit(f"Unknown asset {name!r}; see --list")
        download(name, force=args.force)


if __name__ == "__main__":
    main()
