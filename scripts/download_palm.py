"""Download and extract PALM subjects from HuggingFace.

Requires a HuggingFace token with access to shampali/PALM-Dataset (gated).
The token is read from $HF_HOME/token (set via `huggingface-cli login`).

Usage:
    # Download subjects 0000-0009
    python scripts/download_palm.py --subjects 0000-0009 --out /data/hohs2/palm/subjects

    # Download specific subjects
    python scripts/download_palm.py --subjects 0000,0005,0042 --out /data/hohs2/palm/subjects

    # Download all 263 subjects (184 GB)
    python scripts/download_palm.py --subjects all --out /data/hohs2/palm/subjects
"""

import argparse
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "shampali/PALM-Dataset"
TOTAL_SUBJECTS = 263


def parse_subjects(spec: str) -> list[str]:
    if spec.lower() == "all":
        return [f"{i:04d}" for i in range(TOTAL_SUBJECTS)]
    parts = spec.split(",")
    result = []
    for p in parts:
        p = p.strip()
        if "-" in p:
            lo, hi = p.split("-", 1)
            for i in range(int(lo), int(hi) + 1):
                result.append(f"{i:04d}")
        else:
            result.append(f"{int(p):04d}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", required=True,
                    help="'all', range '0000-0009', or comma-separated '0000,0005'")
    ap.add_argument("--out", required=True, help="output directory for extracted subjects")
    ap.add_argument("--keep-zips", action="store_true", help="keep zip files after extraction")
    ap.add_argument("--zip-cache", default=None,
                    help="directory to cache downloaded zips (default: <out>/_zips)")
    args = ap.parse_args()

    subjects = parse_subjects(args.subjects)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_dir = Path(args.zip_cache) if args.zip_cache else out_dir / "_zips"
    zip_dir.mkdir(parents=True, exist_ok=True)

    for sid in subjects:
        subj_dir = out_dir / sid
        if subj_dir.exists() and (subj_dir / "poses.npy").exists():
            print(f"[skip] {sid} already extracted")
            continue

        repo_path = f"PALM/XR20A/zips/{sid}.zip"
        print(f"Downloading {sid}.zip ...")
        local_zip = hf_hub_download(
            repo_id=REPO_ID,
            filename=repo_path,
            repo_type="dataset",
            local_dir=str(zip_dir),
        )

        print(f"Extracting {sid} ...")
        with zipfile.ZipFile(local_zip, "r") as zf:
            zf.extractall(out_dir)

        if not args.keep_zips:
            Path(local_zip).unlink(missing_ok=True)

        if (subj_dir / "poses.npy").exists():
            print(f"  {sid} OK")
        else:
            print(f"  WARNING: {sid} extracted but poses.npy not found")

    print(f"\nDone: {len(subjects)} subjects -> {out_dir}")


if __name__ == "__main__":
    main()
