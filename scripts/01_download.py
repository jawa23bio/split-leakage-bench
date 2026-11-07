#!/usr/bin/env python
"""Download ATF3 K562 peaks from ENCODE and the hg38 genome from UCSC.

Default experiment: ENCSR028UIU (ATF3 ChIP-seq in K562, used in the ENCODE-DREAM
challenge). The script asks the ENCODE API for that experiment, picks a released
GRCh38 IDR-thresholded narrowPeak file, and prints which one it chose.

Usage:
    python scripts/01_download.py --out data
    python scripts/01_download.py --out data --peaks-accession ENCFFxxxxxx   # pick a file yourself
    python scripts/01_download.py --out data --skip-genome                   # genome already present
"""

import argparse
import gzip
import shutil
import sys
from pathlib import Path

import requests

ENCODE = "https://www.encodeproject.org"
HG38 = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
# ENCODE's GRCh38 exclusion list (blacklist v2). Optional: --blacklist
BLACKLIST_ACCESSION = "ENCFF356LFX"
HEADERS = {"Accept": "application/json"}
OUTPUT_PREFERENCE = [
    "conservative IDR thresholded peaks",
    "optimal IDR thresholded peaks",
    "IDR thresholded peaks",
]


def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}\n  -> {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def pick_peak_file(experiment: str):
    r = requests.get(f"{ENCODE}/experiments/{experiment}/?format=json", headers=HEADERS, timeout=60)
    r.raise_for_status()
    files = r.json().get("files", [])
    candidates = [
        f for f in files
        if f.get("file_type") == "bed narrowPeak"
        and f.get("assembly") == "GRCh38"
        and f.get("status") == "released"
        and f.get("output_type") in OUTPUT_PREFERENCE
    ]
    if not candidates:
        sys.exit(f"No released GRCh38 IDR narrowPeak file found for {experiment}. "
                 "Look at the experiment page and pass --peaks-accession.")
    candidates.sort(key=lambda f: (not f.get("preferred_default", False),
                                   OUTPUT_PREFERENCE.index(f["output_type"])))
    print("candidate peak files:")
    for f in candidates:
        print(f"  {f['accession']}  {f['output_type']}  preferred_default={f.get('preferred_default', False)}")
    return candidates[0]["accession"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--experiment", default="ENCSR028UIU")
    ap.add_argument("--peaks-accession", help="use this ENCFF file instead of picking one")
    ap.add_argument("--skip-genome", action="store_true")
    ap.add_argument("--blacklist", action="store_true", help="also download the ENCODE GRCh38 exclusion list")
    args = ap.parse_args()
    out = Path(args.out)

    acc = args.peaks_accession or pick_peak_file(args.experiment)
    print(f"using peaks: {acc}")
    download(f"{ENCODE}/files/{acc}/@@download/{acc}.bed.gz", out / "peaks.narrowPeak.gz")

    if args.blacklist:
        download(f"{ENCODE}/files/{BLACKLIST_ACCESSION}/@@download/{BLACKLIST_ACCESSION}.bed.gz",
                 out / "blacklist.bed.gz")

    if not args.skip_genome:
        gz = out / "hg38.fa.gz"
        download(HG38, gz)
        print("decompressing (about 3 GB; keeps UCSC soft-masking, lowercase = repeat)")
        with gzip.open(gz, "rb") as fin, open(out / "hg38.fa", "wb") as fout:
            shutil.copyfileobj(fin, fout)
        gz.unlink()

    (out / "SOURCE.txt").write_text(f"experiment={args.experiment}\npeaks={acc}\ngenome=UCSC hg38\n")
    print("done")


if __name__ == "__main__":
    main()
