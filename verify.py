"""verify.py

Check that the local machine has the full Tahoe-100M dataset before training.

The raw data (merge_data/, metadata/) is too large for git and is git-ignored,
so this script is the gatekeeper: it confirms the expected files and their
counts are present. Exits non-zero if anything critical is missing.

Usage:
    python verify.py                 # checks the repo directory
    python verify.py --root /path    # checks a specific data root
"""

import argparse
import os
import sys
import json

# (label, path, critical)  — critical = missing file fails the check
EXPECTED_FILES = [
    ("Gene vocabulary", "metadata/gene_vocabulary.json", True),
    ("Gene metadata", "metadata/gene_metadata.parquet", True),
    ("Drug metadata", "metadata/drug_metadata.parquet", True),
    ("Sample metadata", "metadata/sample_metadata.parquet", True),
    ("Obs metadata", "metadata/obs_metadata.parquet", True),
    ("Chemical info (drug JSON)", "metadata/Chemcial_info/drug_metadata.json", True),
    ("Chemical info (cluster labels)", "metadata/Chemcial_info/drug_cluster_labels_k9.parquet", True),
    ("Chemical info (MoLFormer pooled emb)", "metadata/Chemcial_info/molformer_embeddings.csv", False),
]

EXPECTED_WELLS = 1344  # merge_data/smp_*.parquet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    root = args.root
    failures = []
    print(f"verify: checking dataset root: {root}\n")

    # --- merge_data well parquets ----------------------------------------
    merge_dir = os.path.join(root, "merge_data")
    n_wells = 0
    if os.path.isdir(merge_dir):
        n_wells = len([
            f for f in os.listdir(merge_dir)
            if f.startswith("smp_") and f.endswith(".parquet")
        ])
    ok_wells = n_wells == EXPECTED_WELLS
    print(f"[{'PASS' if ok_wells else 'FAIL'}] merge_data/: {n_wells}/{EXPECTED_WELLS} sample parquets")
    if not ok_wells:
        failures.append("merge_data wells")

    # --- metadata files ---------------------------------------------------
    for label, rel, critical in EXPECTED_FILES:
        path = os.path.join(root, rel)
        ok = os.path.isfile(path) and os.path.getsize(path) > 0
        size = os.path.getsize(path) if os.path.isfile(path) else 0
        print(f"[{'PASS' if ok else 'FAIL'}] {rel} ({size/1e6:.1f} MB)" if ok
              else f"[{'PASS' if ok else 'FAIL'}] {rel} (MISSING)")
        if not ok and critical:
            failures.append(rel)

    # --- gene vocabulary sanity ------------------------------------------
    vocab_path = os.path.join(root, "metadata/gene_vocabulary.json")
    n_genes = None
    if os.path.isfile(vocab_path):
        try:
            with open(vocab_path) as f:
                n_genes = len(json.load(f))
        except Exception:
            n_genes = -1
    ok_vocab = n_genes and n_genes > 60000
    print(f"[{'PASS' if ok_vocab else 'FAIL'}] gene vocabulary size: {n_genes} (expected >60,000)")
    if not ok_vocab:
        failures.append("gene vocabulary")

    # --- checkpoints dir (optional) ---------------------------------------
    ckpt_dir = os.path.join(root, "checkpoints")
    if os.path.isdir(ckpt_dir):
        print(f"[PASS] checkpoints/: {len(os.listdir(ckpt_dir))} file(s)")
    else:
        print("[INFO] checkpoints/: absent (created at first train)")

    print()
    if failures:
        print(f"VERIFY FAILED — missing/critical: {', '.join(failures)}")
        sys.exit(1)
    print("VERIFY PASSED — dataset is present and ready for training.")


if __name__ == "__main__":
    main()