# Run Guide — Tahoe-100M Virtual Cell Model

How to prepare data, run a smoke test, launch full training, and evaluate the model.

---

## 1. Prerequisites

- **Python 3.11+** (code has been exercised under 3.11 / 3.12)
- **GPU recommended** for full training (CUDA). CPU works but is very slow at full scale.
- **Disk**: `merge_data/` (~330 GB, 1,344 files) must already be present locally; keep headroom for checkpoints.

### Install dependencies

```bash
# PyTorch (pick the index for your CUDA version; omit --index-url for CPU-only)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Everything else
pip install pandas numpy pyarrow anndata scanpy scipy scikit-learn umap-learn transformers pyyaml
```

The MoLFormer checkpoint (`ibm-research/MoLFormer-XL-both-10pct`) is loaded lazily on first use and cached by HuggingFace — no manual download step.

---

## 2. Fix config paths (IMPORTANT — do this first)

`config.yaml` currently points at a macOS external drive (`/Volumes/T7/mtp/Tahoe-100M/...`).
Update the `paths:` block to your local layout (Windows example below):

```yaml
paths:
  merge_data: H:/mtp/Tahoe-100M/merge_data
  gene_vocab: H:/mtp/Tahoe-100M/metadata/gene_vocabulary.json
  drug_json: H:/mtp/Tahoe-100M/metadata/Chemcial_info/drug_metadata.json
  cluster_labels: H:/mtp/Tahoe-100M/metadata/Chemcial_info/drug_cluster_labels_k9.parquet
  sample_metadata: H:/mtp/Tahoe-100M/metadata/sample_metadata.parquet
  checkpoint_dir: H:/mtp/Tahoe-100M/checkpoints
```

---

## 3. Verify the dataset

```bash
python verify.py
```

Expect `VERIFY PASSED — dataset is present and ready for training.`
This checks the 1,344 `smp_*.parquet` wells, critical metadata tables, and the 62k+ gene vocabulary.
It exits non-zero if anything critical is missing.

---

## 4. Smoke test (do this first)

```bash
python train.py --smoke
```

What `--smoke` does (overrides from `config.yaml` → `smoke:`):
- Tiny model: `emb_dim=128`, 1 layer, 2 heads
- Capped data: 400 cells/well, 2,000 genes, 32 coord cells
- 1 chemical + 1 DMSO per step, 1 epoch, forced `cpu`
- Early stop after 8 steps
- Saves `checkpoints/smoke.pt` (~100 MB)

Use it to validate the install, paths, and data loading before any real run.

---

## 5. Full training

```bash
python train.py                          # full config.yaml: 100 epochs, device auto (cuda > mps > cpu)
python train.py --epochs 5 --device cuda # quick GPU run
python train.py --seed 0                 # reproducible run
python train.py --config path/to/config.yaml
```

Training facts:
- **1 step = 1 cell line** + target chemicals (default 3) + DMSO controls (default 2)
- **1 epoch = each usable cell line once** (~50 steps)
- Batch is an 11-well load: 2 DMSO + 9 chemical wells (1 per k9 drug cluster)
- Checkpoint saved every `save_every` (default 20) epochs → `checkpoints/epoch_<n>.pt`, plus the final epoch
- **bf16 mixed precision is ON by default on CUDA** (`train.amp: bf16`). Set `amp: none` in `config.yaml` for pure fp32; `fp16` is also accepted (uses a GradScaler).
- Per-step log line: `[step N] epoch=.. line=.. task=.. G=.. B=.. loss=..`

---

## 6. Evaluation

```bash
python evaluate.py --checkpoint checkpoints/epoch_19.pt
python evaluate.py --smoke --checkpoint checkpoints/smoke.pt
python evaluate.py --checkpoint checkpoints/epoch_9.pt --device cuda
```

Metrics per cell line (aggregated at the end):
- **pearson** — corr(predicted mean, actual mean) over genes
- **delta** — corr(pred − control, actual − control); must be > 0 (model learns the drug effect)
- **identity** — corr(control, actual); the baseline to beat (~0.98)

---

## 7. CLI reference

### train.py

| Flag | Default | Description |
|---|---|---|
| `--config PATH` | `./config.yaml` | Config file to load |
| `--smoke` | off | Tiny memory-safe run (1 epoch, ≤8 steps, CPU) |
| `--epochs N` | config | Override epochs |
| `--device DEV` | auto | `cuda` / `mps` / `cpu` |
| `--seed N` | config | Override random seed |

### evaluate.py

| Flag | Default | Description |
|---|---|---|
| `--config PATH` | `./config.yaml` | Config file to load |
| `--smoke` | off | Use smoke model/data sizes |
| `--checkpoint PATH` | none | Checkpoint to load (skips random init) |
| `--device DEV` | auto | `cuda` / `mps` / `cpu` |

### verify.py

| Flag | Default | Description |
|---|---|---|
| `--root PATH` | repo dir | Dataset root to check |

---

## 8. Optional offline preprocessing

If the raw parquets **do not** contain `cell_cordi_1` / `cell_cordi_2`, the model falls back to random
coordinates. To compute and write real per-cell-line UMAP coordinates back into every well (in place):

```bash
# preview what would be processed
python extra_util/add_cell_coords_to_parquet.py --dry-run

# actually compute coords (8 worker processes, ~50-dim SVD -> 2-D UMAP per cell line)
python extra_util/add_cell_coords_to_parquet.py
```

Already-processed files are skipped automatically; use `--force` to recompute.

---

## 9. Notes & gotchas

- **Never load the dataset wholesale.** All code paths work one well (or one 11-well batch) at a time — keep it that way.
- **CLS marker**: every row starts `genes[0]==1`, `expressions[0]==-2.0` — the loader strips these automatically, do not "fix" the data manually.
- **Token offset +3**: gene tokens are remapped (`-3`) to 0-based vocabulary columns by `utils/load_utils.py`.
- **MoLFormer on MPS**: skipped by default (`torch.linalg.qr` unsupported) — passes `device="cpu"` automatically.
- **Config contains a latent scheduler setting** (`train.warmup_frac`) but `train.py` does not currently implement warmup/cosine decay — LR is fixed at `1e-4`.
- **Excluded cell lines**: `data.exclude_cell_lines` (default `[CVCL_1531, CVCL_1571]`) drops outlier-tiny lines before training/eval — they had only 129/210 cells and kept just 3k/5.5k genes, producing degenerate steps.
- **MoLFormer / transformers compat**: `models/utils.py` auto-patches `create_bidirectional_mask` (removed in transformers ≥ 4.53) and drops `token_type_ids` from the tokenizer call, so any recent transformers version works (tested on 4.57.6).
- Checkpoints are saved with `weights_only=False` (eval loads them with the same flag).
