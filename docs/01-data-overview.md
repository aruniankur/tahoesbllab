# Data Overview — merge_data & metadata

Schemas, sizes, and encoding rules. **Do not read the parquet files wholesale.**

## Big picture

| Folder | Size | Files | Content |
|---|---|---|---|
| `merge_data/` | ~330 GB | 1,344 | Raw expression, one `smp_*.parquet` per sample well |
| `metadata/` | ~87 GB | ~1,040 | Metadata tables, gene vocab, chemistry, pseudobulk DE |

95.6M cells x 62,710 genes — a full sparse matrix is **>1 TB in RAM**, never materialize it.

## `merge_data/` — expression data

`merge_data/smp_1495.parquet` ... `smp_2838.parquet` (1,344 files, ~245 MB avg). One file = one
sample well; a sample can contain **multiple cell lines** — always filter by `cell_line_id`.

Schema (one row per cell):

| Column | Type | Meaning |
|---|---|---|
| `genes` | list<int64> | Gene **token IDs**, ragged, parallel to `expressions` |
| `expressions` | list<float> | Raw counts, aligned with `genes` |
| `drug` | string | Treatment; `DMSO_TF` = vehicle control |
| `sample` | string | Sample id (`smp_XXXX`) |
| `cell_line_id` | string | Cellosaurus ID (e.g. `CVCL_0459`) |
| `BARCODE_SUB_LIB_ID` | string | Unique per cell; use as obs index |
| `plate` | string | `plate1` ... `plate14` |
| `phase` | string | Cell-cycle phase: `G1` / `S` / `G2M` |
| `drugconc` | double | Concentration in µM |

Encoding rules:
1. **CLS marker**: `genes[0] == 1`, `expressions[0] == -2.0` — strip both.
2. **Token offset**: tokens = `vocab_index + 3`; remap `-3` to index `gene_vocabulary.json` (0-2 reserved).
3. `genes` / `expressions` lengths must match after stripping (`utils/load_utils.py` raises `ValueError`).
4. Duplicate `(cell, gene)` entries possible after coalescing — build matrices with COO or
   `torch.sparse_coo_tensor(...).coalesce()`.

## `metadata/` — tables

| File | Rows | Content |
|---|---|---|
| `gene_vocabulary.json` (+`.jsonl`) | 62,710 | Ensembl ID -> 0-based token index; row order = matrix column order |
| `gene_metadata.parquet` | 62,710 | `gene_symbol`, `ensembl_id`, `token_id` |
| `sample_metadata.parquet` | 1,344 | Per-well QC metrics + drug/concentration |
| `drug_metadata.parquet` | 379 | `targets`, `moa-broad`/`moa-fine`, approval, SMILES, `pubchem_cid` |
| `cell_line_metadata.parquet` | 1,000 | ~ 50 lines x ~20 driver mutations — **not** one row per line |
| `obs_metadata.parquet` | 100.6M | Per-cell QC/phase — **never load fully** |
| `obs_metadata_small_aruni.parquet` | 67,043 | Sample x cell-line lookup (`drug_name`, `conc`, `unit`) |
| `dict_sample.json` | 1,344 | Sample -> gene token list (encoding artifact) |
| `pseudobulk_differential_expression/` | 1,026 shards | Pseudobulk DE results (~87 GB) |

> **Note:** the former lookup `obs_metadata_small_aruni2.csv` was **deleted** — use
> `obs_metadata_small_aruni.parquet` instead. Column names drift between tables
> (`drug` / `drugconc` vs `Compound` / `Concentration` / `Unit`); don't assume one schema.

## `metadata/Chemcial_info/` — chemistry

| File | Content |
|---|---|
| `drug_metadata.json` | 380 `{drug, canonical_smiles}` (incl. `DMSO_TF`) — single source of truth for SMILES |
| `molformer_embeddings.csv` | 380 x 770: `drug`, `smiles` + 768-d MoLFormer embedding |
| `drug_cluster_analysis_k2_to_k10.md` | K-Means clustering narrative; **recommended k=9** drug archetypes |

Embeddings generated with `ibm-research/MoLFormer-XL-both-10pct` (HF transformers,
`trust_remote_code=True`); cached in the CSV, so re-running is optional.

## Utilities

- `utils/load_utils.py` — `load_parquets_to_adata(paths, gene_vocab_path, normalize=False, log1p=False)`
  -> AnnData (CLS-strip, `-3` remap, CSR build, obs = sample/drug/cell_line_id/BARCODE_SUB_LIB_ID/
  plate/phase/drugconc). Pass subsets: all 1,344 wells ~ 1+ TB in RAM.
- `extra_util/add_cell_coords_to_parquet.py` — CLI: per-cell-line UMAP coordinates (TruncatedSVD ->
  UMAP, z-scored) written back into each `smp_*.parquet` as `cell_cordi_1` / `cell_cordi_2` (in place,
  resumable via skip / `--force`).
- `extra_util/virtual_cordi/cell_coords.py` — same idea for AnnData (`assign_cell_coords`).

## Typical pipeline arc

`parquet -> strip CLS + remap tokens -> sparse matrix -> AnnData -> normalize 1e4 -> log1p ->
HVG -> PCA -> neighbors -> UMAP -> (DE / synthetic cells / training batches)`
