# Tahoe-100M Workspace — Index

Read this first. This is a local working copy of the **Tahoe-100M** single-cell perturbation
atlas (100M+ profiles, 50 cancer cell lines, ~1,100 drug perturbations; upstream card:
`../README.md`) plus exploratory analysis code. It is a research scratch workspace, not a
maintained library.

## Directory map

| Path | Contents |
|---|---|
| `merge_data/` | **~330 GB, 1,344 files** — raw expression, one `smp_*.parquet` per sample well (one row per cell, ragged `genes[]` / `expressions[]`). |
| `metadata/` | **~87 GB** — gene vocabulary, metadata tables, `Chemcial_info/` (drug SMILES + MoLFormer embeddings), pseudobulk DE shards. |
| `bioanalysis/` | **Ignored** (`.gitignore` / `.ignore`) — analysis notebooks + `plate_diff_CVCL_0546.zip` (plate-effect study). |
| `extra_util/` | Training-batch prep (`modeltrainhelper2.ipynb`), `add_cell_coords_to_parquet.py`, `virtual_cordi/` (cell coordinates). |
| `models/` | `sbl_cell_transition_model_v2.py` — `SBLCellTransitionModel` (condition A -> B virtual cell). |
| `utils/` | `load_utils.py` — consolidated parquet -> AnnData loader. |
| `docs/` | This folder — `01-data-overview.md`, `objective_goal.md`, `objectivearuni.md`. |

## Non-negotiable gotchas

1. **Never load the data wholesale.** `merge_data/` ~ 330 GB; `obs_metadata.parquet` = 100.6M rows.
   Work one file at a time (~245 MB), filter columns/rows.
2. **CLS marker**: every row starts `genes[0] == 1`, `expressions[0] == -2.0` — strip both.
3. **Token offset +3**: gene tokens = vocab index + 3; remap `tokens - 3` (0-2 reserved).
4. **One sample = multiple cell lines** (mixed spheroids) — always filter by `cell_line_id`.
5. **`tahoe_model` / `Archive.zip` are gone.** `extra_util/modeltrainhelper2.ipynb` still imports
   `tahoe_model.model.chempertubation` and cannot run. The current model is
   `models/sbl_cell_transition_model_v2.py`.
6. **`obs_metadata_small_aruni2.csv` was deleted** — use `metadata/obs_metadata_small_aruni.parquet`
   for sample x cell-line lookups.
7. **Loader code is duplicated** across notebooks (`convert_df_to_adata`); `utils/load_utils.py` is
   the canonical version. Column names drift between tables (`drug` / `drugconc` vs
   `Compound` / `Concentration` / `Unit`).

## Docs

| Question | File |
|---|---|
| Data schemas, encoding, chemistry, utilities | `01-data-overview.md` |
| Modeling objective (condition A -> B) | `objectivearuni.md` |
| **The goal - full statement** | `objective_goal.md` |
