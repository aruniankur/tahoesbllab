# Objective - Gene-Expression Transition Model (Chemical A -> Chemical B)

**The goal, in one sentence:** Train a deep-learning model that takes the gene expression of
all genes of a cell-line population in **chemical A** and predicts the gene expression of that
**same cell line** - same genes, same gene count - in **chemical B**.

We train this model on the Tahoe-100M dataset (100M+ single-cell profiles, 50 cancer cell
lines, ~1,100 drug perturbations; 1,344 raw `smp_*.parquet` wells in `merge_data/`).

---

## 1. Task specification

### Inputs (per example)

| # | Input | Meaning | Data source |
|---|---|---|---|
| 1 | `gene_id` | gene token index, 0-based vocab order (after the -3 remap) | `merge_data` `genes` + `metadata/gene_vocabulary.json` (G = 62,710) |
| 2 | `gene_exp` | expression value of each gene (normalized / log1p density) | `merge_data` `expressions` |
| 3 | `init_chem` | chemical(s) present in condition A (MoLFormer embedding per atom) | `metadata/Chemcial_info/drug_metadata.json` + `molformer_embeddings.csv` |
| 4 | `init_conc` | concentration(s) of the A chemical(s) | `metadata/obs_metadata_small_aruni.parquet` (`conc`) |
| 5 | `init_mask` | presence mask over the A atoms (handles variable #atoms via padding) | derived from the sample's drug list |
| 6 | `final_chem` | chemical(s) present in condition B (target) | same source as #3 |
| 7 | `final_conc` | concentration(s) of the B chemical(s) | same source as #4 |
| 8 | `final_mask` | presence mask over the B atoms | derived |
| 9 | `cell_cordi` | **list of N query coordinates (x, y)** - N ~ 200-300, a hyperparameter | `cell_cordi_1` / `cell_cordi_2` (written back into the parquets by `extra_util/add_cell_coords_to_parquet.py`) |

### Output

For each of the **N queried coordinates** (N ~ 200-300, a hyperparameter) the model predicts
one full expression vector over the same G genes -> output shape **(N, G)** per example
(log1p scale). Stacking the N per-coordinate predictions reconstructs a synthetic single-cell
population (cells x genes). The population-level prediction used for the loss / metric is the
aggregate over the N coordinates (e.g., the mean log1p vector).

---

## 2. Model contract (already designed: `models/sbl_cell_transition_model_v2.py`)

```python
SBLCellTransitionModel.forward(
    gene_ids, gene_exp,
    chem_emb_init, chem_conc_init, chem_mask_init,
    chem_emb_final, chem_conc_final, chem_mask_final,
    cell_coordi)                       # -> (B, N, G) log1p expression, N = #query coords
```

- Genes are **tokens**; their states evolve through an encoder stack (condition A context)
  then a decoder stack (condition B context).
- Chemicals are **cross-attended context**: (MoLFormer embedding + concentration) is projected
  into gene-state space, masked by `chem_mask`.
- The query head takes (final gene state, x, y) and predicts that gene's log1p expression.
  We pass a list of N coordinates (e.g., 200-300); the model returns one expression vector
  per coordinate.

---

## 3. Non-negotiable data constraints

1. **Same plate** for A and B - DMSO reference and drug must come from the SAME plate
   (plate-effect: within-plate separability AUC 0.83 vs cross-plate 0.96; 2,632 same-plate
   DMSO -> drug pairs available).
2. **Same cell line** - one sample well can contain multiple cell lines (mixed spheroids);
   always filter by `cell_line_id`.
3. **CLS marker strip** - every row starts `genes[0] == 1`, `expressions[0] == -2.0`; strip
   both, then remap gene tokens by -3.
4. **Never load data wholesale** - `merge_data/` is ~330 GB; work one `smp_*.parquet` at a
   time (~245 MB each), filter columns/rows.

---

## 4. Success criteria

- Correlation (Pearson / Spearman) between predicted and actual chem-B expression per gene;
  MAE/MSE on the log1p scale.
- Baselines to beat: identity (predict A as B), per-cell-line mean response, per-drug mean response.
- Evaluation: aggregate the per-coordinate predictions (mean over N) and compare to the
  real chem-B population log1p vector, per gene.
- Sanity: within-plate A->B pairs should be substantially easier than cross-plate pairs.

---

## 5. Our job / plan

1. **Dataloader** - emit the 9 inputs from `smp_*.parquet` + metadata, pairing same-plate
   (DMSO -> drug) samples per cell line. Per (cell line, sample) population, sample
   N = 200-300 cells from the chem-B well: their `cell_cordi_1/2` become the query list and
   their real expression becomes the supervision target.
2. **Model** - implement / adapt `SBLCellTransitionModel` (v2 skeleton exists; smoke test
   passes: `(3, 20, 1000)`).
3. **Train** - start on a subset (one plate or one cell line), then scale up.
4. **Evaluate** - against the baselines above; iterate.

---

## 6. Open questions (to resolve while building)

- **Query coordinates at inference (resolved for training, open for inference):** during
  training we sample N = 200-300 cells from the real chem-B population - their actual
  `cell_cordi_1/2` are the queries and their real expression the targets. At inference the
  chem-B cells are unseen, so we must choose where the N coordinates come from: the chem-A
  source cells' coordinates, a fixed grid, or sampled from the cell line's spatial
  distribution.
- `gene_exp` is a scalar per gene in the raw data; the model's `gene_exp_dim` expects a
  vector - need a projection (or set `gene_exp_dim = 1`).
- A -> B pairs: default assumption is DMSO (control) -> drug; drug -> drug is also expressible.
- Concentration encoding: raw value vs log-scale; units per `obs_metadata_small_aruni.parquet`.
