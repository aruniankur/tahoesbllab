# Objective — Aruni: Virtual Cell Model (condition A -> condition B)

**Goal:** Build a model that takes gene expression in condition A (e.g., DMSO control)
and predicts gene expression in condition B (e.g., drug-treated), producing a "virtual cell."

## 1. Design Constraint (from plate-effect analysis — MANDATORY)

> **DMSO reference and drug-condition must come from the SAME PLATE. No cross-plate pairing.**

Rationale (measured on this dataset):

| Effect | Separability AUC | Meaning |
|---|---|---|
| Within-plate well noise (DMSO well A vs B) | 0.83 | Even same-plate wells differ slightly |
| Cross-plate batch effect (DMSO plate1 vs 2) | 0.96 | Plate identity dominates |
| Drug effect (Docetaxel 5µM vs DMSO, cross-plate) | 0.97 | Comparable to plate effect! |
| Drug x plate interaction (FC corr plate6 vs plate14) | r ~ 0.13 | Drug response fingerprint shifts by plate |

**Consequences:**
1. Cross-plate training makes the model learn plate-fingerprints, not drug effects -> confounded.
2. Drug-effect generalization across plates is limited (r ~ 0.13) -> test on held-out plates only,
   always paired with that plate's own DMSO.
3. Same-plate still contains well-position noise (AUC 0.83) -> use the MEAN of the 2 DMSO wells
   per plate as the reference.

## 2. Data Resources (verified)

| Resource | Count |
|---|---|
| Total samples (wells) | 1,344 |
| Plates | 14 (96 wells each) |
| DMSO wells | 2 per plate (28 total) |
| **Same-plate DMSO->drug pairs** | **2,632** (188 per plate) |
| Cell lines per well | 50 (all lines in every well — filter by `cell_line_id`) |
| Compounds | 380 drugs, ~1,100 drug x conc conditions |
| Conditions replicated across plates | 156 conditions on >= 2 plates (avg 1.15 plates/condition) |

**Data encoding notes:**
- Strip CLS marker: `genes[0] == 1`, `expressions[0] == -2.0`
- Gene tokens = vocab_index + 3 -> remap `-3` (tokens 0-2 reserved)
- Sample parquets carry `drug` / `drugconc` / `phase` directly; per-cell-line friendly names come
  from `metadata/obs_metadata_small_aruni.parquet` (`drug_name` / `conc` / `unit`). The former
  `obs_metadata_small_aruni2.csv` lookup is gone — do not reference it.

## 3. Model Pipeline (recommended)

```
+------------------------------------------------------+
| 1. DATA   2,632 same-plate DMSO->drug pairs           |
| 2. REF    mean of the 2 DMSO wells per plate          |
|           (plate-level control baseline)              |
| 3. SPLIT  train 12 plates / val 1 / test 1            |
|           NEVER split by cells — split by PLATE       |
| 4. MODEL  predict delta = drug - DMSO                 |
|           input:  [DMSO expression (this plate)]      |
|                   + [drug identity / MoLFormer embed] |
|           output: [predicted drug expression]         |
| 5. EVAL   corr(predicted, actual) + delta recovery    |
|           vs baseline "predict DMSO as answer"        |
+------------------------------------------------------+
```

### Key design decisions
1. **Condition on the plate's own DMSO baseline** -> model learns the *delta*, the
   plate-effect-free quantity.
2. **Plate-wise data split** (train 12 / val 1 / test 1) -> honest evaluation of generalization
   to unseen plates.
3. **Drug conditioning** via SMILES (MoLFormer 768-d embeddings already computed in
   `metadata/Chemcial_info/molformer_embeddings.csv`) or one-hot drug id.
4. **Baseline to beat:** "predict drug expression = DMSO expression" gives r ~ 0.98 with truth.
   The model must beat that — i.e., recover the *delta* (drug - DMSO) better than zero.

## 4. Evaluation Metrics

| Metric | Target |
|---|---|
| Pearson r (predicted vs actual, per cell line) | > 0.98 (baseline is ~0.98) |
| Delta recovery: corr(predicted delta, true delta) | > 0 (positive = learns drug effect) |
| Per-cell-line breakdown | all 50 lines must show positive delta recovery |
| Held-out plate performance | report separately for test plate |

## 5. Implementation Notes

- **Existing assets to reuse:**
  - `utils/load_utils.py` -> consolidated parquet -> AnnData loader (CLS-strip, token remap, CSR)
  - `metadata/obs_metadata_small_aruni.parquet` -> sample <-> drug/concentration lookup
  - `metadata/Chemcial_info/molformer_embeddings.csv` -> 768-d drug embeddings
  - `metadata/gene_vocabulary.json` + `gene_metadata.parquet` -> gene indexing
  - `models/sbl_cell_transition_model_v2.py` -> `SBLCellTransitionModel` (gene tokens +
    cross-attended chemical context + coordinate query head) — a candidate architecture
- **Caution:** `extra_util/modeltrainhelper2.ipynb` builds cross-plate batches
  (control/low/mid/high from different plates) -> **do NOT use as-is; redesign to same-plate
  pairing.** It also imports the removed `tahoe_model` package and cannot run without restoration.
- Memory: process one plate at a time (~96 wells); never load the full dataset (330 GB).

## 6. Status / TODO

- [x] Plate-effect quantification (all 50 lines, within/cross-plate, drug 2x2) — outputs archived
      in `bioanalysis/plate_diff_CVCL_0546.zip` (see `summary.txt`)
- [x] Phase analysis (not a driver of plate or drug separability; G2M arrest real but minor)
- [x] Same-plate pair inventory (2,632 pairs)
- [ ] Build same-plate training-set generator (plate-wise split)
- [ ] Prototype identity-baseline + delta-recovery ceiling
- [ ] Train virtual-cell model (delta = drug - DMSO, conditioned on DMSO + drug embedding)
- [ ] Evaluate on held-out plates per cell line

---

*Analysis date: 2026-09-18. Plate-effect findings: `bioanalysis/plate_diff_CVCL_0546.zip`.*
