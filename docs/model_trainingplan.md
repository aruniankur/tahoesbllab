# Tahoe-100M Virtual Cell Model — Training Plan

**Status:** Draft v1 — locked decisions marked ✅, open decisions marked ⚠️ with a default choice used throughout.

**Related docs:** `objectivearuni.md` (prior analysis), `01-data-overview.md` (dataset).

---

## 1. Objective

Train a "virtual cell" model that takes gene expression in a **source condition** and predicts
gene expression in a **target condition** (a perturbation), producing a virtual cell.

Two training tasks:

| Task | Source (A) | Target (B) |
|---|---|---|
| Control → chemical | DMSO (vehicle control) | any chemical at any concentration |
| Chemical → chemical | any chemical at any concentration | any other chemical (or same chemical, different concentration) |

Output contract: for a given cell line, the model predicts **per-gene log1p counts** for
**N = 200 cells** sampled at the target condition's spatial coordinates.

---

## 2. Dataset Summary (verified on disk)

| Resource | Count |
|---|---|
| Total wells (samples) | 1,344 |
| Plates | 14 (96 wells each) |
| Cell lines per well | 50 (filter by `cell_line_id`) |
| DMSO wells | 28 (2 per plate) |
| **Chemical wells** | **1,316** |
| Drug names (incl. DMSO_TF) | 380 → **379 chemicals** |
| Unique drug × conc conditions | 1,138 (mostly 3 concs: 0.05 / 0.5 / 5 µM) |
| Genes | 62,710 (tokens 3..62712; tokens 0–2 reserved) |
| k9 MOA-like clusters (MoLFormer-based) | 9 clusters — **imbalanced**: sizes {0:11, 1:52, 2:35, 3:13, 4:81, 5:71, 6:24, 7:43, 8:50} |

Key files:

| Asset | Path |
|---|---|
| Raw per-well parquets | `merge_data/smp_*.parquet` (1,344 files, 140–330 MB each) |
| Consolidated loader | `utils/load_utils.py::load_parquets_to_adata` |
| Drug metadata | `metadata/drug_metadata.parquet` (379 chemicals) |
| Chemical info (JSON + MoLFormer 768-d **pooled** embeddings) | `metadata/Chemcial_info/` |
| MoLFormer **per-atom** embeddings (TO GENERATE) | `metadata/Chemcial_info/molformer_atom_embeddings.pt` (§4.1) |
| Cluster labels (k9) | `metadata/Chemcial_info/drug_cluster_labels_k9.parquet` |
| Gene vocabulary | `metadata/gene_vocabulary.json` (+ `gene_metadata.parquet`) |
| Sample → drug/conc/plate lookup | `metadata/sample_metadata.parquet` |
| Model | `models/externalpertubationmodel.py` + `models/modelutil.py` |

---

## 3. Locked Design Decisions ✅

| # | Decision |
|---|---|
| D1 | **Cross-plate pairing is ALLOWED.** Any DMSO can pair with any chemical. ⚠️ *This explicitly overrides the same-plate constraint from `objectivearuni.md` — see Risks R1.* |
| D2 | **Batch = 1 cell line, 3 chemical targets.** Every step uses exactly one cell line, so all gene IDs in a batch are identical (post per-line filtering). Batch size (number of pairs) may vary between steps. |
| D3 | **Epoch = 50 steps** (each cell line seen once). Optionally 100 steps (each line twice). |
| D4 | **No DMSO averaging.** Each DMSO well is a separate control → 2 controls per chemical (generalization across control wells). |
| D5 | Source expression feature = **`convert_to_distribution`** (per-gene histogram → 128-bin probability distribution, from `modelutil.py`, `steps=129, a=0, b=10`, normalized). |
| D6 | Loss = **`squared_error_loss`** from `modelutil.py`: `((X1-X2)^2).sum(dim).mean()`. |
| D7 | **200 cells** per target = random 200 cells of that (line × target) condition; coordinates `cell_cordi_1/2` come from the parquets. |
| D8 | **Split = pair-based holdout**: train on some (source → target) pairs, evaluate on unseen pairs. |
| D9 | Per-cell-line gene filtering with **`filter_genes(min_cells=10)`** on the combined 11-well adata (control and targets share the same gene set). |
| D10 | Drug conditioning = **MoLFormer per-atom token embeddings** (`last_hidden_state`, 768-d per atom) + per-token mask + concentration scalar → `chem_* [*, n_atoms, 769]` + `chem_*_mask [*, n_atoms]`. Pooled 768-d CSV kept as fallback / for clustering. |

---

## 4. Data Loading Pipeline (RAM-safe)

```text
merge_data/smp_*.parquet  (1,344 files, ~408 GB total)
        │  load 11 files at a time
        ▼
load_parquets_to_adata(11 paths, gene_vocabulary.json)
   • strips CLS marker (genes[0]==1, expressions[0]==-2.0)
   • remaps gene tokens (-3) → 0-based CSR columns
   • concatenates into 1 adata  (obs: drug, conc, cell_line_id, coords, …)
        │
        ▼
split by cell_line_id → 50 per-cell-line adatas
        │
        ▼
for each line adata:  sc.pp.filter_genes(adata, min_cells=10)
        │            (filter on the COMBINED 11-condition adata,
        │             so source & target share gene columns)
        ▼
train step generator (Section 5)
```

**11-file selection (per load):**

| Slot | Selection rule |
|---|---|
| 2 × DMSO | any 2 DMSO wells (any plates) |
| 9 × chemical | 1 chemical per k9 cluster (9 clusters), concentrations cycling 1→0.05, 2→0.5, 3→5, 4→0.05, … 9→5 µM |

**Memory estimate:** ~600k cells × 62,710 genes (sparse, ~3–5k nnz/cell) ≈ a few GB in RAM.
Process cell lines one at a time (create → filter → use → delete) — do **not** keep all 50
line adatas resident simultaneously.

### 4.1 MoLFormer per-atom embeddings (verified ✅)

Verified 2026-10-05 against the locally cached `ibm-research/MoLFormer-XL-both-10pct`
(transformers 5.17, trust_remote_code):

| Property | Verified value |
|---|---|
| Per-token (≈per-atom) embedding | `last_hidden_state` → `[B, seq_len, 768]` |
| Attention mask | `attention_mask` → `[B, seq_len]`, 1 = real token, 0 = padding |
| Tokenization | atom-level SMILES tokens: DMSO `CS(=O)C` → `<bos> C S ( = O ) C <eos>` (8 real tokens) |
| Padding | batch-padded to the longest SMILES in the batch (variable `seq_len` per molecule) |

Notes:
- Special tokens `<bos>` / `<eos>` frame the molecule — keep or strip per O6
  (default used: **strip**, via `strip_special=True`).
- The existing `molformer_embeddings.csv` holds only the **pooled** [CLS] vector (768-d);
  per-atom states are **not** stored and must be generated once (offline step):
  1. load the 380 SMILES from `metadata/Chemcial_info/drug_metadata.json`
  2. run MoLFormer via `models/utils.py::smiles_to_molformer(smiles, pooler=False)`
     (cached locally, no download needed; auto device = cuda → cpu, MPS skipped)
  3. save `metadata/Chemcial_info/molformer_atom_embeddings.pt` as
     `{drug: {"emb": [n_atoms, 768], "mask": [n_atoms]}}` — est. ~70 MB total
- Model compatibility: `ExternalPertubationModel` already accepts multi-token chemicals
  (`chem_init [B, C_init, chem_dim]` + `chem_init_mask [B, C_init]`, masked cross-attention).
  Concentration is concatenated to **every** atom token by `chem_embedding = Linear(768+1, 768)`
  — same conc for all atoms of a molecule, **no code change required**.

---

## 5. Training Step

### 5.1 Control → chemical (default: B = 6 pairs/step = 3 chemicals × 2 DMSO wells)

```text
1. pick 1 cell line L
2. source = DMSO cells of line L        (2 wells → 2 separate controls)
3. targets = 3 chemicals of line L      (random 3 of the 9 in the load)

gene_ids     [G]                        filtered gene set of line L
gene_exp     [B, G, 128]                per-source convert_to_distribution (128 bins)
chem_init    [B, A_init, 769]           DMSO atoms (MoLFormer per-atom 768 + conc), A_init = n_atoms(DMSO)
chem_init_mask[B, A_init]               1 = real atom, 0 = padding
chem_final   [B, A_final, 769]          target chemical atoms (+ conc)
chem_final_mask[B, A_final]             1 = real atom, 0 = padding
cell_coordi  [B, 200, 2]                random 200 coords from target's cells

model output [B, 200, G]
        │  mean over the 200 cells (dim=1)
        ▼
predicted mean log1p counts  [B, G]
        vs
actual mean log1p counts     [B, G]     (from target cells of line L)
        ▼
squared_error_loss(pred, actual)   → backward
```

### 5.2 Chemical → chemical

Same step shape, except:
- `chem_init` = source chemical embedding + conc (not DMSO)
- `chem_final` = target chemical embedding + conc
- Default pairing: all directed pairs among the 3 chemicals in the step, **self-pairs excluded,
  same-chemical-different-concentration allowed** (e.g., 0.05 → 5 µM). ⚠️ *open decision, see O2.*

### 5.3 Epoch accounting

| Definition | Steps |
|---|---|
| 1 epoch = each of 50 lines once | **50 steps** ✅ (D3) |
| 1 epoch = each of 50 lines twice | **100 steps** |

With 3-of-9 chemicals per line per step, a given chemical appears in a line roughly every
3rd epoch (random sampling). ⚠️ *If full 9-chemical coverage per line per epoch is required →
3 steps/line → 150 steps/epoch. See O3.*

---

## 6. Model Wiring (`ExternalPertubationModel`)

| Model input | Data source | Shape |
|---|---|---|
| `gene_ids` | filtered gene token IDs of line L | [G] |
| `gene_exp` | `convert_to_distribution(DMSO cells of L)` | [B, G, 128] |
| `chem_init` / `chem_init_conc` / `chem_init_mask` | DMSO per-atom (MoLFormer `last_hidden_state` 768 + conc) | [B, A, 769] / [B,1] / [B, A] |
| `chem_final` / `chem_final_conc` / `chem_final_mask` | target chemical per-atom (+ conc) | [B, A, 769] / [B,1] / [B, A] |
| `cell_coordi` | 200 random coords of target cells | [B, 200, 2] |
| **output** | per-cell per-gene log1p counts | [B, 200, G] |

Architecture flow (as implemented):

```text
gene_ids ──embedding──► gene_emb [B,G,768]
                              │
chem_init(+conc) ──Linear──► chem_emb_init
                              │
gene_emb ◄──cross-attention── chem_emb_init   (PerturbationCrossAttention)
                              │
state = Linear([gene_emb, gene_exp, gene_pert])      [B,G,768]
                              │
final_state = PerturbationStateAttentionStack(gene_emb, state, chem_emb_final)
                              │
output = GeneCellCoordinateIntegrationModel(final_state, cell_coordi)   [B,200,G]
                              │  (shared Linear([gene_state, coord]) + ReLU)
                              ▼
                        mean over 200 cells → [B,G]
```

Config defaults: `emb_dim=768, chem_coordi_dim=768, chem_conc_dim=1, gene_exp_coordi=128,
coord_dim=2, heads=8, layers=4`. `C_init` / `C_final` = per-molecule atom counts (8–~80),
variable per batch; masks handle padding.

---

## 7. Loss & Evaluation

### Loss
- `squared_error_loss(predicted_mean_log1p, actual_mean_log1p)` — MSE over genes, averaged
  over the batch. ⚠️ *Alternative: per-cell loss on [B,200,G] before mean — see O4.*

### Evaluation metrics (report per cell line + overall)

| Metric | Definition | Why |
|---|---|---|
| Pearson r (predicted vs actual, per cell line) | corr over genes | raw fit quality |
| **Delta recovery** | corr(predicted − control, actual − control) | must be > 0 to prove the model learns the drug effect, not just copies the control |
| Identity baseline | "predict control as answer" | r ≈ 0.98 — the bar the model must beat (R5) |
| Held-out pair performance | eval on unseen (source→target) pairs | generalization |

---

## 8. Train / Val / Test Split

- **Pair-based holdout** (D8): split (source → target) pairs.
- ⚠️ Two strengths — see O5:
  - **Weak**: unseen (source → target) *pairs* only — both chemicals have appeared in training
    (with other partners). Model has seen both MoLFormer embeddings.
  - **Strong**: hold out entire *chemicals* (never in training at all).
- Default used in this plan: **weak pair split (90/10)**, with the strong chemical split
  reported as a stress test.
- No plate-based split (cross-plate allowed, D1). ⚠️ Consequence: evaluate plate-wise anyway
  as a diagnostic (R1).

---

## 9. Hyperparameters (defaults)

| Hyperparameter | Default | Notes |
|---|---|---|
| Optimizer | AdamW | β1=0.9, β2=0.999, eps=1e-8 |
| Learning rate | 1e-4 | warmup 5% of steps, cosine decay to 1e-6 |
| Weight decay | 0.01 | |
| Batch size (B) | 6 | 3 chemicals × 2 DMSO (control→chem) |
| N cells per target | 200 | random, resampled every step (augmentation) |
| Epochs | 100 | = 50 steps each; 5,000 total steps |
| Gradient clipping | 1.0 | |
| Gradient accumulation | as needed | if B > 6 exceeds GPU memory |
| Mixed precision | bf16 (if Ampere+) | fallback fp32 |
| Seed | 0 | |

---

## 10. Open Decisions ⚠️ (defaults used above)

| # | Question | Default used |
|---|---|---|
| O1 | DMSO count per 11-file load: 2 or 3? (D4 uses 2 → B=6) | **2 DMSO wells** |
| O2 | Chemical→chemical pairing: directed pairs among the 3 in-batch chemicals? same chemical different conc allowed? | **yes / yes (self excluded)** |
| O3 | 3-of-9 chemicals per line per epoch (50 steps) OK, or all-9 coverage (150 steps/epoch)? | **3-of-9 random, 50 steps** |
| O4 | Loss on mean over 200 cells [B,G], or per-cell [B,200,G]? | **mean over 200 → [B,G]** |
| O5 | Eval: weak pair split vs strong unseen-chemical split? | **weak 90/10 + strong as stress test** |
| O6 | MoLFormer token framing: keep `<bos>`/`<eos>` in the atom sequence, or strip? (A varies 8–~80 tokens/molecule) | **keep bos/eos** |

---

## 11. Risks & Mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | **Plate confounding** (accepted by D1). Own analysis: cross-plate DMSO separability AUC 0.96 ≈ drug effect 0.97; same drug's response correlates r≈0.13 across plates. | Model may learn plate identity as a shortcut. Report held-out plate performance separately as a diagnostic. |
| R2 | **Cluster imbalance** (sizes 11…81). "1 per class" sampling oversamples rare clusters. | Accept for diversity; log per-cluster coverage; optionally weight sampling. |
| R3 | **Per-line gene filtering → variable G per batch.** | Model takes explicit `gene_ids`; loss is per-gene-indexed. Ensure filter applied on the combined 11-well line adata (D9). |
| R4 | **Memory.** 50 line adatas must not be resident at once. | Stream line-by-line; delete after use. |
| R5 | **Identity baseline r≈0.98.** Raw correlation is misleading. | Always report delta recovery (Section 7). |
| R6 | **DMSO at 3 "concentrations"?** Verify which conc DMSO_TF carries and how `chem_conc` is set for controls. | Confirm from `sample_metadata.drugname_drugconc`. |
| R7 | **min_cells=10** may drop genes in small lines (some wells ~few k cells). | Log genes kept per line; sanity-check G per line. |
| R8 | **Variable atom counts** (8–~80 tokens/molecule). | Pad to max atoms in batch + mask (MoLFormer-native); conc broadcast per atom. |

---

## 12. Status / TODO

- [x] Dataset inventory (wells, drugs, concs, plates, genes)
- [x] Model architecture mapped to data pipeline
- [x] Locked design decisions (D1–D10)
- [ ] Resolve open decisions O1–O5
- [ ] Implement 11-file batch sampler (1-per-cluster, conc cycling)
- [ ] Implement per-line adata splitter + filter_genes
- [ ] Implement step generator (both tasks) → tensor dict
- [ ] Wire ExternalPertubationModel forward + squared_error_loss
- [ ] Implement pair-split (weak + strong) 
- [ ] Baseline: identity (control-as-answer) delta-recovery ceiling
- [ ] Train 100 epochs; log loss, r, delta recovery per line
- [ ] Evaluate on held-out pairs + held-out plates (diagnostic)

---

*Plan drafted: 2026-10-05. Decisions D1–D10 confirmed in discussion; O1–O5 pending.*
