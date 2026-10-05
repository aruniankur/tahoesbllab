"""data/step_generator.py

Turns a :class:`BatchSpec` (11 wells) into model-ready tensor dicts, one step
at a time. Pipeline (RAM-safe):

    11 parquet wells  --load_utils-->  1 adata
                     --split-->       50 per-cell-line adatas
                     --filter_genes--> filtered (min_cells, NO HVG; optional smoke cap)
    then per step: pick 1 cell line + 3 target chemicals + DMSO source cells
                   -> tensor dict for ExternalPertubationModel.

Per-atom MoLFormer embeddings are computed on the fly once per batch and cached
globally (no .pt cache file needed).
"""

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import anndata as ad

from utils.load_utils import _df_to_adata, _load_gene_vocab_df, OBS_COLUMNS
from models.utils import smiles_to_molformer

COORD_COLS = ["cell_cordi_1", "cell_cordi_2"]
READ_COLUMNS = OBS_COLUMNS + COORD_COLS + ["genes", "expressions"]


class LoadedBatch:
    """The in-memory version of one 11-well batch: 50 filtered line adatas."""

    def __init__(self, spec):
        self.spec = spec
        self.line_adata = {}       # cell_line_id -> adata (genes filtered/capped)
        self.gene_ids = {}         # cell_line_id -> np.int64 original gene columns
        self.lines = []            # cell lines present (sorted)
        self.targets = {}          # (line, sample) -> mean log1p [G] tensor
        self.drug_atoms = {}       # drug -> (emb [A,768] tensor, mask [A] bool)

    def has_sample(self, line, sample):
        if line not in self.line_adata:
            return False
        return (self.line_adata[line].obs["sample"].values == sample).any()


class TahoeStepGenerator:
    DMSO = "DMSO_TF"

    def __init__(self, cfg):
        self.cfg = cfg
        self.gene_vocab_df = _load_gene_vocab_df(cfg.paths.gene_vocab)
        self.drug_atoms_cache = {}   # global across batches (drug -> (emb, mask))

    # ------------------------------------------------------------------
    # batch construction
    # ------------------------------------------------------------------
    def load_batch(self, spec, progress=None):
        max_cells = self.cfg.data.max_cells_per_well
        batch = LoadedBatch(spec)

        adatas = []
        for sample in spec.all_samples():
            path = f"{self.cfg.paths.merge_data}/{sample}.parquet"
            df = pd.read_parquet(path, columns=READ_COLUMNS)
            if max_cells:
                df = df.head(max_cells)
            adatas.append(_df_to_adata(df, self.gene_vocab_df))
            if progress is not None:
                progress.update(1)

        adata = ad.concat(adatas, axis=0, join="inner", merge="same")

        exclude_lines = set(self.cfg.data.exclude_cell_lines or [])
        for line in sorted(adata.obs["cell_line_id"].astype(str).unique()):
            if line in exclude_lines:
                continue
            line_ad = adata[adata.obs["cell_line_id"].astype(str) == line].copy()
            sc.pp.filter_genes(line_ad, min_cells=self.cfg.data.min_cells)
            if line_ad.n_vars == 0:
                continue
            gene_idx = line_ad.var.index.astype(int).to_numpy()
            if self.cfg.data.max_genes:
                line_ad = line_ad[:, : self.cfg.data.max_genes]
                gene_idx = gene_idx[: self.cfg.data.max_genes]
            batch.line_adata[line] = line_ad
            batch.gene_ids[line] = gene_idx
            if progress is not None:
                progress.update(1)

        batch.lines = sorted(batch.line_adata.keys())

        for line, line_ad in batch.line_adata.items():
            X = line_ad.X
            for sample in set(line_ad.obs["sample"].values):
                mask = line_ad.obs["sample"].values == sample
                xs = X[mask]
                if xs.shape[0] == 0:
                    continue
                batch.targets[(line, sample)] = self._mean_log1p(xs)

        self._cache_drug_atoms(spec)
        batch.drug_atoms = {
            d: self.drug_atoms_cache[d] for d in spec.all_drugs()
            if d in self.drug_atoms_cache
        }
        return batch

    def _mean_log1p(self, xs):
        """Mean log1p(raw counts) over cells -> [G] float32 tensor."""
        if self.cfg.data.target_log1p:
            mean = np.asarray(xs.log1p().mean(axis=0)).ravel()
        else:
            mean = np.asarray(xs.mean(axis=0)).ravel()
        return torch.as_tensor(mean, dtype=torch.float32)

    def _cache_drug_atoms(self, spec):
        for drug in spec.all_drugs():
            if drug in self.drug_atoms_cache:
                continue
            emb, mask = smiles_to_molformer(
                spec.drug_smiles[drug], strip_special=True, device="cpu"
            )
            self.drug_atoms_cache[drug] = (emb[0].float(), mask[0])

    # ------------------------------------------------------------------
    # step construction
    # ------------------------------------------------------------------
    def make_step(self, batch, line, task=None, rng=None, n_chem=None, n_dmso=None):
        """One training step tensor dict for ``ExternalPertubationModel``.

        ``task``: "control_chem" (DMSO -> chemicals) or "chem_chem".
        Returns a dict with model kwargs + ``target`` [B,G] + ``meta``.
        """
        rng = rng or np.random.default_rng(0)
        n_chem = n_chem or self.cfg.step.n_chem
        n_dmso = n_dmso or self.cfg.step.n_dmso
        spec = batch.spec

        if task is None:
            task = self.cfg.step.task
        if task == "both":
            task = rng.choice(["control_chem", "chem_chem"])

        gene_idx = batch.gene_ids[line]
        line_ad = batch.line_adata[line]
        var_names = line_ad.var_names.to_numpy()
        usable = [s for s in spec.chem_samples if batch.has_sample(line, s)]

        # --- choose sources / targets -----------------------------------
        if task == "control_chem":
            if len(usable) < n_chem:
                raise ValueError(
                    f"line {line} has only {len(usable)} usable chemical wells "
                    f"(need {n_chem})"
                )
            targets = rng.choice(usable, size=n_chem, replace=False).tolist()
            sources = spec.dmso_samples[:n_dmso]   # each DMSO well separately
            pairs = [(s, t) for t in targets for s in sources]
        else:  # chem_chem
            if len(usable) < n_chem * 2:
                raise ValueError(
                    f"line {line} needs {n_chem * 2} chemical wells for chem_chem "
                    f"(has {len(usable)})"
                )
            targets = rng.choice(usable, size=n_chem, replace=False).tolist()
            t_drugs = {self._drug_of(spec, t) for t in targets}
            cand = [s for s in usable if self._drug_of(spec, s) not in t_drugs]
            if len(cand) < n_chem:
                cand = [s for s in usable if s not in targets]
            sources = rng.choice(cand, size=n_chem, replace=False).tolist()
            pairs = list(zip(sources, targets))

        # --- build tensors -----------------------------------------------
        B = len(pairs)
        gene_ids = torch.as_tensor(
            np.repeat(gene_idx[None, :], B, axis=0), dtype=torch.long
        )

        gene_exp, chem_init, chem_init_conc, chem_init_mask = [], [], [], []
        chem_final, chem_final_conc, chem_final_mask = [], [], []
        cell_coordi, target = [], []

        for s, t in pairs:
            gene_exp.append(self._source_density(line_ad, s, var_names, rng))
            chem_init.append(batch.drug_atoms[self._drug_of(spec, s)][0])
            chem_init_conc.append([float(self._conc_of(spec, s))])
            chem_init_mask.append(batch.drug_atoms[self._drug_of(spec, s)][1])
            chem_final.append(batch.drug_atoms[self._drug_of(spec, t)][0])
            chem_final_conc.append([float(self._conc_of(spec, t))])
            chem_final_mask.append(batch.drug_atoms[self._drug_of(spec, t)][1])
            cell_coordi.append(self._sample_coords(line_ad, t, rng))
            target.append(batch.targets[(line, t)])

        chem_init = self._pad_embeddings(chem_init)
        chem_init_mask = self._pad_masks(chem_init_mask)
        chem_final = self._pad_embeddings(chem_final)
        chem_final_mask = self._pad_masks(chem_final_mask)

        return {
            "gene_ids": gene_ids,
            "gene_exp": torch.stack(gene_exp),
            "chem_init": torch.stack(chem_init),
            "chem_init_conc": torch.as_tensor(chem_init_conc),
            "chem_init_mask": torch.stack(chem_init_mask),
            "chem_final": torch.stack(chem_final),
            "chem_final_conc": torch.as_tensor(chem_final_conc),
            "chem_final_mask": torch.stack(chem_final_mask),
            "cell_coordi": torch.stack(cell_coordi),
            "target": torch.stack(target),
            "meta": {
                "line": line,
                "task": task,
                "G": len(gene_idx),
                "n_genes_vocab": len(self.gene_vocab_df),
                "sources": sources,
                "targets": targets,
                "pairs": pairs,
            },
        }

    def _pad_embeddings(self, embs):
        """Pad per-atom embeddings to the max atom count in the list."""
        max_len = max(e.shape[0] for e in embs)
        return [
            e if e.shape[0] == max_len else torch.nn.functional.pad(
                e, (0, 0, 0, max_len - e.shape[0]))
            for e in embs
        ]

    def _pad_masks(self, masks):
        """Pad attention masks (bool) to the max atom count in the list."""
        max_len = max(m.shape[0] for m in masks)
        return [
            m if m.shape[0] == max_len else torch.nn.functional.pad(
                m, (0, max_len - m.shape[0]))
            for m in masks
        ]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _drug_of(self, spec, sample):
        if sample in spec.dmso_samples:
            return self.DMSO
        return spec.chem_drugs[spec.chem_samples.index(sample)]

    def _conc_of(self, spec, sample):
        if sample in spec.dmso_samples:
            return spec.dmso_concs[spec.dmso_samples.index(sample)]
        return spec.chem_concs[spec.chem_samples.index(sample)]

    def _source_density(self, line_ad, sample, var_names, rng):
        """convert_to_distribution over a sample's cells -> [G, 128]."""
        from models.modelutil import convert_to_distribution

        mask = line_ad.obs["sample"].values == sample
        X = line_ad.X[mask]
        if X.shape[0] > self.cfg.data.max_cells_for_density:
            keep = rng.choice(X.shape[0], self.cfg.data.max_cells_for_density,
                              replace=False)
            X = X[keep]
        dense = torch.as_tensor(np.asarray(X.todense()), dtype=torch.float32)
        if self.cfg.data.source_log1p:
            dense = torch.log1p(dense)
        return convert_to_distribution(dense).to(torch.float32)  # [G, 128]

    def _sample_coords(self, line_ad, sample, rng):
        n = self.cfg.data.n_coord_cells
        obs = line_ad.obs
        mask = obs["sample"].values == sample
        if COORD_COLS[0] not in obs.columns or COORD_COLS[1] not in obs.columns:
            return torch.rand(n, 2, dtype=torch.float32) * 10.0 - 5.0
        c = obs.loc[mask, COORD_COLS].to_numpy(dtype=np.float32)
        if len(c) == 0:
            return torch.rand(n, 2, dtype=torch.float32) * 10.0 - 5.0
        idx = rng.choice(len(c), size=n, replace=(len(c) < n))
        return torch.as_tensor(c[idx], dtype=torch.float32)

    @staticmethod
    def to_device(step, device):
        out = {}
        for k, v in step.items():
            if k == "meta":
                out[k] = v
            elif isinstance(v, torch.Tensor):
                out[k] = v.to(device)
        return out