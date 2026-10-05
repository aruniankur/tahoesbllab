"""data/sampler.py

Pure well-selection logic for Tahoe-100M training batches.

No heavy data is loaded here (only the small metadata tables). The sampler
decides *which* 11 wells make up a training batch:

    2 DMSO wells (any plate — cross-plate allowed, plan D1)
    9 chemical wells, one per k9 cluster, concentrations cycling
      0.05 / 0.5 / 5 in cluster order (plan D3 batch rule)

The actual parquet loading happens in ``data/step_generator.py``.
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BatchSpec:
    """The 11 wells (plus their chemistry) that make up one training batch."""

    dmso_samples: list = field(default_factory=list)
    chem_samples: list = field(default_factory=list)
    chem_drugs: list = field(default_factory=list)
    chem_concs: list = field(default_factory=list)
    chem_plates: list = field(default_factory=list)
    dmso_concs: list = field(default_factory=list)
    drug_smiles: dict = field(default_factory=dict)

    def all_samples(self):
        return self.dmso_samples + self.chem_samples

    def all_drugs(self):
        return list(self.drug_smiles.keys())


class TahoeSampler:
    """Select DMSO + chemical wells for training/eval batches."""

    DMSO_DRUG = "DMSO_TF"
    N_CLUSTERS = 9

    def __init__(self, sample_meta_path, drug_json_path, cluster_path,
                 merge_data_dir, seed=0):
        self.df = pd.read_parquet(sample_meta_path)
        self.cluster = (
            pd.read_parquet(cluster_path).set_index("drug")["cluster"].to_dict()
        )
        with open(drug_json_path) as f:
            self.smiles = {d["drug"]: d["canonical_smiles"] for d in json.load(f)}
        self.merge_data_dir = merge_data_dir
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def select_batch(self, n_chemical=9, n_dmso=2,
                     concentrations=(0.05, 0.5, 5.0),
                     exclude_samples=None, rng=None):
        """Return a :class:`BatchSpec` of one training batch.

        ``exclude_samples`` : collection of sample ids to never pick (held-out
        pairs / plates). Sampling is done *without* replacement within a call.
        """
        rng = rng or self.rng
        exclude = set(exclude_samples or [])
        cand = self.df[~self.df["sample"].isin(exclude)]

        dmso_df = cand[cand["drug"] == self.DMSO_DRUG]
        chem_df = cand[cand["drug"] != self.DMSO_DRUG]

        dmso = dmso_df["sample"].to_numpy()
        if len(dmso) < n_dmso:
            raise ValueError(
                f"only {len(dmso)} DMSO wells available after exclusions "
                f"(need {n_dmso})"
            )
        dmso_samples = rng.choice(dmso, size=n_dmso, replace=False).tolist()
        dmso_concs = [
            float(dmso_df.loc[dmso_df["sample"] == s, "drugconc"].iloc[0])
            for s in dmso_samples
        ]

        chem_samples, chem_drugs, chem_concs, chem_plates = [], [], [], []
        for ci in range(self.N_CLUSTERS):
            drugs = sorted(
                d for d, c in self.cluster.items()
                if c == ci and d != self.DMSO_DRUG
            )
            drugs = [d for d in drugs if d in set(chem_df["drug"])]
            if not drugs:
                continue
            drug = rng.choice(drugs)
            conc = float(concentrations[ci % len(concentrations)])
            sample, used_conc = self._pick_well(chem_df, drug, conc, rng)
            chem_samples.append(sample)
            chem_drugs.append(drug)
            chem_concs.append(used_conc)
            chem_plates.append(str(chem_df.loc[
                chem_df["sample"] == sample, "plate"].iloc[0]))

        if len(chem_samples) < n_chemical:
            raise ValueError(
                f"only {len(chem_samples)} chemical wells selected (need {n_chemical})"
            )

        used = dict(self.smiles)
        used = {d: used[d] for d in set(chem_drugs) | {self.DMSO_DRUG}}

        return BatchSpec(
            dmso_samples=dmso_samples,
            chem_samples=chem_samples,
            chem_drugs=chem_drugs,
            chem_concs=chem_concs,
            chem_plates=chem_plates,
            dmso_concs=dmso_concs,
            drug_smiles=used,
        )

    def held_out_samples(self, frac=0.1, rng=None):
        """Random held-out sample ids (pair-based eval)."""
        rng = rng or self.rng
        samples = self.df["sample"].to_numpy()
        n = max(1, int(round(len(samples) * frac)))
        return set(rng.choice(samples, size=n, replace=False).tolist())

    def well_path(self, sample):
        return os.path.join(self.merge_data_dir, f"{sample}.parquet")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _pick_well(self, chem_df, drug, conc, rng):
        """Pick a sample for (drug, conc); fall back to any conc of the drug."""
        sub = chem_df[chem_df["drug"] == drug]
        exact = sub[sub["drugconc"].astype(float) == float(conc)]
        if len(exact) > 0:
            sample = rng.choice(exact["sample"].to_numpy())
            return str(sample), float(conc)
        # fallback: any well of this drug (log the actual concentration used)
        sample = rng.choice(sub["sample"].to_numpy())
        used = float(sub.loc[sub["sample"] == sample, "drugconc"].iloc[0])
        return str(sample), used