"""Shared utilities for loading Tahoe-100M raw parquet wells into AnnData.

The raw per-well parquet files (merge_data/smp_*.parquet) store one row per
cell as ragged parallel lists: ``genes`` (int token IDs) and ``expressions``
(raw float counts). The first entry of every row is a marker/sentinel
(``genes[0] == 1`` / ``expressions[0] == -2.0``) and must be stripped.
Gene token IDs are offset by +3 relative to the 0-based index into the
sorted ``gene_vocabulary.json`` vocabulary, hence the ``- 3`` remap below.
"""

import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import coo_matrix

OBS_COLUMNS = [
    "sample",
    "drug",
    "cell_line_id",
    "BARCODE_SUB_LIB_ID",
    "plate",
    "phase",
    "drugconc",
]

# Per-cell UMAP coordinates, written back into the parquets by
# extra_util/add_cell_coords_to_parquet.py. Only attached to obs when
# present in the file (optional — files not yet processed lack them).
COORD_COLUMNS = ["cell_cordi_1", "cell_cordi_2"]


def _load_gene_vocab_df(gene_vocab_path):
    """Load gene_vocabulary.json into a DataFrame sorted by token index.

    Returns a DataFrame with columns ['gene', 'index'] where the row order
    (ascending token index) defines the 0-based column order of the output
    expression matrix.
    """
    with open(gene_vocab_path, "r") as f:
        gene_vocab = json.load(f)
    return (
        pd.DataFrame(gene_vocab.items(), columns=["gene", "index"])
        .sort_values(by="index")
        .reset_index(drop=True)
    )


def _df_to_adata(df, gene_vocab_df):
    """Convert one raw well DataFrame into a raw-count AnnData.

    Strips the CLS marker/sentinel, remaps gene tokens to 0-based columns,
    drops cells whose gene/expression lists mismatch, and builds a sparse
    cells x n_genes count matrix. obs_names are set to BARCODE_SUB_LIB_ID
    (unique per cell) so adatas from different wells can be concatenated
    without index collisions.
    """
    df = df.copy()
    n_original = len(df)
    df["genes"] = [row[row != 1] for row in df["genes"]]
    df["genes"] = df["genes"] - 3
    df["expressions"] = [row[row != -2.0] for row in df["expressions"]]

    valid = df["genes"].str.len().eq(df["expressions"].str.len())
    df = df.loc[valid]
    if len(df) < n_original:
        sample = df["sample"].iloc[0] if not df.empty else "?"
        raise ValueError(
            f"{sample}: parquet file has {n_original} rows but only "
            f"{len(df)} have matching gene/expression lengths; "
            f"{n_original - len(df)} cell(s) would be silently dropped."
        )

    lengths = df["genes"].str.len().to_numpy()
    row_idx = np.repeat(np.arange(len(df)), lengths)
    gene_idx = np.concatenate(df["genes"].to_numpy())
    values = np.concatenate(df["expressions"].to_numpy())

    X = coo_matrix(
        (values, (row_idx, gene_idx)),
        shape=(len(df), len(gene_vocab_df)),
    ).tocsr()

    obs = df.loc[:, OBS_COLUMNS].copy()

    # Attach per-cell coordinates when the file has them (added by
    # add_cell_coords_to_parquet.py). Missing columns are skipped so files
    # that were never processed still load fine.
    for col in COORD_COLUMNS:
        if col in df.columns:
            obs[col] = df[col].values

    obs.index = obs["BARCODE_SUB_LIB_ID"].astype(str).values

    var = gene_vocab_df[["gene"]].rename(columns={"gene": "gene_name"}).copy()
    var.index = var.index.astype(str)

    return ad.AnnData(X=X, obs=obs, var=var)


def load_parquets_to_adata(parquet_paths, gene_vocab_path, normalize=False, log1p=False):
    """Load one or more raw Tahoe-100M wells into a single AnnData.

    Parameters
    ----------
    parquet_paths : str or list[str]
        Path(s) to merge_data/smp_*.parquet files. A single string is treated
        as a one-element list.
    gene_vocab_path : str
        Path to metadata/gene_vocabulary.json.
    normalize : bool or float, default False
        False = leave raw counts untouched. True = library-size normalize
        each cell to a target sum of 1e4 (scanpy standard). A number is used
        directly as the target sum (e.g. normalize=5e4).
    log1p : bool, default False
        Apply log1p to X after normalization (or to raw counts if normalize
        is False).

    Returns
    -------
    ad.AnnData
        Raw or normalized/log1p counts, cells x genes, with obs annotations
        (sample, drug, cell_line_id, BARCODE_SUB_LIB_ID, plate, phase,
        drugconc, plus cell_cordi_1 / cell_cordi_2 when present in the
        source parquet) and var = gene names.

    Notes
    -----
    Merging many wells can be very memory hungry: all 1344 wells together
    hold ~95.6M cells x 62,710 genes (~1+ TB of sparse data in RAM), so pass
    subsets (a plate, one drug across concentrations, selected samples).
    Normalization and log1p are per-cell operations, so applying them on the
    merged object is identical to per-file normalization.
    """
    if isinstance(parquet_paths, (str, os.PathLike)):
        paths = [parquet_paths]
    else:
        paths = list(parquet_paths)
    if not paths:
        raise ValueError("parquet_paths must contain at least one file path")

    gene_vocab_df = _load_gene_vocab_df(gene_vocab_path)

    adatas = []
    for path in paths:
        df = pd.read_parquet(path)
        adatas.append(_df_to_adata(df, gene_vocab_df))

    if len(adatas) == 1:
        adata = adatas[0]
    else:
        adata = ad.concat(adatas, axis=0, join="inner", merge="same")

    if normalize:
        target_sum = 1e4 if normalize is True else float(normalize)
        sc.pp.normalize_total(adata, target_sum=target_sum)
    if log1p:
        sc.pp.log1p(adata)

    return adata