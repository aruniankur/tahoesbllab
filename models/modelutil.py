"""
modelutil.py

Common utilities and shared neural-network building blocks for the
external- and internal-perturbation models.

Two groups live here:

1. Data / loss helpers (generalized from ``first.ipynb``):
   * ``convert_to_distribution`` – per-cell expression -> per-gene probability
     distributions over expression bins.
   * ``squared_error_loss``      – ``((X1 - X2) ** 2).sum(dim).mean()``.

2. Shared attention modules used by both top-level models:
   * ``PerturbationCrossAttention``             – genes attend over perturbation tokens.
   * ``PerturbationStateAttention``             – one gene/perturbation state block.
   * ``PerturbationStateAttentionStack``        – stack of those blocks.
   * ``GeneCellCoordinateIntegrationModel`` – fuse a gene state with cell coords.

"Chemical token" is read generically as "perturbation token": the internal
model reuses the exact same blocks with gene-knock tokens.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Data / loss helpers
# =====================================================================

def _batched_histogram_counts(X, bin_edges):
    """Per-(batch, gene) histogram counts over cells, matching torch.histogram.

    Computes histograms without a Python loop, so it handles batched inputs and
    large gene sets efficiently. Values below ``bin_edges[0]`` or greater than
    or equal to ``bin_edges[-1]`` are excluded, matching ``torch.histogram``.

    Parameters
    ----------
    X : torch.Tensor
        Shape ``(..., cells, genes)``.
    bin_edges : torch.Tensor
        1D tensor of bin edges of length ``E`` (so ``E - 1`` bins).

    Returns
    -------
    torch.Tensor
        Counts of shape ``(..., genes, E - 1)``.
    """
    n_edges = bin_edges.numel()
    n_bins = n_edges - 1

    lead_shape = X.shape[:-2]
    n_cells = X.shape[-2]
    n_genes = X.shape[-1]
    n_batch = 1
    for d in lead_shape:
        n_batch *= d

    values = X.reshape(n_batch, n_cells, n_genes)

    # right=True => index i means bin_edges[i-1] <= value < bin_edges[i]
    # (1-based). Valid histogram bins are i in [1, n_edges - 1].
    idx = torch.bucketize(values, bin_edges, right=True)  # [N, C, G]
    valid = (idx >= 1) & (idx <= n_bins)
    bin_id = (idx - 1).clamp(0, n_bins - 1)
    bin_id = torch.where(valid, bin_id, n_bins)  # n_bins = sentinel (excluded)

    # Combine (batch, gene, bin) into one flat index and scatter-count.
    batch_ix = torch.arange(n_batch, device=X.device).view(n_batch, 1, 1)
    gene_ix = torch.arange(n_genes, device=X.device).view(1, 1, n_genes)
    flat = (batch_ix * n_genes + gene_ix) * n_bins + bin_id  # [N, C, G]

    counts = torch.bincount(
        flat[valid], minlength=n_batch * n_genes * n_bins
    ).reshape(n_batch, n_genes, n_bins)

    return counts.reshape(*lead_shape, n_genes, n_bins)


def convert_to_distribution(X, steps=129, a=0.0, b=10.0, eps=1e-8, normalize=True):
    """Convert per-cell gene expression into per-gene probability distributions.

    For every (batch, gene), the expression values across cells are histogrammed
    into ``steps`` edges spanning ``[a, b]`` and normalized into a probability
    distribution over the resulting bins.

    Parameters
    ----------
    X : torch.Tensor
        Expression tensor of shape ``(..., cells, genes)``. A 2D
        ``(cells, genes)`` input gives a 2D ``(genes, bins)`` output.
    steps : int
        Number of bin *edges* (output distribution has ``steps - 1`` bins).
        Default 129 -> 128 bins.
    a, b : float
        Lower and upper edges of the histogram range.
    eps : float
        Added to the counts before normalizing, to avoid division by zero for
        genes with no in-range expression.
    normalize : bool
        If True (default), return probabilities (sum to 1 along the last axis).
        If False, return the raw per-bin counts.

    Returns
    -------
    torch.Tensor
        Shape ``(..., genes, steps - 1)``.
    """
    if X.dim() < 2:
        raise ValueError(f"X must have at least 2 dims (..., cells, genes), got {tuple(X.shape)}")

    bin_edges = torch.linspace(a, b, steps, device=X.device, dtype=X.dtype)
    counts = _batched_histogram_counts(X, bin_edges)

    if not normalize:
        return counts

    denom = counts.sum(dim=-1, keepdim=True).to(torch.float32) + eps
    return counts.to(torch.float32) / denom


def squared_error_loss(X1, X2, dim=-1):
    """Squared-error loss: ``((X1 - X2) ** 2).sum(dim).mean()``.

    ``X1``/``X2`` are not treated as separate per-batch losses: the squared
    differences are summed over the feature axis (``dim``) and then **averaged
    over the batch** (all remaining axes), returning a single scalar. For 2D
    ``(cells, genes)`` inputs this is identical to the notebook's
    ``((X1 - X2) ** 2).sum(axis=1).mean()``.

    Parameters
    ----------
    X1, X2 : torch.Tensor
        Tensors of matching shape.
    dim : int
        Axis to sum over before taking the mean. Default -1.

    Returns
    -------
    torch.Tensor
        Scalar tensor.
    """
    if X1.shape != X2.shape:
        raise ValueError(f"Shape mismatch: {tuple(X1.shape)} vs {tuple(X2.shape)}")
    return ((X1 - X2) ** 2).sum(dim=dim).mean()


# Convenience aliases for the data helpers.
loss = squared_error_loss
convert_to_dist = convert_to_distribution


# =====================================================================
# Shared attention building blocks
# =====================================================================

class PerturbationCrossAttention(nn.Module):
    """Multi-head cross-attention where genes (queries) attend over perturbation tokens."""

    def __init__(self, dim=768, num_heads=8):
        super().__init__()

        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.out_proj = nn.Linear(dim, dim)

    def forward(self, gene_emb, pert_emb, mask):
        Q = self.q_proj(gene_emb)
        K = self.k_proj(pert_emb)
        V = self.v_proj(pert_emb)

        B, G, _ = Q.shape
        _, A, _ = K.shape

        Q = Q.view(B, G, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, A, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, A, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = mask.bool()
        mask = mask[:, None, None, :]
        attn_scores = attn_scores.masked_fill(~mask, torch.finfo(attn_scores.dtype).min)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.contiguous().view(B, G, self.dim)

        return self.out_proj(attn_output)


class PerturbationStateAttention(nn.Module):
    """One block of gene/perturbation state attention.

    Steps
    -----
    1. Gene -> Perturbation cross-attention (masked over perturbation tokens).
    2. Concatenate [global_gene_emb, last_state, cross_output].
    3. Gene self-attention over that combination.
    4. Residual from ``last_state``.
    """

    def __init__(self, dim=768, num_heads=8, dropout=0.1):
        super().__init__()

        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Gene -> Chemical Cross-Attention
        self.gene_q = nn.Linear(dim, dim)
        self.pert_k = nn.Linear(dim, dim)
        self.pert_v = nn.Linear(dim, dim)
        self.cross_out = nn.Linear(dim, dim)
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_dropout = nn.Dropout(dropout)

        # Gene Self-Attention
        # Input: [global_gene_emb, last_state, cross_output] -> 3D per gene.
        self.self_qkv = nn.Linear(dim * 3, dim * 3)
        self.self_out = nn.Linear(dim, dim)
        self.self_norm = nn.LayerNorm(dim)
        self.self_dropout = nn.Dropout(dropout)

        # Residual from last_state
        self.residual_proj = nn.Linear(dim, dim)

    def split_heads(self, x):
        """[B, N, D] -> [B, H, N, head_dim]."""
        B, N, D = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def merge_heads(self, x):
        """[B, H, N, head_dim] -> [B, N, D]."""
        B, H, N, D = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, N, H * D)

    def forward(self, global_gene_emb, last_state, perturbation_tokens, perturbation_mask):
        """
        Parameters
        ----------
        global_gene_emb : [B, G, D]
        last_state : [B, G, D]
        perturbation_tokens : [B, C, D]
        perturbation_mask : [B, C]
            1 = valid chemical token, 0 = padding / invalid chemical token.

        Returns
        -------
        output : [B, G, D]
        """
        B, G, D = global_gene_emb.shape

        # 1. Gene -> Perturbation Cross-Attention
        Q = self.gene_q(global_gene_emb)   # queries from genes
        K = self.pert_k(perturbation_tokens)  # keys from perturbation tokens
        V = self.pert_v(perturbation_tokens)  # values from perturbation tokens

        Q = self.split_heads(Q)  # [B, H, G, head_dim]
        K = self.split_heads(K)  # [B, H, C, head_dim]
        V = self.split_heads(V)  # [B, H, C, head_dim]

        # Attention scores: [B, H, G, C]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 2. Perturbation-token mask (1 = valid, 0 = padding)
        perturbation_mask = perturbation_mask.bool()          # [B, C]
        perturbation_mask = perturbation_mask[:, None, None, :]  # [B, 1, 1, C]
        scores = scores.masked_fill(~perturbation_mask, torch.finfo(scores.dtype).min)

        # 3. Softmax over the perturbation-token dimension
        weights = F.softmax(scores, dim=-1)

        # 4. Weighted perturbation values
        cross_output = torch.matmul(weights, V)   # [B, H, G, head_dim]
        cross_output = self.merge_heads(cross_output)  # [B, G, D]

        cross_output = self.cross_out(cross_output)
        cross_output = self.cross_dropout(cross_output)
        cross_output = self.cross_norm(cross_output)

        # 5. Combine gene embedding + previous state + perturbation interaction
        combined = torch.cat([global_gene_emb, last_state, cross_output], dim=-1)

        # 6. Gene self-attention
        Q, K, V = self.self_qkv(combined).chunk(3, dim=-1)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # NOTE: no perturbation mask here; the mask only applies to the
        # gene->perturbation cross-attention above.
        weights = F.softmax(scores, dim=-1)

        output = torch.matmul(weights, V)
        output = self.merge_heads(output)
        output = self.self_out(output)
        output = self.self_dropout(output)

        # 7. Residual from last_state
        output = self.self_norm(output + last_state)
        return output


class PerturbationStateAttentionStack(nn.Module):
    """Stack of ``PerturbationStateAttention`` blocks operating on the cell state."""

    def __init__(self, dim=768, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            PerturbationStateAttention(dim=dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, global_gene_emb, cell_state, perturbation_tokens, perturbation_mask):
        state = cell_state
        for layer in self.layers:
            state = layer(global_gene_emb, state, perturbation_tokens, perturbation_mask)
        return state


class GeneCellCoordinateIntegrationModel(nn.Module):
    """Fuse a per-gene state with per-cell coordinates to produce per-gene outputs.

    Produces a single scalar per (cell, gene) pair via a shared linear layer over
    the concatenated [gene_state, cell_coordinate] features.
    """

    def __init__(self, gene_dim=768, coord_dim=2):
        super().__init__()
        self.model = nn.Linear(gene_dim + coord_dim, 1)
        self.act = nn.ReLU()

    def forward(self, state, coordi):
        B, num_genes, gene_dim = state.shape

        if B != coordi.shape[0]:
            raise ValueError("Batch size of state and coordi must match.")

        # state:  (B, G, gene_dim)
        # coordi: (B, N, coord_dim)

        state = state.unsqueeze(1)      # (B, 1, G, gene_dim)
        coordi = coordi.unsqueeze(2)    # (B, N, 1, coord_dim)

        # Broadcast across cells and genes:
        state = state.expand(-1, coordi.shape[1], -1, -1)
        coordi = coordi.expand(-1, -1, num_genes, -1)

        all_info = torch.cat([state, coordi], dim=-1)  # (B, N, G, gene_dim + coord_dim)
        output = self.model(all_info).squeeze(-1)      # (B, N, G)
        output = self.act(output)
        return output



if __name__ == "__main__":
    torch.manual_seed(0)

    # ---- data / loss helpers ----
    X = torch.rand(400, 50) * 12 - 1  # some values outside [0, 10]
    edges = torch.linspace(0, 10, 129)
    ref = torch.stack([torch.histogram(X[:, g], bins=edges)[0] for g in range(X.shape[1])])
    ref = ref / ref.sum(dim=-1, keepdim=True)
    got = convert_to_distribution(X)
    print("distribution 2D shape:", tuple(got.shape))
    print("matches notebook loop:", torch.allclose(got, ref, atol=1e-6))

    Xb = torch.rand(2, 400, 50) * 10
    db = convert_to_distribution(Xb)
    print("distribution batched shape:", tuple(db.shape))
    print("batch matches per-sample:", torch.allclose(db[0], convert_to_distribution(Xb[0]), atol=1e-6))

    Y = X + torch.randn_like(X) * 0.1
    print("2D loss == notebook expr:", torch.allclose(
        squared_error_loss(X, Y), ((X - Y) ** 2).sum(axis=1).mean()))

    # ---- shared attention blocks ----
    B, G, A, D = 2, 50, 10, 32
    gene_emb = torch.randn(B, G, D)
    tokens = torch.randn(B, A, D)
    mask = torch.ones(B, A, dtype=torch.bool)
    cross = PerturbationCrossAttention(dim=D, num_heads=4)(gene_emb, tokens, mask)
    state = torch.randn(B, G, D)
    stack = PerturbationStateAttentionStack(dim=D, num_heads=4, num_layers=2)
    out = stack(gene_emb, state, tokens, mask)
    coord = torch.randn(B, 20, 2)
    final = GeneCellCoordinateIntegrationModel(gene_dim=D, coord_dim=2)(out, coord)
    print("cross:", tuple(cross.shape), "stack:", tuple(out.shape), "final:", tuple(final.shape))
