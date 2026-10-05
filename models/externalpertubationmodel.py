"""
externalpertubationmodel.py

Top-level model for **external** perturbations (chemicals + proteins).

The shared attention blocks live in ``modelutil.py``; this file only contains the
top-level model that wires them together:

    gene embeddings  -> cross-attention over the initial chemical tokens
    gene expression  -> initial cell state
    final chemicals  -> stacked gene/chemical state attention
    cell coordinates -> per-cell, per-gene output prediction

``ExternalPertubationModel`` is the top-level model.
"""

import torch
import torch.nn as nn

try:  # package import (from model import ...)
    from .modelutil import (
        PerturbationCrossAttention,
        PerturbationStateAttentionStack,
        GeneCellCoordinateIntegrationModel,
    )
except ImportError:  # direct script execution (python externalpertubationmodel.py)
    from modelutil import (
        PerturbationCrossAttention,
        PerturbationStateAttentionStack,
        GeneCellCoordinateIntegrationModel,
    )


class ExternalPertubationModel(nn.Module):
    """Top-level external-perturbation model.

    Parameters
    ----------
    num_of_genes : int
        Size of the gene vocabulary for ``nn.Embedding``.
    emb_dim : int
        Internal embedding / hidden dimension.
    chem_coordi_dim : int
        Dimension of a chemical token's coordinate/structural representation.
    chem_conc_dim : int
        Dimension of the per-chemical concentration feature.
    gene_exp_coordi : int
        Dimension of the gene expression feature fed into the initial state.
    coord_dim : int
        Dimension of the cell coordinate vector (usually 2).
    number_of_heads : int
        Number of attention heads.
    number_of_layers : int
        Number of stacked state-attention blocks.
    """

    def __init__(self, num_of_genes=1000,
                 emb_dim=768,
                 chem_coordi_dim=768,
                 chem_conc_dim=1,
                 gene_exp_coordi=128,
                 coord_dim=2,
                 number_of_heads=8,
                 number_of_layers=4):
        super().__init__()

        self.gene_embedding = nn.Embedding(num_of_genes, emb_dim)

        self.chem_embedding = nn.Linear(chem_coordi_dim + chem_conc_dim, emb_dim)

        self.pert_cross_attention = PerturbationCrossAttention(
            dim=emb_dim, num_heads=number_of_heads
        )

        self.cell_state_layer = nn.Linear(emb_dim + gene_exp_coordi + emb_dim, emb_dim)

        self.pert_state_attention = PerturbationStateAttentionStack(
            dim=emb_dim,
            num_heads=number_of_heads,
            num_layers=number_of_layers,
            dropout=0.1,
        )

        self.GeneCellCoordinateIntegraModel = GeneCellCoordinateIntegrationModel(
            gene_dim=emb_dim, coord_dim=coord_dim
        )

    def forward(self,
                gene_ids,
                gene_exp,
                chem_init,
                chem_init_conc,
                chem_init_mask,
                chem_final,
                chem_final_conc,
                chem_final_mask,
                cell_coordi):
        """
        Parameters
        ----------
        gene_ids : [B, G] long
        gene_exp : [B, G, gene_exp_coordi]
        chem_init : [B, C_init, chem_coordi_dim]
        chem_init_conc : [B, 1]
        chem_init_mask : [B, C_init]
        chem_final : [B, C_final, chem_coordi_dim]
        chem_final_conc : [B, 1]
        chem_final_mask : [B, C_final]
        cell_coordi : [B, N, coord_dim]

        Returns
        -------
        output : [B, N, G]
        """
        gene_emb = self.gene_embedding(gene_ids)

        chem_emb_init = self.chem_embedding(
            torch.cat(
                [chem_init,
                 chem_init_conc.unsqueeze(1).repeat(1, chem_init.shape[1], 1)],
                dim=-1,
            )
        )
        chem_emb_final = self.chem_embedding(
            torch.cat(
                [chem_final,
                 chem_final_conc.unsqueeze(1).repeat(1, chem_final.shape[1], 1)],
                dim=-1,
            )
        )

        gene_pert_output = self.pert_cross_attention(
            gene_emb, chem_emb_init, chem_init_mask
        )

        state = self.cell_state_layer(
            torch.cat([gene_emb, gene_exp, gene_pert_output], dim=-1)
        )

        final_state = self.pert_state_attention(
            gene_emb, state, chem_emb_final, chem_final_mask
        )

        output = self.GeneCellCoordinateIntegraModel(final_state, cell_coordi)

        return output



if __name__ == "__main__":
    # Smoke test with dummy tensors (mirrors the notebook shape checks).
    B = 2
    num_genes = 500
    num_chem_init = 10
    num_chem_final = 30
    chem_dim = 256

    model = ExternalPertubationModel(
        num_of_genes=1000,
        emb_dim=256,
        chem_coordi_dim=256,
        chem_conc_dim=1,
        gene_exp_coordi=128,
        coord_dim=2,
        number_of_heads=1,
        number_of_layers=1,
    )

    gene_id = torch.randint(0, num_genes, (B, num_genes))
    gene_exp = torch.randn(B, num_genes, 128)
    chem_init = torch.randn(B, num_chem_init, chem_dim)
    chem_init_conc = torch.randn(B, 1)
    chem_init_mask = torch.randint(0, 2, (B, num_chem_init)).bool()
    chem_final = torch.randn(B, num_chem_final, chem_dim)
    chem_final_conc = torch.randn(B, 1)
    chem_final_mask = torch.randint(0, 2, (B, num_chem_final)).bool()
    cell_coordi = torch.randn(B, 200, 2)

    output = model(
        gene_id, gene_exp,
        chem_init, chem_init_conc, chem_init_mask,
        chem_final, chem_final_conc, chem_final_mask,
        cell_coordi,
    )
    print("output:", output.shape)  # expect [2, 200, 500]
