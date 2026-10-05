"""models/utils.py

Reusable inference helpers for the perturbation models.

Currently provides:

* ``smiles_to_molformer`` – run MoLFormer over a list of SMILES and return
  per-atom token embeddings + attention masks. Model and tokenizer are loaded
  lazily and cached module-level, so repeated calls reuse the same instance.

Verified against the locally cached ``ibm-research/MoLFormer-XL-both-10pct``
checkpoint (transformers 5.17, trust_remote_code): ``last_hidden_state`` has
shape ``[B, seq_len, 768]`` and ``attention_mask`` is ``[B, seq_len]`` with
``1`` for real tokens and ``0`` for padding.
"""

import os

import torch
from transformers import AutoModel, AutoTokenizer

MOLFORMER_MODEL_ID = os.environ.get(
    "MOLFORMER_MODEL_ID", "ibm-research/MoLFormer-XL-both-10pct"
)

_LOADED = {"tokenizer": None, "model": None, "device": None}


def _ensure_molformer_compat():
    """Patch ``transformers.masking_utils.create_bidirectional_mask`` if missing.

    The cached MoLFormer remote module (``modeling_molformer.py``) imports this
    old BERT-style helper from ``transformers.masking_utils``; it was removed in
    transformers >= 4.53. Inject a drop-in replacement so the same code works on
    any transformers version (local dev + L40 server).
    """
    try:
        from transformers.masking_utils import (  # noqa: F401
            create_bidirectional_mask,
        )
        return
    except ImportError:
        pass

    import transformers.masking_utils as mu

    def create_bidirectional_mask(config, inputs_embeds, attention_mask,
                                  *args, **kwargs):
        """Old BERT-style bidirectional extended mask: [B, S] -> [B, 1, 1, S].

        0.0 at kept positions, ``torch.finfo(dtype).min`` at masked positions
        (MoLFormer self-attention later tests ``mask == 0`` to recover 1/0).
        """
        dtype = getattr(config, "torch_dtype", None) or inputs_embeds.dtype
        if attention_mask.dim() == 3:
            extended = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for attention_mask (shape {attention_mask.shape})"
            )
        extended = extended.to(dtype=dtype)
        return (1.0 - extended) * torch.finfo(dtype).min

    mu.create_bidirectional_mask = create_bidirectional_mask


def _lazy_load(device):
    """Load (once) and cache the MoLFormer tokenizer + model on ``device``."""
    if (
        _LOADED["tokenizer"] is None
        or _LOADED["model"] is None
        or _LOADED["device"] != device
    ):
        _ensure_molformer_compat()
        _LOADED["tokenizer"] = AutoTokenizer.from_pretrained(
            MOLFORMER_MODEL_ID, trust_remote_code=True
        )
        _LOADED["model"] = AutoModel.from_pretrained(
            MOLFORMER_MODEL_ID, trust_remote_code=True
        )
        _LOADED["model"].to(device)
        _LOADED["model"].eval()
        _LOADED["device"] = device
    return _LOADED["tokenizer"], _LOADED["model"]


def _resolve_device(device):
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    # MPS is intentionally skipped: MoLFormer's Performer attention uses
    # torch.linalg.qr, which is not implemented on MPS. Pass device="mps"
    # explicitly (with PYTORCH_ENABLE_MPS_FALLBACK=1) if you really want it.
    return "cpu"


def smiles_to_molformer(
    smiles,
    device=None,
    batch_size=32,
    pooler=False,
    strip_special=True,
):
    """Encode SMILES with MoLFormer into per-atom embeddings and masks.

    Parameters
    ----------
    smiles : str or list[str]
        One or more SMILES strings. A single string is treated as a
        one-element batch.
    device : str or torch.device, optional
        Device to run on. Defaults to auto-detection (cuda, else cpu — MPS is
        skipped because MoLFormer uses ``torch.linalg.qr``, unsupported there).
    batch_size : int, default 32
        Number of molecules per inference batch (bounded memory for long runs).
    pooler : bool, default False
        If True, also return the pooled [CLS] vector (``pooler_output``,
        shape ``[B, 768]``) as a third return value.
    strip_special : bool, default True
        If True, drop the ``<bos>`` (first) and ``<eos>`` (last real) tokens so
        the sequence contains only atom tokens (plan decision O6).

    Returns
    -------
    embeddings : torch.Tensor
        Per-atom token embeddings, shape ``[B, T, 768]`` (``last_hidden_state``).
    masks : torch.Tensor
        Bool attention mask, shape ``[B, T]``; ``True`` = real atom, ``False``
        = padding. Compatible with ``PerturbationCrossAttention``.
    pooled : torch.Tensor, optional
        Only if ``pooler=True``. Pooled [CLS] vector, shape ``[B, 768]``.

    Notes
    -----
    Tensors are returned on the active device. The MoLFormer checkpoint is
    loaded lazily and cached, so the first call is slow and later calls reuse
    the loaded instance.
    """
    if isinstance(smiles, str):
        smiles = [smiles]
    if not smiles:
        raise ValueError("smiles must contain at least one SMILES string")

    device = _resolve_device(device)
    tokenizer, model = _lazy_load(device)

    all_embs, all_masks, all_pooled = [], [], []
    with torch.no_grad():
        for start in range(0, len(smiles), batch_size):
            batch = smiles[start : start + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,  # MolformerModel.forward has no token_type_ids
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            out = model(**enc)
            emb = out.last_hidden_state  # [B, T, 768]
            mask = enc["attention_mask"].bool()  # [B, T]

            if strip_special:
                emb = emb[:, 1:, :]
                mask = mask[:, 1:]
                lens = mask.sum(dim=1)  # now points at <eos> (last real token)
                seq_len = mask.shape[1]
                arange = torch.arange(seq_len, device=device).unsqueeze(0)
                is_eos = arange == (lens - 1).unsqueeze(1)
                emb = emb.masked_fill(is_eos.unsqueeze(-1), 0.0)
                mask = mask & ~is_eos

            all_embs.append(emb)
            all_masks.append(mask)
            if pooler:
                all_pooled.append(out.pooler_output)

    embeddings = torch.cat(all_embs, dim=0)
    masks = torch.cat(all_masks, dim=0)

    if pooler:
        pooled = torch.cat(all_pooled, dim=0)
        return embeddings, masks, pooled
    return embeddings, masks


if __name__ == "__main__":
    torch.manual_seed(0)

    test_smiles = [
        "CS(=O)C",  # DMSO
        "B(C(CC(C)C)NC(=O)C(CC1=CC=CC=C1)NC(=O)C2=NC=CN=C2)(O)O",  # Bortezomib
    ]

    emb, mask = smiles_to_molformer(test_smiles, strip_special=True)
    print("embeddings:", tuple(emb.shape), "| masks:", tuple(mask.shape))
    print("mask dtype:", mask.dtype)

    assert emb.ndim == 3 and emb.shape[-1] == 768
    assert mask.dtype == torch.bool
    assert emb.shape[0] == mask.shape[0] == len(test_smiles)
    assert emb.shape[1] == mask.shape[1]

    # DMSO "CS(=O)C" -> 9 real tokens (incl. <bos>/<eos>), stripped -> 7.
    # Tokens are atom-level SMILES tokens; symbols like ( = ) are separate.
    assert mask[0].sum().item() == 7, f"expected 7 DMSO tokens, got {mask[0].sum().item()}"
    # Both molecules share the padded width of the batch.
    assert emb.shape[1] == mask[1].sum().item() + (mask[1] == 0).sum().item()

    emb2, mask2, pooled = smiles_to_molformer(
        test_smiles, pooler=True, strip_special=False
    )
    print("pooled:", tuple(pooled.shape))
    assert pooled.shape == (2, 768)
    assert emb2.shape[1] == mask2[1].sum().item()  # no padding gaps when keeping bos/eos

    # Single-string input wraps to a 1-element batch.
    emb3, mask3 = smiles_to_molformer("CS(=O)C")
    assert emb3.shape[0] == 1

    print("smoke test passed")