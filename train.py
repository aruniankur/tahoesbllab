"""train.py

Train the Tahoe-100M virtual-cell model.

Usage:
    python train.py                    # full config
    python train.py --smoke            # tiny, fast memory-safe run (1 epoch)
    python train.py --epochs 5 --device cpu
    python train.py --config path/to/config.yaml

Design (see docs/model_trainingplan.md):
  - one step = 1 cell line + n_chem target chemicals (+ n_dmso DMSO controls)
  - one 11-well batch -> 50 line adatas -> up to 50 steps per epoch
  - epoch = each usable cell line once
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.sampler import TahoeSampler                      # noqa: E402
from data.step_generator import TahoeStepGenerator         # noqa: E402
from models.externalpertubationmodel import ExternalPertubationModel  # noqa: E402
from models.modelutil import squared_error_loss            # noqa: E402
from utils.config import to_attr_dict                      # noqa: E402
from tqdm import tqdm                                      # noqa: E402


def load_config(path):
    with open(path) as f:
        cfg = to_attr_dict(yaml.safe_load(f))
    base = os.path.dirname(os.path.abspath(path))
    for k, v in cfg["paths"].items():
        if not os.path.isabs(v):
            cfg["paths"][k] = os.path.join(base, v)
    return cfg


def resolve_device(device):
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def apply_smoke(cfg):
    s = cfg["smoke"]
    cfg["data"].update(
        max_cells_per_well=s["max_cells_per_well"],
        max_genes=s["max_genes"],
        n_coord_cells=s["n_coord_cells"],
    )
    cfg["step"].update(n_chem=s["n_chem"], n_dmso=s["n_dmso"])
    cfg["model"].update(
        emb_dim=s["emb_dim"], number_of_heads=s["number_of_heads"],
        number_of_layers=s["number_of_layers"],
    )
    cfg["train"].update(
        epochs=s["epochs"], device=s["device"], log_every=s["log_every"]
    )
    cfg["smoke"]["max_steps"] = s.get("max_steps", 8)
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke(cfg)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.device is not None:
        cfg["train"]["device"] = args.device
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed

    seed_all(cfg["train"]["seed"])
    device = resolve_device(cfg["train"]["device"])
    amp = str(cfg["train"].get("amp", "bf16")).lower()
    use_amp = device.startswith("cuda") and amp != "none"
    amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    print(f"[train] device={device} config={args.config} smoke={args.smoke} "
          f"amp={amp if use_amp else 'fp32'}")

    sampler = TahoeSampler(
        cfg["paths"]["sample_metadata"],
        cfg["paths"]["drug_json"],
        cfg["paths"]["cluster_labels"],
        cfg["paths"]["merge_data"],
        seed=cfg["sampling"]["seed"],
    )
    gen = TahoeStepGenerator(cfg)

    model = ExternalPertubationModel(**cfg["model"])
    model.to(device)
    print(f"[train] model params={sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if (use_amp and amp == "fp16") else None
    )
    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)

    rng = np.random.default_rng(cfg["train"]["seed"])
    steps_done = 0
    max_steps = cfg["smoke"].get("max_steps") if args.smoke else None

    for epoch in tqdm(range(1, cfg["train"]["epochs"] + 1), desc="epoch"):
        sel = tqdm(total=1, desc=f"epoch {epoch} - select chemicals", leave=False)
        spec = sampler.select_batch(
            n_chemical=cfg["sampling"]["n_chemical_wells"],
            n_dmso=cfg["sampling"]["n_dmso_wells"],
            concentrations=cfg["sampling"]["concentrations"],
        )
        sel.update(1)
        sel.close()

        load = tqdm(
            spec.all_samples(),
            desc=f"epoch {epoch} - load parquet & convert",
            unit="well",
            leave=False,
        )
        batch = gen.load_batch(spec, progress=load)
        load.close()

        lines = list(batch.lines)
        rng.shuffle(lines)
        pbar = tqdm(lines, desc=f"epoch {epoch} - training", unit="step", leave=False)
        for line in pbar:
            try:
                step = gen.make_step(batch, line, rng=rng)
            except ValueError as e:
                tqdm.write(f"  [skip] {line}: {e}")
                continue

            step = gen.to_device(step, device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                pred = model(**{k: v for k, v in step.items() if k not in ("target", "meta")})
            pred_mean = pred.float().mean(dim=1)             # [B, G] fp32
            loss = squared_error_loss(pred_mean, step["target"])

            if not torch.isfinite(loss):
                bad = {k: bool((v != v).any()) for k, v in step.items()
                       if isinstance(v, torch.Tensor)}
                tqdm.write(
                    f"  [skip nan] {line}: task={step['meta']['task']} "
                    f"G={step['meta']['G']} B={pred.shape[0]} "
                    f"loss={loss.item()} nan_tensors={bad}"
                )
                opt.zero_grad()
                continue

            opt.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["train"]["grad_clip"])
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["train"]["grad_clip"])
                opt.step()
            steps_done += 1

            pbar.set_postfix_str(
                f"loss={loss.item():.3f} task={step['meta']['task']} "
                f"G={step['meta']['G']} B={pred.shape[0]}"
            )

            if max_steps and steps_done >= max_steps:
                tqdm.write(f"[train] reached smoke max_steps={max_steps}; stopping")
                ckpt = os.path.join(cfg["paths"]["checkpoint_dir"], "smoke.pt")
                torch.save({
                    "epoch": epoch, "steps": steps_done,
                    "model": model.state_dict(), "optimizer": opt.state_dict(),
                    "config": cfg,
                }, ckpt)
                tqdm.write(f"[train] checkpoint saved: {ckpt}")
                return

        if (epoch + 1) % cfg["train"]["save_every"] == 0 or \
                epoch == cfg["train"]["epochs"] - 1:
            ckpt = os.path.join(
                cfg["paths"]["checkpoint_dir"], f"epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch, "steps": steps_done,
                "model": model.state_dict(), "optimizer": opt.state_dict(),
                "config": cfg,
            }, ckpt)
            tqdm.write(f"[train] checkpoint saved: {ckpt}")

    tqdm.write(f"[train] done. steps={steps_done}")


if __name__ == "__main__":
    main()