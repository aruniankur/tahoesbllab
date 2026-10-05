"""evaluate.py

Evaluate the trained model on held-out pairs.

For each cell line x chemical in a held-out batch:
    source = DMSO cells of that line (mean log1p = control baseline)
    target = the chemical's actual mean log1p counts

Metrics per cell line (aggregated at the end):
    pearson r        corr(predicted_mean, actual_mean)          over genes
    delta recovery   corr(pred - control, actual - control)     (must be > 0)
    identity baseline corr(control, actual)                     (the bar to beat)

Usage:
    python evaluate.py [--smoke] [--checkpoint checkpoints/epoch_9.pt]
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.sampler import TahoeSampler                      # noqa: E402
from data.step_generator import TahoeStepGenerator         # noqa: E402
from models.externalpertubationmodel import ExternalPertubationModel  # noqa: E402
from train import apply_smoke, load_config, resolve_device, seed_all  # noqa: E402


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((a * b).sum() / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke(cfg)
    if args.device is not None:
        cfg["train"]["device"] = args.device

    seed_all(cfg["train"]["seed"])
    device = resolve_device(cfg["train"]["device"])
    amp = str(cfg["train"].get("amp", "bf16")).lower()
    use_amp = device.startswith("cuda") and amp != "none"
    amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16

    sampler = TahoeSampler(
        cfg["paths"]["sample_metadata"],
        cfg["paths"]["drug_json"],
        cfg["paths"]["cluster_labels"],
        cfg["paths"]["merge_data"],
        seed=cfg["sampling"]["seed"] + 1,
    )
    gen = TahoeStepGenerator(cfg)

    model = ExternalPertubationModel(**cfg["model"])
    model.to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device,
                           weights_only=False)
        model.load_state_dict(state["model"])
        print(f"[eval] loaded checkpoint: {args.checkpoint} (epoch {state['epoch']})")
    model.eval()

    held_out = sampler.held_out_samples(frac=0.1)
    spec = sampler.select_batch(
        n_chemical=cfg["sampling"]["n_chemical_wells"],
        n_dmso=cfg["sampling"]["n_dmso_wells"],
        concentrations=cfg["sampling"]["concentrations"],
        exclude_samples=held_out,
        rng=np.random.default_rng(42),
    )
    batch = gen.load_batch(spec)
    print(f"[eval] batch: {len(batch.lines)} lines, {len(spec.chem_samples)} chemicals")

    rng = np.random.default_rng(7)
    rows = []
    with torch.no_grad():
        for line in batch.lines:
            line_ad = batch.line_adata[line]
            dmso_mask = np.isin(line_ad.obs["sample"].values, spec.dmso_samples)
            if dmso_mask.sum() == 0:
                continue
            control = torch.as_tensor(
                np.asarray(line_ad.X[dmso_mask].log1p().mean(axis=0)).ravel(),
                dtype=torch.float32,
            )

            for t in spec.chem_samples:
                if (line, t) not in batch.targets:
                    continue
                step = gen.make_step(batch, line, task="control_chem",
                                     rng=rng, n_chem=1, n_dmso=1)
                step = gen.to_device(step, device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    pred = model(**{k: v for k, v in step.items()
                                    if k not in ("target", "meta")})
                pred_mean = pred.float().mean(dim=1)[0].cpu().numpy()   # [G]
                actual = step["target"][0].cpu().numpy()        # [G]
                ctrl = control.numpy()

                rows.append({
                    "line": line,
                    "drug": spec.chem_drugs[spec.chem_samples.index(t)],
                    "conc": spec.chem_concs[spec.chem_samples.index(t)],
                    "pearson": pearson(pred_mean, actual),
                    "delta": pearson(pred_mean - ctrl, actual - ctrl),
                    "identity": pearson(ctrl, actual),
                })

    print(f"\n{'cell_line':<12}{'pearson':>10}{'delta':>10}{'identity':>10}")
    for r in rows:
        print(f"{r['line']:<12}{r['pearson']:>10.4f}{r['delta']:>10.4f}"
              f"{r['identity']:>10.4f}")

    if rows:
        for k in ("pearson", "delta", "identity"):
            vals = [r[k] for r in rows if not np.isnan(r[k])]
            print(f"mean {k:<9}: {np.mean(vals):.4f}  (n={len(vals)})")
    else:
        print("[eval] no valid (line, chemical) pairs evaluated")


if __name__ == "__main__":
    main()