"""
Figures showing what the learned homoscedastic sigmas correspond to.

Three claims, one figure each:
  fig1_core_vs_tail.png    per-task residual histograms with learned sigma, the
                           robust core (16-84 half-spread) and the RMS marked.
                           Shows the learned sigma tracks the CORE, while the
                           RMS is inflated by the tail.
  fig2_sigma_ladder.png    per-task bars of {learned sigma, sigma*=sqrt(2<Huber>),
                           RMS, robust core}, each normalised to that task's
                           robust core so the four tasks share one axis. Shows
                           learned sigma ~ core (near 1) and RMS towering above.
  fig3_region_averaging.png  per-task barrel-core vs endcap-core with the single
                           learned sigma drawn as one line between them: a scalar
                           homoscedastic sigma cannot represent the region split.

All residuals are in each task's LOSS/working units so they are directly
comparable to the learned sigma read from the checkpoint:
    eta   raw pseudorapidity        (Huber delta 0.10)
    phi   radians, wrapped          (delta 0.05)
    logpt ln(pT), i.e. fractional   (delta 0.20)
    z0    z-scored (mm / z0_std)     (delta 1.00)

Usage (env-var invocation, matching the repo convention):
    env PREDS=<run>/preds.npz \
        CKPT=<run>/checkpoints/ruche_eta_phi_pt_z0_charge.pt \
        STATS_PATH=/pbs/home/l/llambert/data/target_stats.json \
        OUTDIR=results/uncertainty_figs \
        python scripts/plot_uncertainty_vs_resolution.py

CKPT is optional; if omitted (or it has no loss_state_dict), the learned-sigma
overlays are skipped and only the measured resolutions are drawn.
"""

import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASKS = ["eta", "phi", "logpt", "z0"]
LABELS = {"eta": r"$\eta$ residual",
          "phi": r"$\phi$ residual [rad]",
          "logpt": r"$\ln p_T$ residual",
          "z0": r"$z_0$ residual [z-scored]"}
DELTAS = {"eta": 0.10, "phi": 0.05, "logpt": 0.20, "z0": 1.00}


def wrap(dphi):
    return np.arctan2(np.sin(dphi), np.cos(dphi))


def robust_core(e):
    q16, q84 = np.percentile(e, [16.0, 84.0])
    return 0.5 * float(q84 - q16)


def rms(e):
    return float(np.sqrt(np.mean(e * e)))


def huber(e, d):
    a = np.abs(e)
    return np.where(a <= d, 0.5 * e * e, d * (a - 0.5 * d))


def sigma_star(e, d):
    return float(np.sqrt(2.0 * huber(e, d).mean()))


def load_residuals(preds_path, z0_std):
    d = np.load(preds_path)
    return {
        "eta": d["pred_eta"] - d["truth_eta"],
        "phi": wrap(d["pred_phi"] - d["truth_phi"]),
        "logpt": np.log(d["pred_pt"]) - np.log(d["truth_pt"]),
        "z0": (d["pred_z0"] - d["truth_z0"]) / z0_std,
    }, np.abs(d["truth_eta"])


def load_learned_sigma(ckpt_path):
    if not ckpt_path or not Path(ckpt_path).exists():
        return None
    import torch
    ls = torch.load(ckpt_path, map_location="cpu").get("loss_state_dict", {})
    if "log_sigma" not in ls:
        return None
    vals = np.exp(ls["log_sigma"].numpy())
    return {t: float(v) for t, v in zip(TASKS, vals)}


def fig_core_vs_tail(res, learned, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, t in zip(axes.ravel(), TASKS):
        e = res[t]
        core, r = robust_core(e), rms(e)
        # window must show the peak AND the RMS marker (which can sit far in the
        # tail); go a little past whichever is larger.
        lim = 1.25 * max(4.0 * core, r)
        ax.hist(np.clip(e, -lim, lim), bins=120, color="0.75",
                edgecolor="none", log=True)
        ax.axvline(core, color="C0", lw=1.8, label=f"robust core = {core:.4f}")
        ax.axvline(-core, color="C0", lw=1.8)
        ax.axvline(r, color="C3", lw=1.8, ls="--", label=f"RMS = {r:.4f}")
        ax.axvline(-r, color="C3", lw=1.8, ls="--")
        if learned is not None:
            s = learned[t]
            ax.axvspan(-s, s, color="C2", alpha=0.18,
                       label=f"learned $\\sigma$ = {s:.4f}")
        ax.set_xlabel(LABELS[t])
        ax.set_ylabel("electrons / bin (log)")
        ax.set_xlim(-lim, lim)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Learned homoscedastic sigma tracks the core, not the RMS",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = outdir / "fig1_core_vs_tail.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_sigma_ladder(res, learned, outdir):
    # normalise every quantity to the task's robust core -> shared y-axis
    cores = {t: robust_core(res[t]) for t in TASKS}
    series = {
        "learned $\\sigma$": [learned[t] / cores[t] if learned else np.nan for t in TASKS],
        "sigma* = sqrt(2<Huber>)":
            [sigma_star(res[t], DELTAS[t]) / cores[t] for t in TASKS],
        "RMS": [rms(res[t]) / cores[t] for t in TASKS],
    }
    x = np.arange(len(TASKS))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["C2", "C1", "C3"]
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + (i - 1) * w, vals, w, label=name, color=colors[i])
    ax.axhline(1.0, color="C0", lw=1.5,
               label="robust core (reference = 1)")
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS)
    ax.set_ylabel("scale / robust core")
    ax.set_yscale("log")
    ax.set_title("Each scale relative to the core: learned sigma sits on the "
                 "core, RMS is tail-inflated")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = outdir / "fig2_sigma_ladder.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_region_averaging(res, abs_eta, learned, outdir):
    barrel = abs_eta < 1.5
    endcap = (abs_eta >= 1.5) & (abs_eta < 3.0)
    x = np.arange(len(TASKS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # normalise each task to its own barrel core so all four are legible on one
    # axis (working units differ per task); the point is the relative gap.
    b_abs = {t: robust_core(res[t][barrel]) for t in TASKS}
    b = [1.0 for _ in TASKS]
    e = [robust_core(res[t][endcap]) / b_abs[t] for t in TASKS]
    ax.bar(x - w / 2, b, w, label="barrel core", color="C0")
    ax.bar(x + w / 2, e, w, label="endcap core", color="C1")
    if learned is not None:
        for i, t in enumerate(TASKS):
            ax.hlines(learned[t] / b_abs[t], x[i] - w, x[i] + w, color="C2",
                      lw=2.5,
                      label="learned $\\sigma$ (single scalar)" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(TASKS)
    ax.set_ylabel("resolution / barrel core")
    ax.set_title("One homoscedastic sigma averages over the barrel/endcap split")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = outdir / "fig3_region_averaging.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main():
    preds_path = Path(os.environ["PREDS"])
    stats_path = Path(os.environ.get(
        "STATS_PATH", "data/electrons/eta_phi_pt_z0_charge/target_stats.json"))
    ckpt_path = os.environ.get("CKPT", "")
    outdir = Path(os.environ.get("OUTDIR", "results/uncertainty_figs"))
    outdir.mkdir(parents=True, exist_ok=True)

    z0_std = float(json.loads(stats_path.read_text())["truth_z0"]["std"])
    res, abs_eta = load_residuals(preds_path, z0_std)
    learned = load_learned_sigma(ckpt_path)

    if learned is None:
        print("[warn] no learned sigma (CKPT missing or has no loss_state_dict); "
              "drawing measured resolutions only.")
    else:
        print("learned sigma:", {t: round(learned[t], 5) for t in TASKS})

    p1 = fig_core_vs_tail(res, learned, outdir)
    p2 = fig_sigma_ladder(res, learned, outdir)
    p3 = fig_region_averaging(res, abs_eta, learned, outdir)
    for p in (p1, p2, p3):
        print("wrote", p)


if __name__ == "__main__":
    main()