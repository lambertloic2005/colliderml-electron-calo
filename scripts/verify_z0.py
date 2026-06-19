"""
One-off smoke test for the z0 / uncertainty-weighting edits.

Run from the repo root:
    python scripts/verify_z0_changes.py
    python scripts/verify_z0_changes.py --stats path/to/target_stats.json

It imports the REAL KinematicLoss from the training script and checks:
  1. log_sigma is a trainable parameter picked up by loss_fn.parameters()
  2. the total loss is finite
  3. gradients flow to the z0 head (pred column 4)
  4. gradients flow to log_sigma (so the optimizer can actually move it)
  5. eta/phi/pt heads still receive gradients
It does NOT need the dataset, a checkpoint, or a GPU.
"""

import argparse
import importlib.util
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
DEFAULT_STATS = REPO / "data/electrons/eta_phi_pt_z0_charge/target_stats.json"


def load_kinematic_loss():
    path = REPO / "scripts" / "train_eta_phi_pt_z0_charge.py"
    spec = importlib.util.spec_from_file_location("trainmod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # runs top-level imports (torch, wandb, package)
    return mod.KinematicLoss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    if not args.stats.exists():
        raise FileNotFoundError(
            f"target_stats.json not found at {args.stats}; pass --stats <path>"
        )

    KinematicLoss = load_kinematic_loss()
    loss_fn = KinematicLoss(target_stats_path=args.stats)

    # --- check 1: log_sigma is a registered, trainable parameter ---
    params = dict(loss_fn.named_parameters())
    assert "log_sigma" in params, "log_sigma is NOT a parameter of the loss"
    assert params["log_sigma"].numel() == 4, "log_sigma should have 4 entries"
    assert params["log_sigma"].requires_grad, "log_sigma must require grad"
    print("[1] log_sigma is a trainable parameter, shape",
          tuple(params["log_sigma"].shape))
    print("    initial per-task precision exp(-2*log_sigma):",
          torch.exp(-2.0 * params["log_sigma"].detach()).tolist())

    # report the beamspot scale the model is regressing against
    print(f"    z0 stats: mean={loss_fn.z0_mean.item():.2f} mm  "
          f"std={loss_fn.z0_std.item():.2f} mm  "
          f"(this std ~= the z0_prior_rmse you should expect)")

    # --- synthetic batch ---
    B = args.batch
    pred = torch.randn(B, 5, requires_grad=True)         # [deta, dphi_e, dphi_p, dlogpt, dz0]
    target = torch.randn(B, 4)                           # [eta, phi, logpt, z0]  (normalized)
    phi_c = torch.zeros(B)
    eta_c = torch.zeros(B)
    log_sum_et = torch.zeros(B)
    z0_anchor = torch.randn(B) * 1000.0                  # deliberately huge & noisy
    charge = torch.sign(torch.randn(B))
    charge[charge == 0] = 1.0

    total, logs = loss_fn(pred, target, phi_c, eta_c, log_sum_et, z0_anchor, charge)

    # --- check 2: finite loss ---
    assert torch.isfinite(total), f"total loss is not finite: {total}"
    print(f"[2] total loss is finite: {total.item():.4f}")

    total.backward()

    # --- check 3: z0 head gets gradient ---
    gz0 = pred.grad[:, 4]
    assert torch.isfinite(gz0).all() and gz0.abs().sum() > 0, \
        "z0 head (pred[:,4]) received NO gradient"
    print(f"[3] z0 head receives gradient: |grad| mean = {gz0.abs().mean():.3e}")

    # --- check 4: log_sigma gets gradient (optimizer can move it) ---
    gls = params["log_sigma"].grad
    assert gls is not None and torch.isfinite(gls).all() and gls.abs().sum() > 0, \
        "log_sigma received NO gradient (did you add loss_fn.parameters() to AdamW?)"
    print(f"[4] log_sigma receives gradient: {gls.tolist()}")

    # --- check 5: eta/phi/pt heads still get gradient ---
    for i, nm in [(0, "eta"), (1, "phi_e"), (2, "phi_p"), (3, "logpt")]:
        g = pred.grad[:, i]
        assert g.abs().sum() > 0, f"{nm} head received no gradient"
    print("[5] eta / phi / logpt heads all receive gradient")

    # --- check 6: z0_rmse_mm diagnostic is in mm (not normalized) ---
    print(f"[6] z0_rmse_mm diagnostic on random batch = "
          f"{logs['z0_rmse_mm'].item():.1f} mm "
          f"(should be ~z0_std scale, i.e. mm not ~1)")

    print("\nALL CHECKS PASSED — the loss wiring is correct.")
    print("Next: retrain, then point the test checkpoint_path at the new .pt file.")


if __name__ == "__main__":
    main()