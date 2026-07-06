"""Test whether the eta residual is dominated by the unknown vertex z.

Physics: truth eta is defined from the momentum at the PRODUCTION VERTEX,
while the calorimeter measures the shower POSITION. Converting a cluster
position (r, z_cluster) into a vertex-referenced eta requires knowing the
vertex z. To first order,

    d(eta) / d(z_vertex) = -1 / (r_eff * cosh(eta)),

so an error dz in the model's implicit vertex estimate produces an eta error
    d(eta) ~= -dz / (r_eff * cosh(eta)).

If eta is vertex-dominated, the eta and z0 residuals should be strongly
ANTI-correlated, the fitted slope d(eta_res)/d(z0_res) in each |eta| bin
should equal -1/(r_eff * cosh(eta)) with a SINGLE consistent r_eff
(~ the ECal barrel radius, in mm), and subtracting the fitted z0 component
should collapse the eta sigma toward the intrinsic position resolution.

Usage:
    python scripts/diagnose_eta_z0_coupling.py results/<run>/preds.npz

Requires preds.npz saved with truth_z0 / pred_z0 (see extended np.savez in
test_eta_phi_pt_z0_charge.py).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from colliderml_electron.resolution import gaussian_resolution  # noqa: E402


def main() -> None:
    preds_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("preds.npz")
    d = np.load(preds_path)
    for key in ("truth_eta", "pred_eta", "truth_z0", "pred_z0"):
        if key not in d.files:
            raise KeyError(
                f"{key!r} missing from {preds_path}. Re-run the test script "
                "with the extended np.savez (truth_z0/pred_z0 added)."
            )

    eta_res = d["pred_eta"] - d["truth_eta"]
    z0_res = d["pred_z0"] - d["truth_z0"]          # mm
    abs_eta = np.abs(d["truth_eta"])

    rho = float(np.corrcoef(eta_res, z0_res)[0, 1])
    print(f"n = {len(eta_res)}")
    print(f"global Pearson rho(eta_res, z0_res) = {rho:+.3f}")
    print("(vertex-dominated eta predicts a strong NEGATIVE correlation)\n")

    print(f"{'|eta| bin':>12s} {'n':>6s} {'rho':>7s} {'slope [1/mm]':>13s} "
          f"{'r_eff [mm]':>11s} {'sig_eta':>8s} {'sig_eta|z0':>10s}")
    edges = [0.0, 0.4, 0.8, 1.2, 1.5, 1.7]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (abs_eta >= lo) & (abs_eta < hi)
        if m.sum() < 100:
            continue
        e, z = eta_res[m], z0_res[m]
        rho_b = float(np.corrcoef(e, z)[0, 1])
        # LS slope of eta_res on z0_res
        slope = float(np.cov(e, z)[0, 1] / np.var(z))
        cosh_bar = float(np.mean(np.cosh(d["truth_eta"][m])))
        r_eff = -1.0 / (slope * cosh_bar) if slope < 0 else float("nan")
        sig = gaussian_resolution(e).sigma
        # eta residual with the fitted z0 component removed:
        sig_cond = gaussian_resolution(e - slope * z).sigma
        print(f"[{lo:4.1f},{hi:4.1f}) {int(m.sum()):6d} {rho_b:+7.3f} "
              f"{slope:13.3e} {r_eff:11.0f} {sig:8.4f} {sig_cond:10.4f}")

    print(
        "\nInterpretation:\n"
        "  * rho ~ -0.7 or stronger AND a stable r_eff across bins (close to\n"
        "    your ECal barrel radius) => eta is vertex/pointing-dominated:\n"
        "    any longitudinal-pointing gain improves eta AND z0 together.\n"
        "  * sig_eta|z0 << sig_eta shows how much eta resolution is left once\n"
        "    the shared vertex error is removed (the intrinsic position part).\n"
        "  * rho ~ 0 => eta and z0 errors are independent; eta has its own\n"
        "    headroom and pointing work only pays once."
    )


if __name__ == "__main__":
    main()