"""Geometry helpers: xyz <-> (eta, phi), shower axis projection."""
from __future__ import annotations
import numpy as np


def xyz_to_eta_phi(x, y, z):
    """Cartesian -> (eta, phi). Works for scalars or arrays."""
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    r = np.hypot(x, y)
    theta = np.arctan2(r, z)
    with np.errstate(divide="ignore"):
        eta = -np.log(np.tan(theta / 2))
    phi = np.arctan2(y, x)
    return eta, phi


def momentum_to_eta_phi(px, py, pz):
    """Same math as xyz_to_eta_phi, just relabeled for momentum vectors."""
    return xyz_to_eta_phi(px, py, pz)


def axis_from_momentum(px: float, py: float, pz: float) -> np.ndarray:
    """Unit 3-vector along the particle's flight direction (from the IP)."""
    v = np.array([px, py, pz], dtype=np.float64)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero-momentum vector")
    return v / n


def along_perp(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    axis_dir: np.ndarray,
    origin: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project hits onto an axis through `origin` (default: IP at (0,0,0)).

    Returns (s, r): s is the longitudinal coordinate along axis_dir (mm),
    r is the perpendicular distance from the axis line (mm).
    """
    if origin is None:
        origin = np.zeros(3, dtype=np.float64)
    pts = np.stack([x, y, z], axis=1).astype(np.float64) - origin
    s = pts @ axis_dir                                # (N,)
    perp = pts - np.outer(s, axis_dir)                # (N,3)
    r = np.linalg.norm(perp, axis=1)
    return s, r


def delta_eta_phi(eta, phi, eta0, phi0):
    """Element-wise (eta - eta0, phi - phi0) with phi wrapped to (-pi, pi)."""
    deta = np.asarray(eta) - eta0
    dphi = np.asarray(phi) - phi0
    dphi = np.mod(dphi + np.pi, 2 * np.pi) - np.pi
    return deta, dphi


if __name__ == "__main__":
    # 1. Vector along +x: eta=0, phi=0
    eta, phi = xyz_to_eta_phi(1.0, 0.0, 0.0)
    print(f"+x   -> eta={float(eta):.4f}  phi={float(phi):.4f}  (expect 0, 0)")

    # 2. Vector at (1,1,0): eta=0, phi=pi/4
    eta, phi = xyz_to_eta_phi(1.0, 1.0, 0.0)
    print(f"+x+y -> eta={float(eta):.4f}  phi={float(phi):.4f}  (expect 0, 0.7854)")

    # 3. along_perp, axis=+z, point (10,0,100) -> s=100, r=10
    axis = axis_from_momentum(0, 0, 1)
    s, r = along_perp(np.array([10.0]), np.array([0.0]), np.array([100.0]), axis)
    print(f"axis=+z, point (10,0,100): s={s[0]:.2f}  r={r[0]:.2f}  (expect 100, 10)")

    # 4. axis along +x+z (45deg), point (10,0,10) projects entirely along axis
    axis = axis_from_momentum(1, 0, 1)
    s, r = along_perp(np.array([10.0]), np.array([0.0]), np.array([10.0]), axis)
    print(f"axis=+x+z, point (10,0,10): s={s[0]:.4f}  r={r[0]:.4f}  (expect 14.1421, 0)")
