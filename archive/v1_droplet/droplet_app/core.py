"""Numpy-only predictor for beta_max. Loads model/bundle.npz + model/meta.json (written by export_model.py).

No sklearn needed at run time: the GP posterior (Matern 3/2 ARD + white noise, normalised y) is evaluated
directly from the exported training inputs, alpha vector and Cholesky factor. export_model.py checks that this
reproduces sklearn's mean and std to ~1e-10 before it writes the bundle.
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SQ3 = np.sqrt(3.0)


def phi_smooth(spacing_um, depth_um, w):
    """Smooth-plateau area fraction of a square grid of laser tracks (phi = 1 for a smooth plate)."""
    if depth_um <= 0 or spacing_um <= 0:
        return 1.0
    wid = w["25"] if depth_um >= 15 else w["6"]
    return (max(spacing_um - wid, 0.0) / spacing_um) ** 2


class Predictor:
    def __init__(self, folder=HERE / "model"):
        b = np.load(folder / "bundle.npz")
        self.meta = json.load(open(folder / "meta.json"))
        self.Xtr, self.alpha, self.L = b["Xtr"], b["alpha"], b["L"]
        self.mu_x, self.sd_x, self.ls = b["mu_x"], b["sd_x"], b["ls"]
        self.c, self.noise = float(b["c"]), float(b["noise"])
        self.y_mean, self.y_std = float(b["y_mean"]), float(b["y_std"])
        self.mcd_loc, self.mcd_prec, self.mah_sorted = b["mcd_loc"], b["mcd_prec"], b["mah_sorted"]
        self.features = self.meta["features"]
        self.backbone = self.meta.get("backbone")          # None or {"type": "laan", "A": float}
        self.q = self.meta["conformal_q"]                  # multiplier on GP std for a 90% band
        self.q_surface = self.meta["surface_halfwidth_log"]  # largest per-surface 90th pct of |log error| (worst-surface band)

    # ------------------------------------------------------------ features
    def features_for(self, D_mm, V, rho, sigma, mu, spacing_um=0.0, depth_um=0.0, phi=None, texvol=None):
        D = D_mm * 1e-3
        Re, We = rho * V * D / mu, rho * V ** 2 * D / sigma
        if phi is None:
            phi = phi_smooth(spacing_um, depth_um, self.meta["track_width_um"])
        if texvol is None:
            texvol = (1 - phi) * depth_um
        f = {"logRe": np.log(Re), "logWe": np.log(We), "logD": np.log(D_mm), "phi": phi, "texvol": texvol}
        return f, Re, We

    def _kernel(self, Xa, Xb):
        d = (Xa[:, None, :] - Xb[None, :, :]) / self.ls
        r = np.sqrt((d ** 2).sum(-1))
        return self.c * (1 + SQ3 * r) * np.exp(-SQ3 * r)

    def _backbone(self, Re, We):
        if not self.backbone:
            return 0.0
        A = self.backbone["A"]; P = We * Re ** (-0.4); s = np.sqrt(P)
        return np.log(Re ** 0.2 * s / (A + s))

    def predict_log(self, X, Re, We):
        """X: (n, n_features) raw features. Returns mean and std of log(beta)."""
        Xs = (np.atleast_2d(X) - self.mu_x) / self.sd_x
        Ks = self._kernel(Xs, self.Xtr)
        mean = Ks @ self.alpha * self.y_std + self.y_mean + self._backbone(np.asarray(Re), np.asarray(We))
        v = _tri_solve(self.L, Ks.T)
        var = np.clip(self.c + self.noise - (v ** 2).sum(0), 1e-12, None) * self.y_std ** 2
        return mean, np.sqrt(var)

    def ood_pct(self, X):
        d = np.atleast_2d(X) - self.mcd_loc
        m = np.sqrt(np.einsum("ij,jk,ik->i", d, self.mcd_prec, d))
        return np.searchsorted(self.mah_sorted, m) / len(self.mah_sorted) * 100

    def predict(self, D_mm, V, rho, sigma, mu, spacing_um=0.0, depth_um=0.0, phi=None, texvol=None):
        f, Re, We = self.features_for(D_mm, V, rho, sigma, mu, spacing_um, depth_um, phi, texvol)
        x = np.array([[f[k] for k in self.features]])
        m, s = self.predict_log(x, [Re], [We])
        m, s = float(m[0]), float(s[0])
        pct = float(self.ood_pct(x)[0])
        warn = []
        if pct >= 99.5: warn.append("strong shift from the training data (Mahalanobis ≥ 99.5th pct)")
        elif pct >= 95: warn.append("moderate shift from the training data (Mahalanobis ≥ 95th pct)")
        if Re < 70: warn.append("Re < 70: below the fitted range of the Laan scaling")
        rng = self.meta["train_ranges"]
        for k, v in {"Re": Re, "We": We, "D_mm": D_mm}.items():
            if not (rng[k][0] <= v <= rng[k][1]): warn.append(f"{k} = {v:.3g} outside training range {rng[k][0]:.3g}–{rng[k][1]:.3g}")
        if f["phi"] > 0.999:
            warn.append("smooth surface: wettability differs from the textured training surfaces; blind-test coverage "
                        f"of the 90% band was {self.meta['refh_coverage']:.0%}, so use the wide band")
        return {
            "beta_max": float(np.exp(m)),
            "interval90": [float(np.exp(m - self.q * s)), float(np.exp(m + self.q * s))],
            "interval90_wide": [float(np.exp(m - max(self.q_surface, self.q * s))), float(np.exp(m + max(self.q_surface, self.q * s)))],
            "Re": float(Re), "We": float(We), "phi": float(f["phi"]), "texvol_um": float(f["texvol"]),
            "ood_percentile": pct, "warnings": warn, "model": self.meta["model_name"],
        }


def _tri_solve(L, B):
    from numpy.linalg import solve
    try:
        from scipy.linalg import solve_triangular
        return solve_triangular(L, B, lower=True)
    except Exception:
        return solve(L, B)


def fluids():
    return json.load(open(HERE / "model" / "meta.json"))["fluids"]


def surfaces():
    return json.load(open(HERE / "model" / "meta.json"))["surfaces"]
