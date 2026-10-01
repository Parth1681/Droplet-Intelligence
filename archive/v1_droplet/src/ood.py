"""OOD detector: Isolation Forest + Mahalanobis distance in the physics/surface feature space.
Scores are reported as percentiles of the training distribution (no invented confidence).
Also flags Re < 70, the lower bound of the Laan et al. (2014) scaling law's fitted range."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.covariance import MinCovDet
from src.data import load
from src.models_extra import feats, F


class OODDetector:
    def fit(self, X):
        self.iso = IsolationForest(n_estimators=500, contamination="auto", random_state=0).fit(X)
        self.mcd = MinCovDet(random_state=0).fit(X)
        self.iso_tr = np.sort(-self.iso.score_samples(X))
        self.mah_tr = np.sort(self.mcd.mahalanobis(X))
        return self

    def score(self, X):
        iso = np.searchsorted(self.iso_tr, -self.iso.score_samples(X)) / len(self.iso_tr) * 100
        mah = np.searchsorted(self.mah_tr, self.mcd.mahalanobis(X)) / len(self.mah_tr) * 100
        return pd.DataFrame({"iso_pct": iso, "mah_pct": mah})

    @staticmethod
    def label(iso_pct, mah_pct):
        s = max(iso_pct, mah_pct)
        return "in-distribution" if s < 95 else ("moderate shift" if s < 99.5 else "strong shift")


def run():
    t, r = load(); t = feats(t); r = feats(r.assign(spacing=0.0, depth=0.0))
    rows = []
    for s in sorted(t["sample"].unique()):                      # each textured surface as if unseen
        te = (t["sample"] == s).values
        sc = OODDetector().fit(t.loc[~te, F].values).score(t.loc[te, F].values)
        rows.append({"surface": s, "iso_pct_median": sc.iso_pct.median(), "mah_pct_median": sc.mah_pct.median(),
                     "frac_flagged(>95)": float((sc.max(axis=1) >= 95).mean())})
    sc = OODDetector().fit(t[F].values).score(r[F].values)
    rows.append({"surface": "REF-H", "iso_pct_median": sc.iso_pct.median(), "mah_pct_median": sc.mah_pct.median(),
                 "frac_flagged(>95)": float((sc.max(axis=1) >= 95).mean())})
    out = pd.DataFrame(rows); out.to_csv("results/ood_scores.csv", index=False)
    print(out.round(2).to_string(index=False))
    print("Re<70 rows: textured", int((t.Re < 70).sum()), "REF-H", int((r.Re < 70).sum()))


if __name__ == "__main__":
    run()
