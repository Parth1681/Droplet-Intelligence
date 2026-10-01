"""Data loading, physics groups and leakage-safe condition IDs for the Može et al. droplet dataset."""
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
TEX_COLS = ["sample", "D", "V", "rho", "sigma", "mu", "spacing", "depth",
            "Re", "We", "tc", "eta", "beta", "vlam"]
REF_COLS = ["sample", "D", "V", "rho", "sigma", "mu", "Re", "We", "beta", "vlam"]


def glycerol_level(mu):
    """Map viscosity (Pa s) to nominal glycerol wt.% (5 fluid families)."""
    return pd.cut(mu, [0, 0.0012, 0.003, 0.02, 0.08, 1.0],
                  labels=[0, 20, 60, 78, 91]).astype(int)


def add_physics(d):
    d = d.copy()
    d["Re_c"] = d.rho * d.V * d.D / d.mu
    d["We_c"] = d.rho * d.V ** 2 * d.D / d.sigma
    d["Oh"] = d.mu / np.sqrt(d.rho * d.sigma * d.D)
    d["P"] = d.We * d.Re ** (-0.4)                      # Laan impact parameter
    d["logRe"], d["logWe"], d["logOh"] = np.log(d.Re), np.log(d.We), np.log(d.Oh)
    d["gly"] = glycerol_level(d.mu)
    return d


def condition_ids(d, gap=0.10):
    """Group replicate impacts: same surface + fluid family + velocity cluster.
    Velocities within a (surface, fluid) are clustered by gaps > `gap` m/s (0.10 gives 292 clean 5-repeat groups)."""
    ids = pd.Series(index=d.index, dtype=object)
    for (s, g), sub in d.groupby(["sample", "gly"]):
        v = sub.V.sort_values()
        cl = (v.diff() > gap).cumsum()
        ids.loc[v.index] = [f"{s}|{g}|{c}" for c in cl]
    return ids


def load():
    t = pd.read_csv(RAW / "01 - Rebound Data.csv", encoding="latin1")
    r = pd.read_csv(RAW / "02 - Rebound Data - REF-H.csv", encoding="latin1")
    t.columns, r.columns = TEX_COLS, REF_COLS
    t, r = add_physics(t), add_physics(r)
    t["cond"], r["cond"] = condition_ids(t), condition_ids(r)
    return t, r
