#!/usr/bin/env python3
from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import NMF

try:
    from .analysis_gse163211 import GSE, build_endpoints, load_gse163211_from_cache, select_variable_genes
except ImportError:  # pragma: no cover - supports direct script execution
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str((repo_root / "scripts").resolve()))
    from pipeline.analysis_gse163211 import GSE, build_endpoints, load_gse163211_from_cache, select_variable_genes  # type: ignore


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean(axis=0)) / df.std(axis=0, ddof=0).replace(0, np.nan)


def fit_nmf_scores(expr_log: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    X = expr_log.T.to_numpy(dtype=float)
    model = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=2000)
    W = model.fit_transform(X)
    W_df = pd.DataFrame(W, index=expr_log.columns, columns=[f"endotype_{i+1}" for i in range(k)])
    return _standardize(W_df)


def assoc_or_per_1sd(y: pd.Series, score: pd.Series) -> tuple[float, float, float, float]:
    design = sm.add_constant(pd.DataFrame({"score": score}), has_constant="add")
    data = pd.concat([y, design], axis=1).dropna()
    fit = sm.GLM(data[y.name], data.drop(columns=[y.name]), family=sm.families.Binomial()).fit()
    coef = float(fit.params["score"])
    se = float(fit.bse["score"])
    or_ = math.exp(coef)
    lo = math.exp(coef - 1.96 * se)
    hi = math.exp(coef + 1.96 * se)
    p = float(fit.pvalues["score"])
    return or_, lo, hi, p


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "results" / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    expr, meta, _platform = load_gse163211_from_cache(repo_root / "data" / "geo_cache")
    meta2 = build_endpoints(meta)
    y = meta2["y_fibrotic_nash"].rename("y_fibrotic_nash")

    expr_log = select_variable_genes(expr, n=5000)

    ks = [2, 3, 4]
    seeds = [1, 2, 3, 4, 5]

    rows = []
    for k, seed in itertools.product(ks, seeds):
        Wz = fit_nmf_scores(expr_log, k=k, seed=seed)
        for col in Wz.columns:
            or_, lo, hi, p = assoc_or_per_1sd(y, Wz[col])
            rows.append(
                {
                    "dataset_id": GSE,
                    "k": k,
                    "seed": seed,
                    "component": col,
                    "effect_type": "OR_per_1SD",
                    "effect": or_,
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "pvalue": p,
                    "n": int(Wz.shape[0]),
                    "events": int(y.loc[Wz.index].sum()),
                }
            )

    df = pd.DataFrame(rows)
    df["fdr"] = sm.stats.multipletests(df["pvalue"].to_numpy(), method="fdr_bh")[1]
    out_path = out_dir / "nmf_sensitivity.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
