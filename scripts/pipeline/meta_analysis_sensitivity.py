#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.stats.meta_analysis import combine_effects


def _dersimonian_laird(theta: np.ndarray, se: np.ndarray) -> dict[str, float]:
    w = 1.0 / np.square(se)
    mu_fixed = float(np.sum(w * theta) / np.sum(w))
    Q = float(np.sum(w * np.square(theta - mu_fixed)))
    df = max(1, int(len(theta) - 1))
    C = float(np.sum(w) - (np.sum(np.square(w)) / np.sum(w)))
    tau2 = 0.0 if C <= 0 else max(0.0, (Q - df) / C)
    w_re = 1.0 / (np.square(se) + tau2)
    mu_re = float(np.sum(w_re * theta) / np.sum(w_re))
    se_re = float(math.sqrt(1.0 / np.sum(w_re)))
    ci_lo = mu_re - 1.96 * se_re
    ci_hi = mu_re + 1.96 * se_re
    I2 = 0.0
    if Q > df and Q > 0:
        I2 = max(0.0, (Q - df) / Q) * 100.0
    return {
        "mu_log_or": mu_re,
        "se_mu": se_re,
        "ci_lower_log": ci_lo,
        "ci_upper_log": ci_hi,
        "tau2": tau2,
        "I2_percent": I2,
        "Q": Q,
        "df": float(df),
        "k": float(len(theta)),
        "method": "DL",
    }


def _iterated_re(theta: np.ndarray, se: np.ndarray) -> dict[str, float]:
    res = combine_effects(theta, np.square(se), method_re="iterated")
    mu = float(res.mean_effect_re)
    se_mu = float(math.sqrt(res.var_eff_w_re))
    ci = res.conf_int()[1]
    ci_lo = float(ci[0])
    ci_hi = float(ci[1])
    Q = float(res.q)
    df = float(res.df)
    I2 = 0.0
    if Q > df and Q > 0:
        I2 = max(0.0, (Q - df) / Q) * 100.0
    tau2 = float(res.tau2) if hasattr(res, "tau2") else float("nan")
    return {
        "mu_log_or": mu,
        "se_mu": se_mu,
        "ci_lower_log": ci_lo,
        "ci_upper_log": ci_hi,
        "tau2": tau2,
        "I2_percent": I2,
        "Q": Q,
        "df": df,
        "k": float(res.k),
        "method": "iterated",
    }


def _reml_hksj(theta: np.ndarray, se: np.ndarray) -> dict[str, float]:
    vi = np.square(se.astype(float))
    theta = theta.astype(float)

    def objective(tau2: float) -> float:
        v = vi + max(0.0, tau2)
        w = 1.0 / v
        mu = float(np.sum(w * theta) / np.sum(w))
        q = float(np.sum(w * np.square(theta - mu)))
        return float(np.sum(np.log(v)) + np.log(np.sum(w)) + q)

    upper = max(1.0, float(np.var(theta, ddof=1) * 10.0 if theta.size > 1 else 1.0))
    opt = optimize.minimize_scalar(objective, bounds=(0.0, upper), method="bounded")
    tau2 = max(0.0, float(opt.x)) if opt.success else 0.0
    v = vi + tau2
    w = 1.0 / v
    mu = float(np.sum(w * theta) / np.sum(w))
    q_hk = float(np.sum(w * np.square(theta - mu)))
    df = max(1, int(theta.size - 1))
    scale = max(1.0e-12, q_hk / df)
    se_mu = float(math.sqrt(scale / np.sum(w)))
    tcrit = float(stats.t.ppf(0.975, df))
    ci_lo = mu - tcrit * se_mu
    ci_hi = mu + tcrit * se_mu
    w_fixed = 1.0 / vi
    mu_fixed = float(np.sum(w_fixed * theta) / np.sum(w_fixed))
    q_cochran = float(np.sum(w_fixed * np.square(theta - mu_fixed)))
    I2 = 0.0
    if q_cochran > df and q_cochran > 0:
        I2 = max(0.0, (q_cochran - df) / q_cochran) * 100.0
    return {
        "mu_log_or": mu,
        "se_mu": se_mu,
        "ci_lower_log": ci_lo,
        "ci_upper_log": ci_hi,
        "tau2": tau2,
        "I2_percent": I2,
        "Q": q_cochran,
        "df": float(df),
        "k": float(theta.size),
        "method": "REML_HKSJ",
    }


def build(effect_tsv: Path, out_tsv: Path) -> None:
    eff = pd.read_csv(effect_tsv, sep="\t")
    eff["feature"] = eff["feature"].astype(str)
    eff["log_or"] = pd.to_numeric(eff["log_or"], errors="coerce")
    eff["se"] = pd.to_numeric(eff["se"], errors="coerce")
    eff = eff.dropna(subset=["log_or", "se"])

    rows: list[dict[str, object]] = []
    for feat in sorted(eff["feature"].unique()):
        sub = eff[eff["feature"] == feat].copy()
        theta = sub["log_or"].to_numpy(dtype=float)
        se = sub["se"].to_numpy(dtype=float)
        if theta.size < 2:
            continue
        for m in (_reml_hksj(theta, se), _dersimonian_laird(theta, se), _iterated_re(theta, se)):
            rows.append(
                {
                    "feature": feat,
                    "analysis": "all_cohorts",
                    "method": m["method"],
                    "omitted_cohort": "",
                    "k": int(m["k"]),
                    "pooled_or": math.exp(float(m["mu_log_or"])),
                    "ci_lower": math.exp(float(m["ci_lower_log"])),
                    "ci_upper": math.exp(float(m["ci_upper_log"])),
                    "tau2": float(m["tau2"]),
                    "I2_percent": float(m["I2_percent"]),
                    "Q": float(m["Q"]),
                    "df": float(m["df"]),
                }
            )
        for omitted in sorted(sub["dataset_id"].astype(str).unique()):
            loo = sub[sub["dataset_id"].astype(str) != omitted].copy()
            theta_loo = loo["log_or"].to_numpy(dtype=float)
            se_loo = loo["se"].to_numpy(dtype=float)
            if theta_loo.size < 2:
                continue
            m = _reml_hksj(theta_loo, se_loo)
            rows.append(
                {
                    "feature": feat,
                    "analysis": "leave_one_cohort_out",
                    "method": m["method"],
                    "omitted_cohort": omitted,
                    "k": int(m["k"]),
                    "pooled_or": math.exp(float(m["mu_log_or"])),
                    "ci_lower": math.exp(float(m["ci_lower_log"])),
                    "ci_upper": math.exp(float(m["ci_upper_log"])),
                    "tau2": float(m["tau2"]),
                    "I2_percent": float(m["I2_percent"]),
                    "Q": float(m["Q"]),
                    "df": float(m["df"]),
                }
            )

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_tsv, sep="\t", index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build meta-analysis sensitivity table (DL vs iterated RE).")
    ap.add_argument("--effect-tsv", required=True)
    ap.add_argument("--out-tsv", required=True)
    args = ap.parse_args()
    build(Path(args.effect_tsv), Path(args.out_tsv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
