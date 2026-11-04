#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
leaning_analysis.py
===================
Leaning score analysis (engagement toward reliable vs unreliable sources)
between phase 1 and phase 2, by condition.

ENGAGEMENT SCORE (per interaction = one article shown)
    E = readmore_clicked + full_clicked + liked_final               range [0, 3]

    The distribution of E across interactions is plotted (engagement_00_distribution.png).

BINARY ENGAGEMENT (alternative version, metric "bin")
    engaged = 1 if E > median(E) over all interactions (or E >= median with --bin-rule ge)

LEANING SCORE (per participant and per phase), computed for both versions
    eng: L = mean(E | reliable) - mean(E | unreliable)               range [-3, 3]
    bin: L = P(engaged | reliable) - P(engaged | unreliable)         range [-1, 1]
    L > 0  -> engages more with reliable sources
    L < 0  -> engages more with unreliable sources

Analyses (repeated for "eng" and "bin"; output files end in _eng / _bin):
  1. Within-condition: is L different between phase 1 and phase 2?
     (paired Wilcoxon + paired t-test, Holm correction across conditions)
  2. Between conditions: does the change Delta = L_phase2 - L_phase1 depend on the treatment?
     (Kruskal-Wallis, ANOVA, pairwise Mann-Whitney with Holm)
  3. Agreement between phases for each condition: in which treatment does the
     participant's leaning stay most similar between the two phases?
     Lin's CCC (absolute agreement) with bootstrap CI, ICC(A,1), Spearman,
     % same sign; pairwise CCC differences with a permutation test + Holm.
  4. Article-level model (GEE, clusters = participant; linear for eng, logistic for bin):
     E ~ reliable * phase * condition  -> joint Wald test on the triple interaction.
  5. Engagement (not leaning) in phase 1 vs phase 2, separately for reliable and
     unreliable articles in each condition (participant means, paired Wilcoxon, Holm).

Usage:
    python leaning_analysis.py
    python leaning_analysis.py --data "path/file.csv" --out "path/figures"
"""

import argparse
import os
import sys
import textwrap
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_SM = True
except ImportError:
    HAS_SM = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

RESULTS_DIR = "/Users/jacoponudo/Documents/Projects/inerzia/results"
DATA_PATTERN = "*responses_v4*.csv"   # default: most recent v4 CSV in RESULTS_DIR
DEFAULT_DATA = None
DEFAULT_OUT = "/Users/jacoponudo/Documents/Projects/inerzia/figures"

METRICS = {
    "eng": "Engagement score (read more + read full + like)",
    "bin": "Binary engagement (E above median)",   # label completed in main() with the median
}
SMAX = 3  # maximum engagement score per interaction
METRIC_MAX = {"eng": SMAX, "bin": 1}  # max per-interaction value -> leaning range [-max, +max]

COL_REL = "#009E73"      # reliable (same palette as the dataset)
COL_UNREL = "#E69F00"    # unreliable
COND_COLORS = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]


# --------------------------------------------------------------------------- #
# Loading and cleaning
# --------------------------------------------------------------------------- #
def find_data(path):
    """Use --data if given, otherwise the most recently modified CSV matching DATA_PATTERN in RESULTS_DIR."""
    if path:
        if not os.path.exists(path):
            sys.exit(f"[ERROR] File not found: {path}")
        return path
    import glob
    cands = sorted(glob.glob(os.path.join(RESULTS_DIR, DATA_PATTERN)), key=os.path.getmtime)
    if not cands:
        others = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.csv")))
        sys.exit(f"[ERROR] No file matching '{DATA_PATTERN}' in {RESULTS_DIR}.\n"
                 f"CSV files found: {[os.path.basename(o) for o in others]}\n"
                 f"Pass the file explicitly with --data \"path/file.csv\".")
    if len(cands) > 1:
        print(f"[INFO] {len(cands)} files match '{DATA_PATTERN}': using the most recent one.")
    return cands[-1]


def detect_sep(path):
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        header = f.readline()
    counts = {s: header.count(s) for s in ["\t", ",", ";"]}
    return max(counts, key=counts.get)


def to_bool(s):
    m = {"TRUE": True, "FALSE": False, "T": True, "F": False, "1": True, "0": False,
         "1.0": True, "0.0": False, "YES": True, "NO": False, "VERO": True, "FALSO": False}
    return s.astype(str).str.strip().str.upper().map(m)


def load_data(path, phase_a, phase_b, keep_only_prolific=True):
    sep = detect_sep(path)
    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]

    required = ["session_id", "condition", "phase", "reliability_class",
                "readmore_clicked", "full_clicked", "liked_final"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"[ERROR] Missing columns in CSV: {missing}\nColumns found: {list(df.columns)}")

    df["pid"] = df["session_id"].str.strip()
    df = df[df["pid"] != ""]

    # only valid participants: those with a Prolific ID
    if keep_only_prolific:
        if "prolific_pid" not in df.columns:
            sys.exit("[ERROR] Column 'prolific_pid' not found in CSV.")
        ppid = df["prolific_pid"].astype(str).str.strip()
        valid = ~ppid.str.lower().isin(["", "nan", "none", "null", "na", "n/a"])
        n_all, n_ok = df["pid"].nunique(), df.loc[valid, "pid"].nunique()
        print(f"Sessions with Prolific ID: {n_ok} of {n_all} "
              f"(excluded {n_all - n_ok} sessions without Prolific ID)")
        df = df[valid]
        if df.empty:
            sys.exit("[ERROR] No rows with a non-empty prolific_pid.")
    df["phase"] = pd.to_numeric(df["phase"].str.replace(",", "."), errors="coerce")
    cond_num = pd.to_numeric(df["condition"].str.replace(",", "."), errors="coerce")
    df["condition"] = cond_num.astype("Int64").astype(str) if cond_num.notna().all() else df["condition"].str.strip()

    df["reliability_class"] = df["reliability_class"].str.strip().str.lower()
    df = df[df["reliability_class"].isin(["reliable", "unreliable"])]
    df["reliable"] = df["reliability_class"] == "reliable"

    df["readmore"] = to_bool(df["readmore_clicked"]).fillna(False).astype(int)
    df["readfull"] = to_bool(df["full_clicked"]).fillna(False).astype(int)
    df["like"] = to_bool(df["liked_final"]).fillna(False).astype(int)
    df["eng"] = df["readmore"] + df["readfull"] + df["like"]
    df["elig_eng"] = True

    df = df[df["phase"].isin([phase_a, phase_b])].copy()
    if df.empty:
        sys.exit(f"[ERROR] No rows with phase in {{{phase_a}, {phase_b}}}.")

    # one condition per participant (the first one, if inconsistent)
    n_cond = df.groupby("pid")["condition"].nunique()
    if (n_cond > 1).any():
        print(f"[WARNING] {int((n_cond > 1).sum())} participants with more than one condition: using the first one.")
    first_cond = df.groupby("pid")["condition"].first()
    df["condition"] = df["pid"].map(first_cond)

    if "condition_label" in df.columns:
        labels = df.groupby("condition")["condition_label"].first().to_dict()
    else:
        labels = {}
    conds = sorted(df["condition"].unique(), key=lambda x: (len(str(x)), str(x)))
    labels = {c: (labels.get(c) or f"condition {c}") for c in conds}
    return df, conds, labels


# --------------------------------------------------------------------------- #
# Leaning score
# --------------------------------------------------------------------------- #
def compute_leaning(df, metric, phase_a, phase_b):
    d = df[df[f"elig_{metric}"]]
    agg = d.groupby(["pid", "phase", "reliable"])[metric].mean().unstack("reliable")
    agg = agg.reindex(columns=[True, False])
    agg.columns = ["rate_rel", "rate_unrel"]
    agg["leaning"] = agg["rate_rel"] - agg["rate_unrel"]
    long = agg.reset_index()
    cond = df.groupby("pid")["condition"].first()
    long["condition"] = long["pid"].map(cond)

    wide = long.pivot(index="pid", columns="phase", values="leaning").reindex(columns=[phase_a, phase_b])
    wide.columns = ["L_A", "L_B"]
    n_before = len(wide)
    wide = wide.dropna()
    wide["delta"] = wide["L_B"] - wide["L_A"]
    wide["condition"] = wide.index.map(cond)
    return long, wide, n_before - len(wide)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    m = len(pv)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pv[idx])
        adj[idx] = min(running, 1.0)
    out[ok] = adj
    return out


def mean_ci(x, level=0.95):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan
    m = x.mean()
    if n < 2:
        return m, np.nan, np.nan
    h = stats.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h


def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return np.nan
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    den = vx + vy + (mx - my) ** 2
    return np.nan if den == 0 else 2 * cov / den


def icc_a1(x, y):
    """ICC(A,1): two-way, absolute agreement, single measure (McGraw & Wong)."""
    X = np.column_stack([x, y]).astype(float)
    n, k = X.shape
    if n < 2:
        return np.nan
    g = X.mean()
    ssr = k * ((X.mean(1) - g) ** 2).sum()
    ssc = n * ((X.mean(0) - g) ** 2).sum()
    sst = ((X - g) ** 2).sum()
    sse = sst - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k / n * (msc - mse)
    return np.nan if den == 0 else (msr - mse) / den


def boot_ci(x, y, fn, B, rng, level=0.95):
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    vals = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        vals[b] = fn(x[i], y[i])
    vals = vals[~np.isnan(vals)]
    if len(vals) < B * 0.5:
        return np.nan, np.nan
    a = (1 - level) / 2
    return np.quantile(vals, a), np.quantile(vals, 1 - a)


def safe_spearman(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    r = stats.spearmanr(x, y)
    return r[0], r[1]


def within_tests(wide, conds):
    rows = []
    for c in conds:
        w = wide[wide["condition"] == c]
        d = w["delta"].values
        n = len(d)
        mA, _, _ = mean_ci(w["L_A"])
        mB, _, _ = mean_ci(w["L_B"])
        md, lo, hi = mean_ci(d)
        sd = d.std(ddof=1) if n > 1 else np.nan
        t = p_t = W = p_w = np.nan
        if n >= 2 and sd > 0:
            t, p_t = stats.ttest_1samp(d, 0.0)
        if n >= 2 and np.any(d != 0):
            try:
                W, p_w = stats.wilcoxon(d, zero_method="wilcox")
            except ValueError:
                pass
        rows.append(dict(condition=c, n=n, mean_L_phaseA=mA, mean_L_phaseB=mB,
                         mean_delta=md, delta_ci_lo=lo, delta_ci_hi=hi, sd_delta=sd,
                         cohen_dz=(md / sd if sd and sd > 0 else np.nan),
                         t=t, p_ttest=p_t, W=W, p_wilcoxon=p_w))
    out = pd.DataFrame(rows)
    out["p_ttest_holm"] = holm(out["p_ttest"])
    out["p_wilcoxon_holm"] = holm(out["p_wilcoxon"])
    return out


def between_tests(wide, conds, col="delta"):
    groups = [wide.loc[wide["condition"] == c, col].values for c in conds]
    groups_ok = [g for g in groups if len(g) > 0]
    res = dict(variable=col, kw_H=np.nan, kw_p=np.nan, anova_F=np.nan, anova_p=np.nan)
    if len(groups_ok) >= 2:
        try:
            res["kw_H"], res["kw_p"] = stats.kruskal(*groups_ok)
        except ValueError:
            pass
        if all(len(g) >= 2 for g in groups_ok):
            res["anova_F"], res["anova_p"] = stats.f_oneway(*groups_ok)
    pw = []
    for c1, c2 in combinations(conds, 2):
        g1 = wide.loc[wide["condition"] == c1, col].values
        g2 = wide.loc[wide["condition"] == c2, col].values
        U = p = np.nan
        if len(g1) > 0 and len(g2) > 0:
            try:
                U, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            except ValueError:
                pass
        pw.append(dict(cond_1=c1, cond_2=c2, n_1=len(g1), n_2=len(g2),
                       mean_1=np.mean(g1) if len(g1) else np.nan,
                       mean_2=np.mean(g2) if len(g2) else np.nan, U=U, p=p))
    pw = pd.DataFrame(pw)
    if not pw.empty:
        pw["p_holm"] = holm(pw["p"])
    return res, pw


def agreement(wide, conds, B, n_perm, rng):
    rows = []
    data = {}
    for c in conds:
        w = wide[wide["condition"] == c]
        x, y = w["L_A"].values, w["L_B"].values
        data[c] = (x, y)
        n = len(x)
        c_val = ccc(x, y)
        c_lo, c_hi = boot_ci(x, y, ccc, B, rng)
        rho, p_rho = safe_spearman(x, y)
        rows.append(dict(condition=c, n=n, CCC=c_val, CCC_ci_lo=c_lo, CCC_ci_hi=c_hi,
                         ICC_A1=icc_a1(x, y), spearman_rho=rho, spearman_p=p_rho,
                         mean_abs_delta=np.mean(np.abs(y - x)) if n else np.nan,
                         pct_same_sign=(np.mean(np.sign(x) == np.sign(y)) * 100) if n else np.nan))
    agr = pd.DataFrame(rows)

    # pairwise CCC differences: permutation test on condition labels
    pw = []
    for c1, c2 in combinations(conds, 2):
        x1, y1 = data[c1]
        x2, y2 = data[c2]
        obs = ccc(x1, y1) - ccc(x2, y2)
        p = np.nan
        if len(x1) >= 3 and len(x2) >= 3 and not np.isnan(obs):
            X, Y, n1 = np.r_[x1, x2], np.r_[y1, y2], len(x1)
            cnt, valid = 0, 0
            for _ in range(n_perm):
                i = rng.permutation(len(X))
                d = ccc(X[i[:n1]], Y[i[:n1]]) - ccc(X[i[n1:]], Y[i[n1:]])
                if np.isnan(d):
                    continue
                valid += 1
                cnt += abs(d) >= abs(obs) - 1e-12
            p = (cnt + 1) / (valid + 1) if valid else np.nan
        pw.append(dict(cond_1=c1, cond_2=c2, CCC_1=ccc(x1, y1), CCC_2=ccc(x2, y2),
                       diff=obs, p_perm=p))
    pw = pd.DataFrame(pw)
    if not pw.empty:
        pw["p_perm_holm"] = holm(pw["p_perm"])

    # omnibus: SD of CCCs across conditions vs permutation
    omni_p = np.nan
    sizes = [len(data[c][0]) for c in conds]
    if len(conds) >= 2 and all(s >= 3 for s in sizes):
        X = np.concatenate([data[c][0] for c in conds])
        Y = np.concatenate([data[c][1] for c in conds])
        bounds = np.cumsum([0] + sizes)

        def stat(ix):
            v = [ccc(X[ix[bounds[j]:bounds[j + 1]]], Y[ix[bounds[j]:bounds[j + 1]]]) for j in range(len(conds))]
            return np.nanstd(v)

        obs = stat(np.arange(len(X)))
        perm = np.array([stat(rng.permutation(len(X))) for _ in range(n_perm)])
        perm = perm[~np.isnan(perm)]
        if len(perm):
            omni_p = (np.sum(perm >= obs - 1e-12) + 1) / (len(perm) + 1)
    return agr, pw, omni_p


def gee_model(df, metric, phase_b, tables_dir):
    if not HAS_SM:
        return None, "statsmodels not installed: GEE skipped."
    d = df[df[f"elig_{metric}"]].copy()
    d["y"] = d[metric].astype(float)
    d["rel"] = d["reliable"].astype(int)
    d["ph2"] = (d["phase"] == phase_b).astype(int)
    family = sm.families.Binomial() if metric == "bin" else sm.families.Gaussian()
    unit = "log-odds" if metric == "bin" else "score points"
    try:
        model = smf.gee("y ~ rel * ph2 * C(condition)", groups="pid", data=d,
                        family=family, cov_struct=sm.cov_struct.Exchangeable())
        res = model.fit()
        names = list(res.params.index)
        idx = [i for i, nm in enumerate(names) if "rel:ph2:C(condition)" in nm]
        wald_txt = ""
        if idx:
            R = np.zeros((len(idx), len(names)))
            for r, i in enumerate(idx):
                R[r, i] = 1
            wt = res.wald_test(R, scalar=True)
            stat = float(np.squeeze(wt.statistic))
            pval = float(np.squeeze(wt.pvalue))
            wald_txt = (f"\nJoint Wald test, triple interaction reliable x phase x condition "
                        f"(df={len(idx)}): chi2 = {stat:.3f}, p = {pval:.4g}\n")
        else:
            pval = np.nan
        with open(os.path.join(tables_dir, f"gee_{metric}.txt"), "w") as f:
            f.write(str(res.summary()))
            f.write("\n" + wald_txt)
            f.write("\nInterpretation: 'rel:ph2' = change in the reliable-unreliable gap ({unit}) between phases "
                    "in the reference condition; 'rel:ph2:C(condition)[T.k]' = how much that change differs "
                    "in condition k compared with the reference.\n")
        return pval, wald_txt.strip()
    except Exception as e:  # noqa: BLE001
        return None, f"GEE not estimable ({type(e).__name__}: {e})"


# --------------------------------------------------------------------------- #
# Binary engagement (median split)
# --------------------------------------------------------------------------- #
def add_binary(df, rule="gt"):
    """Binary engagement: an interaction is 'engaged' if its score E is above the median
    of E across all interactions in the analysed sample (rule 'gt'), or at/above it ('ge')."""
    med = float(df["eng"].median())
    df["bin"] = ((df["eng"] > med) if rule == "gt" else (df["eng"] >= med)).astype(int)
    df["elig_bin"] = True
    share = df["bin"].mean() * 100
    if share in (0, 100):
        print(f"[WARNING] Binary engagement is constant ({share:.0f}% engaged): "
              f"try --bin-rule {'ge' if rule == 'gt' else 'gt'}.")
    return med, share


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def wrap(s, w=22):
    return "\n".join(textwrap.wrap(str(s), w))


def score_lim(values, cap=SMAX):
    """Symmetric axis limit adapted to the data, never beyond the theoretical range."""
    v = np.abs(np.asarray(values, float))
    m = np.nanmax(v) if np.isfinite(v).any() else 1.0
    return min(cap, max(0.5, m)) * 1.08


def stars(p):
    if p is None or np.isnan(p):
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def leaning_ylabel(m):
    mx = METRIC_MAX[m]
    if m == "bin":
        return f"Leaning (−{mx} to +{mx})\nP(engaged | reliable) − P(engaged | unreliable)"
    return f"Leaning (−{mx} to +{mx})\nmean E | reliable − mean E | unreliable"


def fig_engagement_distribution(df, med, rule, out, tables, A, B):
    from matplotlib.patches import Patch
    vals = np.arange(0, SMAX + 1)
    engaged_vals = vals[(vals > med) if rule == "gt" else (vals >= med)]
    boundary = engaged_vals.min() - 0.5 if len(engaged_vals) else None
    N = len(df)
    share = df["bin"].mean() * 100

    # table of counts
    tab = (df.groupby(["reliability_class", "phase", "condition", "eng"]).size()
           .rename("n").reset_index())
    tab.to_csv(os.path.join(tables, "engagement_distribution.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw=dict(width_ratios=[1, 1.35, 1.1]))

    # (a) all interactions
    ax = axes[0]
    counts = df["eng"].value_counts().reindex(vals, fill_value=0)
    pct = counts / counts.sum() * 100
    cols = ["#4D4D4D" if v in engaged_vals else "#C8C8C8" for v in vals]
    ax.bar(vals, pct, color=cols, width=0.7, edgecolor="black", lw=0.5)
    for v, p_, c_ in zip(vals, pct, counts):
        ax.text(v, p_ + pct.max() * 0.02, f"{p_:.1f}%\n(n={c_})", ha="center", va="bottom", fontsize=8)
    if boundary is not None:
        ax.axvline(boundary, color="#C0392B", ls="--", lw=1.3)
    ax.set_xticks(vals)
    ax.set_xlabel("Engagement score E per interaction")
    ax.set_ylabel("% of interactions")
    ax.set_ylim(0, pct.max() * 1.3)
    ax.set_title(f"(a) All interactions (N = {N})\nmedian E = {med:g}; "
                 f"engaged if E {'>' if rule == 'gt' else '≥'} median → {share:.1f}%", fontsize=10)
    ax.legend(handles=[Patch(color="#C8C8C8", label="not engaged (binary)"),
                       Patch(color="#4D4D4D", label="engaged (binary)")],
              fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # (b) by reliability x phase
    ax = axes[1]
    combos = [(A, "reliable", COL_REL, None), (A, "unreliable", COL_UNREL, None),
              (B, "reliable", COL_REL, "//"), (B, "unreliable", COL_UNREL, "//")]
    width = 0.2
    for k, (ph, rel, color, hatch) in enumerate(combos):
        s = df.loc[(df["phase"] == ph) & (df["reliability_class"] == rel), "eng"]
        p_ = s.value_counts().reindex(vals, fill_value=0) / max(len(s), 1) * 100
        ax.bar(vals + (k - 1.5) * width, p_, width, color=color, hatch=hatch, edgecolor="black", lw=0.5,
               alpha=0.9, label=f"Phase {ph:g} – {rel} (n={len(s)})")
    if boundary is not None:
        ax.axvline(boundary, color="#C0392B", ls="--", lw=1.3)
    ax.set_xticks(vals)
    ax.set_xlabel("Engagement score E per interaction")
    ax.set_ylabel("% of interactions within group")
    ax.set_title("(b) By source reliability and phase", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # (c) components
    ax = axes[2]
    comps = [("readmore", "Read more"), ("readfull", "Read full"), ("like", "Like (final)")]
    for k, (ph, rel, color, hatch) in enumerate(combos):
        s = df[(df["phase"] == ph) & (df["reliability_class"] == rel)]
        p_ = [s[c].mean() * 100 if len(s) else np.nan for c, _ in comps]
        ax.bar(np.arange(3) + (k - 1.5) * width, p_, width, color=color, hatch=hatch,
               edgecolor="black", lw=0.5, alpha=0.9)
    ax.set_xticks(range(3))
    ax.set_xticklabels([lab for _, lab in comps])
    ax.set_ylabel("% of interactions with the action")
    ax.set_title("(c) Components of E (bars as in b)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Distribution of engagement per user–article interaction "
                 "(red dashed line = binary threshold)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(out, "engagement_00_distribution.png"), dpi=300)
    plt.close(fig)


def fig_leaning_by_phase(w, m, conds, labels, A, B, out):
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    nc = len(conds)
    for i, c in enumerate(conds):
        sub = w[w["condition"] == c]
        stats_ = [mean_ci(sub["L_A"]), mean_ci(sub["L_B"])]
        mu = np.array([s[0] for s in stats_])
        lo = np.array([s[1] for s in stats_])
        hi = np.array([s[2] for s in stats_])
        x = np.array([0, 1]) + (i - (nc - 1) / 2) * 0.07
        yerr = np.vstack([np.nan_to_num(mu - lo), np.nan_to_num(hi - mu)])
        ax.errorbar(x, mu, yerr=yerr, marker="o", ms=7, capsize=3, lw=2,
                    color=COND_COLORS[i % len(COND_COLORS)],
                    label=f"{c}: {wrap(labels[c], 30)} (n={len(sub)})")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Phase {A:g}", f"Phase {B:g}"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_title(METRICS[m])
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylabel(leaning_ylabel(m))
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8.5)
    fig.suptitle("Leaning toward reliable sources, by phase and condition\n(mean ± 95% CI)")
    fig.tight_layout(rect=[0, 0.16, 1, 0.95])
    fig.savefig(os.path.join(out, f"leaning_01_by_phase_condition_{m}.png"), dpi=300)
    plt.close(fig)


def fig_paired(w, m, conds, labels, within, A, B, out, rng):
    nc = len(conds)
    lim = score_lim(w[["L_A", "L_B"]].values, cap=METRIC_MAX[m])
    fig, axes = plt.subplots(1, nc, figsize=(3.4 * nc, 4.2), sharey=True)
    for i, (ax, c) in enumerate(zip(np.atleast_1d(axes), conds)):
        sub = w[w["condition"] == c]
        col = COND_COLORS[i % len(COND_COLORS)]
        jit = rng.uniform(-0.04, 0.04, size=(len(sub), 2))
        for (_, r), j in zip(sub.iterrows(), jit):
            ax.plot([0 + j[0], 1 + j[1]], [r["L_A"], r["L_B"]], color=col, alpha=0.25, lw=1)
        ax.plot([0, 1], [sub["L_A"].mean(), sub["L_B"].mean()], color="black", lw=3, marker="o", ms=8)
        row = within[within["condition"] == c].iloc[0]
        ax.set_title(f"{wrap(labels[c], 24)}\nn={int(row.n)}  Wilcoxon p(Holm)={row.p_wilcoxon_holm:.3g}",
                     fontsize=9)
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Phase {A:g}", f"Phase {B:g}"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-lim, lim)
        ax.grid(axis="y", alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("Leaning")
    fig.suptitle(f"Individual leaning trajectories — {METRICS[m]}")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"leaning_02_paired_{m}.png"), dpi=300)
    plt.close(fig)


def fig_agreement_scatter(w, m, conds, labels, agr, A, B, out, rng):
    nc = len(conds)
    lim = score_lim(w[["L_A", "L_B"]].values, cap=METRIC_MAX[m])
    jit = 0.025  # display-only jitter
    fig, axes = plt.subplots(1, nc, figsize=(3.5 * nc, 3.9), sharex=True, sharey=True)
    for i, (ax, c) in enumerate(zip(np.atleast_1d(axes), conds)):
        sub = w[w["condition"] == c]
        jx = rng.uniform(-jit, jit, len(sub))
        jy = rng.uniform(-jit, jit, len(sub))
        ax.scatter(sub["L_A"] + jx, sub["L_B"] + jy, s=28, alpha=0.6,
                   color=COND_COLORS[i % len(COND_COLORS)], edgecolor="white", lw=0.5)
        ax.plot([-lim, lim], [-lim, lim], color="grey", ls="--", lw=1)
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5)
        r = agr[agr["condition"] == c].iloc[0]
        ax.text(0.03, 0.97,
                f"CCC = {r.CCC:.2f} [{r.CCC_ci_lo:.2f}, {r.CCC_ci_hi:.2f}]\n"
                f"ICC(A,1) = {r.ICC_A1:.2f}\nρ = {r.spearman_rho:.2f}   n = {int(r.n)}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))
        ax.set_title(wrap(labels[c], 26), fontsize=9)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Leaning phase {A:g}")
    np.atleast_1d(axes)[0].set_ylabel(f"Leaning phase {B:g}")
    fig.suptitle(f"Agreement of leaning between phases — {METRICS[m]}")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"leaning_03_agreement_scatter_{m}.png"), dpi=300)
    plt.close(fig)


def fig_delta(w, within, between, m, conds, labels, out, rng):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    data = [w.loc[w["condition"] == c, "delta"].values for c in conds]
    dlim = score_lim(w["delta"].values, cap=2 * METRIC_MAX[m])
    bp = ax.boxplot(data, positions=range(len(conds)), widths=0.5, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black"))
    for i, (patch, d) in enumerate(zip(bp["boxes"], data)):
        col = COND_COLORS[i % len(COND_COLORS)]
        patch.set_facecolor(col)
        patch.set_alpha(0.3)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=14, color=col, alpha=0.7)
        p = within.loc[within["condition"] == conds[i], "p_wilcoxon_holm"].iloc[0]
        ax.text(i, dlim, stars(p), ha="center", va="bottom", fontsize=10)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([wrap(labels[c], 14) for c in conds], fontsize=8)
    ax.set_ylim(-dlim, dlim * 1.15)
    ax.set_title(f"{METRICS[m]}\nKruskal-Wallis across conditions: p = {between['kw_p']:.3g}", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylabel("Δ leaning (phase 2 − phase 1)")
    fig.suptitle("Change in leaning by condition (stars: Wilcoxon vs 0, Holm)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out, f"leaning_04_delta_by_condition_{m}.png"), dpi=300)
    plt.close(fig)


def engagement_phase_tests(long, wide, m, conds, A, B):
    """Participant-level mean engagement per phase, separately for reliable and unreliable
    articles; paired Wilcoxon phase B vs phase A in each condition x reliability (Holm over all)."""
    L = long[long["pid"].isin(wide.index)]
    rows, data = [], {}
    for c in conds:
        sub = L[L["condition"] == c]
        for col, rel in [("rate_rel", "reliable"), ("rate_unrel", "unreliable")]:
            piv = sub.pivot(index="pid", columns="phase", values=col).reindex(columns=[A, B]).dropna()
            d1, d2 = piv[A].values, piv[B].values
            data[(c, rel)] = (d1, d2)
            diff = d2 - d1
            p = np.nan
            if len(diff) >= 2 and np.any(diff != 0):
                try:
                    p = stats.wilcoxon(diff).pvalue
                except ValueError:
                    pass
            rows.append(dict(condition=c, reliability=rel, n=len(diff),
                             mean_phaseA=np.mean(d1) if len(d1) else np.nan,
                             mean_phaseB=np.mean(d2) if len(d2) else np.nan,
                             median_phaseA=np.median(d1) if len(d1) else np.nan,
                             median_phaseB=np.median(d2) if len(d2) else np.nan,
                             mean_diff=np.mean(diff) if len(diff) else np.nan,
                             p_wilcoxon=p))
    res = pd.DataFrame(rows)
    res["p_wilcoxon_holm"] = holm(res["p_wilcoxon"])
    return res, data


def fig_engagement_phase_box(res, data, m, conds, labels, A, B, out, rng):
    from matplotlib.patches import Patch
    mx = METRIC_MAX[m]
    allv = np.concatenate([np.r_[a, b] for a, b in data.values()]) if data else np.array([mx])
    top = min(mx, np.nanmax(allv)) if np.isfinite(allv).any() else mx
    top = max(top, 0.1 * mx)
    nc = len(conds)
    fig, axes = plt.subplots(1, nc, figsize=(3.6 * nc, 5), sharey=True)
    for i, (ax, c) in enumerate(zip(np.atleast_1d(axes), conds)):
        for g, (rel, color) in enumerate([("reliable", COL_REL), ("unreliable", COL_UNREL)]):
            d1, d2 = data[(c, rel)]
            x1, x2 = g - 0.18, g + 0.18
            bp = ax.boxplot([d1, d2], positions=[x1, x2], widths=0.3, patch_artist=True,
                            showfliers=False, medianprops=dict(color="black"))
            for patch, alpha, hatch in zip(bp["boxes"], [0.25, 0.6], [None, "//"]):
                patch.set_facecolor(color)
                patch.set_alpha(alpha)
                patch.set_edgecolor("0.25")
                if hatch:
                    patch.set_hatch(hatch)
            j1 = rng.uniform(-0.05, 0.05, len(d1))
            j2 = rng.uniform(-0.05, 0.05, len(d2))
            for a, b, ja, jb in zip(d1, d2, j1, j2):
                ax.plot([x1 + ja, x2 + jb], [a, b], color="0.55", lw=0.5, alpha=0.45, zorder=2)
            ax.scatter(x1 + j1, d1, s=12, color=color, edgecolor="white", lw=0.3, zorder=3)
            ax.scatter(x2 + j2, d2, s=12, color=color, edgecolor="white", lw=0.3, zorder=3)
            p = res.loc[(res["condition"] == c) & (res["reliability"] == rel), "p_wilcoxon_holm"].iloc[0]
            yb = top * 1.04
            ax.plot([x1, x1, x2, x2], [yb - top * 0.015, yb, yb, yb - top * 0.015], color="black", lw=0.8)
            ax.text(g, yb, stars(p), ha="center", va="bottom", fontsize=9)
        n = int(res.loc[res["condition"] == c, "n"].max())
        ax.set_title(f"{wrap(labels[c], 26)}\nn={n}", fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["reliable", "unreliable"])
        for lab, col in zip(ax.get_xticklabels(), [COL_REL, COL_UNREL]):
            lab.set_color(col)
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(-0.04 * top, top * 1.16)
        ax.grid(axis="y", alpha=0.3)
    ylab = ("Share of interactions engaged (0–1)" if m == "bin"
            else "Mean engagement score per interaction (0–3)")
    np.atleast_1d(axes)[0].set_ylabel(f"{ylab}\n(one point = one participant)")
    fig.legend(handles=[Patch(facecolor="0.85", edgecolor="0.25", label=f"Phase {A:g}"),
                        Patch(facecolor="0.6", edgecolor="0.25", hatch="//", label=f"Phase {B:g}")],
               loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"Engagement in phase {A:g} vs phase {B:g}, by condition and source reliability — "
                 f"{METRICS[m]}\n(stars: paired Wilcoxon phase {B:g} vs {A:g}, Holm across all panels)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    fig.savefig(os.path.join(out, f"leaning_05_engagement_by_phase_{m}.png"), dpi=300)
    plt.close(fig)


def fig_agreement_forest(agr, omni, m, conds, labels, out):
    fig, ax = plt.subplots(figsize=(7, 0.9 * len(conds) + 2.4))
    a = agr.set_index("condition").reindex(conds)
    y = np.arange(len(conds))[::-1]
    for i, c in enumerate(conds):
        r = a.loc[c]
        col = COND_COLORS[i % len(COND_COLORS)]
        if not np.isnan(r.CCC_ci_lo):
            ax.plot([r.CCC_ci_lo, r.CCC_ci_hi], [y[i], y[i]], color=col, lw=2.5)
        ax.plot(r.CCC, y[i], "o", color=col, ms=9, label="CCC [95% bootstrap CI]" if i == 0 else None)
        ax.plot(r.ICC_A1, y[i] - 0.18, "D", mfc="white", mec=col, ms=6,
                label="ICC(A,1)" if i == 0 else None)
    ax.axvline(0, color="grey", ls="--", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([wrap(labels[c], 26) for c in conds], fontsize=8.5)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Agreement phase 1 ↔ phase 2")
    ax.set_title(f"{METRICS[m]}\nomnibus permutation p = {omni:.3g}", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("In which condition is leaning most stable between phases?")
    fig.tight_layout(rect=[0, 0.07, 1, 0.93])
    fig.savefig(os.path.join(out, f"leaning_06_agreement_forest_{m}.png"), dpi=300)
    plt.close(fig)


def fig_between_conditions(PW, BETWEEN, conds, labels, out):
    """Pairwise comparison of Delta between treatments (Mann-Whitney, Holm), score vs binary."""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    ms = [m for m in ["eng", "bin"] if m in PW]
    nc = len(conds)
    cmap = ListedColormap(["#08306B", "#2171B5", "#6BAED6", "#F0F0F0"])
    norm = BoundaryNorm([0, .001, .01, .05, 1.0001], cmap.N)
    fig, axes = plt.subplots(1, len(ms), figsize=(6.2 * len(ms), 0.95 * nc + 3.6))
    for ax, m in zip(np.atleast_1d(axes), ms):
        pw = PW[m]
        M = np.full((nc, nc), np.nan)
        for _, r in pw.iterrows():
            i, j = conds.index(r.cond_1), conds.index(r.cond_2)
            lo, hi = max(i, j), min(i, j)   # lower triangle: row = later condition
            M[lo, hi] = r.p_holm
        Mm = np.ma.masked_invalid(M)
        ax.imshow(Mm, cmap=cmap, norm=norm)
        for i in range(nc):
            for j in range(nc):
                if i <= j:
                    continue
                rr = pw[((pw.cond_1 == conds[j]) & (pw.cond_2 == conds[i])) |
                        ((pw.cond_1 == conds[i]) & (pw.cond_2 == conds[j]))].iloc[0]
                mean_row = rr.mean_1 if rr.cond_1 == conds[i] else rr.mean_2
                mean_col = rr.mean_2 if rr.cond_1 == conds[i] else rr.mean_1
                p = M[i, j]
                col = "white" if (not np.isnan(p) and p < .01) else "black"
                ax.text(j, i, f"p = {p:.3g} {stars(p) if stars(p) != 'n.s.' else ''}\n"
                              f"ΔL̄ diff = {mean_row - mean_col:+.2f}",
                        ha="center", va="center", fontsize=8, color=col)
        short = [wrap(f"{c}: {labels[c]}", 18) for c in conds]
        ax.set_xticks(range(nc))
        ax.set_xticklabels(short, fontsize=7.5, rotation=0)
        ax.set_yticks(range(nc))
        ax.set_yticklabels(short, fontsize=7.5)
        ax.set_xticks(np.arange(-.5, nc), minor=True)
        ax.set_yticks(np.arange(-.5, nc), minor=True)
        ax.grid(which="minor", color="white", lw=2)
        ax.tick_params(which="minor", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{METRICS[m]}\nKruskal-Wallis across conditions: p = {BETWEEN[m]['kw_p']:.3g}",
                     fontsize=10)
    fig.legend(handles=[Patch(color=c, label=l) for c, l in
                        zip(cmap.colors, ["p < .001", "p < .01", "p < .05", "n.s."])],
               loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Δ leaning (phase 2 − phase 1): pairwise comparison between treatments\n"
                 "Mann-Whitney U, Holm-corrected; ΔL̄ diff = mean Δ of row condition − mean Δ of column condition",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0.06, 1, 0.9])
    fig.savefig(os.path.join(out, "leaning_07_between_conditions.png"), dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Supplementary: engagement by article and by topic x reliability
# --------------------------------------------------------------------------- #
def first_words(s, k=3):
    w = str(s).split()
    return " ".join(w[:k]) + (" …" if len(w) > k else "")


def supplementary_articles(df, sup_dir):
    need = ["article_uid", "topic", "title_shown"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        print(f"[WARNING] Supplementary plots skipped: missing columns {missing}")
        return
    os.makedirs(sup_dir, exist_ok=True)
    d = df.copy()
    d["article_uid"] = d["article_uid"].astype(str).str.strip()
    d["topic"] = d["topic"].astype(str).str.strip().replace("", "unknown")

    def most_common(s):
        s = s[s.astype(str).str.strip() != ""]
        return s.mode().iat[0] if len(s) else ""

    g = d.groupby("article_uid")
    art = pd.DataFrame({
        "topic": g["topic"].agg(most_common),
        "reliability": g["reliability_class"].agg(most_common),
        "outlet": g["outlet"].agg(most_common) if "outlet" in d.columns else "",
        "title": g["title_shown"].agg(most_common),
        "n_interactions": g["eng"].size(),
        "n_participants": g["pid"].nunique(),
        "mean_eng": g["eng"].mean(),
        "sd_eng": g["eng"].std(ddof=1),
        "p_readmore": g["readmore"].mean(),
        "p_readfull": g["readfull"].mean(),
        "p_like": g["like"].mean(),
        "p_engaged_bin": g["bin"].mean() if "bin" in d.columns else np.nan,
    })
    tcrit = stats.t.ppf(0.975, (art["n_interactions"] - 1).clip(lower=1))
    half = tcrit * art["sd_eng"] / np.sqrt(art["n_interactions"])
    art["ci_lo"] = art["mean_eng"] - half
    art["ci_hi"] = art["mean_eng"] + half
    art["label"] = art["topic"] + " | " + art["title"].map(first_words)
    art = art.sort_values("mean_eng", ascending=False)
    art.reset_index().to_csv(os.path.join(sup_dir, "articles_engagement.csv"), index=False)

    # test: do articles differ? (descriptive: interactions of the same participant are not independent)
    groups = [x["eng"].values for _, x in d.groupby("article_uid")]
    kw_art = np.nan
    if len(groups) >= 2:
        try:
            kw_art = stats.kruskal(*groups)[1]
        except ValueError:
            pass

    # ---- Plot 1: article ranking ----
    a = art.iloc[::-1]  # highest at the top
    n = len(a)
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.32 * n + 1.8)))
    colors = [COL_REL if r == "reliable" else COL_UNREL for r in a["reliability"]]
    y = np.arange(n)
    ax.barh(y, a["mean_eng"], color=colors, alpha=0.85, height=0.7)
    err = np.vstack([(a["mean_eng"] - a["ci_lo"]).fillna(0), (a["ci_hi"] - a["mean_eng"]).fillna(0)])
    ax.errorbar(a["mean_eng"], y, xerr=err, fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
    overall = d["eng"].mean()
    ax.axvline(overall, color="grey", ls="--", lw=1)
    ax.text(overall, n - 0.3, f" overall mean = {overall:.2f}", color="grey", fontsize=8, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(a["label"], fontsize=8)
    for lab, col in zip(ax.get_yticklabels(), colors):
        lab.set_color(col)
    xmax = min(SMAX, np.nanmax(a["ci_hi"].values) if n else SMAX)
    for yi, (m, k) in enumerate(zip(a["mean_eng"], a["n_interactions"])):
        ax.text(xmax * 1.02, yi, f"n={k}", va="center", fontsize=7, color="0.35")
    ax.set_xlim(0, max(0.5, xmax * 1.12))
    ax.set_ylim(-0.7, n - 0.3 + 0.6)
    ax.set_xlabel("Mean engagement score per interaction (0–3, ± 95% CI)")
    ax.grid(axis="x", alpha=0.3)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=COL_REL, label="reliable"), Patch(color=COL_UNREL, label="unreliable")],
               loc="lower center", ncol=2, fontsize=8, frameon=False)
    ax.set_title(f"Engagement by article (topic | first three words of title)\n"
                 f"Kruskal-Wallis across articles: p = {kw_art:.3g}", fontsize=10)
    fig.tight_layout(rect=[0, 0.4 / fig.get_figheight(), 1, 1])
    fig.savefig(os.path.join(sup_dir, "supp_01_articles_engagement.png"), dpi=300)
    plt.close(fig)

    # ---- Plot 2: boxplot topic x reliability (one point = one article) ----
    topics = sorted(art["topic"].unique())
    rels = ["reliable", "unreliable"]
    fig, ax = plt.subplots(figsize=(max(6, 1.9 * len(topics) + 2), 5))
    rng = np.random.default_rng(0)
    w = 0.34
    for j, rel in enumerate(rels):
        col = COL_REL if rel == "reliable" else COL_UNREL
        pos, data = [], []
        for i, t in enumerate(topics):
            v = art.loc[(art["topic"] == t) & (art["reliability"] == rel), "mean_eng"].values
            pos.append(i + (j - 0.5) * w)
            data.append(v)
        nonempty = [k for k, v in enumerate(data) if len(v)]
        if nonempty:
            bp = ax.boxplot([data[k] for k in nonempty], positions=[pos[k] for k in nonempty],
                            widths=w * 0.85, patch_artist=True, showfliers=False,
                            medianprops=dict(color="black"))
            for patch in bp["boxes"]:
                patch.set_facecolor(col)
                patch.set_alpha(0.35)
        for p, v in zip(pos, data):
            ax.scatter(p + rng.uniform(-w * 0.2, w * 0.2, len(v)), v, s=22, color=col,
                       edgecolor="white", lw=0.5, zorder=3)
    n_rel = [int(((art["topic"] == t) & (art["reliability"] == "reliable")).sum()) for t in topics]
    n_unr = [int(((art["topic"] == t) & (art["reliability"] == "unreliable")).sum()) for t in topics]
    ax.set_xticks(range(len(topics)))
    ax.set_xticklabels([f"{t}\n(articles: {a} rel, {b} unrel)" for t, a, b in zip(topics, n_rel, n_unr)],
                       fontsize=9)
    ax.set_ylabel("Mean engagement score per article (0–3)")
    ax.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COL_REL, alpha=0.6, label="reliable"),
                       Patch(color=COL_UNREL, alpha=0.6, label="unreliable")], fontsize=8)

    kw_t = mw_r = np.nan
    tg = [art.loc[art["topic"] == t, "mean_eng"].values for t in topics]
    if len([x for x in tg if len(x)]) >= 2:
        try:
            kw_t = stats.kruskal(*[x for x in tg if len(x)])[1]
        except ValueError:
            pass
    r1 = art.loc[art["reliability"] == "reliable", "mean_eng"].values
    r2 = art.loc[art["reliability"] == "unreliable", "mean_eng"].values
    if len(r1) and len(r2):
        try:
            mw_r = stats.mannwhitneyu(r1, r2, alternative="two-sided")[1]
        except ValueError:
            pass
    ax.set_title(f"Engagement by topic and reliability (one point = one article)\n"
                 f"topic: Kruskal-Wallis p = {kw_t:.3g}   |   reliable vs unreliable: Mann-Whitney p = {mw_r:.3g}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(sup_dir, "supp_02_topic_reliability_boxplot.png"), dpi=300)
    plt.close(fig)

    print(f"\nSUPPLEMENTARY - articles: {len(art)}, Kruskal-Wallis across articles p={kw_art:.4g}; "
          f"topic p={kw_t:.4g}; reliable vs unreliable (article level) p={mw_r:.4g}")
    print("Top 5 articles by engagement:")
    print(art.head(5)[["topic", "reliability", "label", "n_interactions", "mean_eng"]]
          .to_string(float_format=lambda x: f"{x:.2f}"))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Leaning score analysis reliable vs unreliable, phase 1 vs 2, by condition.")
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help=f"CSV path (default: most recent '{DATA_PATTERN}' in {RESULTS_DIR})")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Figures folder")
    ap.add_argument("--tables", default=None, help="Tables folder (default: <out>/leaning_tables)")
    ap.add_argument("--supp", default=None, help="Supplementary folder (default: <out>/supplementary)")
    ap.add_argument("--phase-a", type=float, default=1)
    ap.add_argument("--phase-b", type=float, default=2)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-no-prolific", action="store_true",
                    help="Also include sessions without prolific_pid (default: excluded)")
    ap.add_argument("--bin-rule", choices=["gt", "ge"], default="gt",
                    help="Binary engagement: 'gt' = E > median (default), 'ge' = E >= median")
    args = ap.parse_args()

    A, B = args.phase_a, args.phase_b
    out = args.out
    tables = args.tables or os.path.join(out, "leaning_tables")
    os.makedirs(out, exist_ok=True)
    os.makedirs(tables, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    data_path = find_data(args.data)
    print(f"Data file: {data_path}")
    df, conds, labels = load_data(data_path, A, B, keep_only_prolific=not args.include_no_prolific)
    print(f"\nData: {len(df)} rows (reliable/unreliable only), {df['pid'].nunique()} participants, "
          f"conditions: {conds}")
    print(df.groupby(["condition", "phase"])["pid"].nunique().unstack("phase").rename_axis(None).to_string(), "\n")

    pd.set_option("display.width", 200)
    fmt = lambda x: f"{x:.3f}"  # noqa: E731

    # ---- engagement distribution and binary version ----
    med, share = add_binary(df, args.bin_rule)
    op = ">" if args.bin_rule == "gt" else "≥"
    METRICS["bin"] = f"Binary engagement (E {op} median = {med:g})"
    print("=" * 90)
    print("ENGAGEMENT DISTRIBUTION (per user-article interaction)")
    print("-" * 90)
    dist = pd.crosstab(df["eng"], [df["phase"], df["reliability_class"]], normalize="columns") * 100
    dist["all"] = df["eng"].value_counts(normalize=True).sort_index() * 100
    print("% of interactions with each score E (columns: phase, reliability):")
    print(dist.round(1).to_string())
    print(f"\nMedian E = {med:g}  ->  binary engagement: E {op} {med:g}  ->  {share:.1f}% of interactions engaged")
    print(df.groupby("reliability_class")["bin"].mean().mul(100).round(1)
          .rename("% engaged").to_string(), "\n")
    fig_engagement_distribution(df, med, args.bin_rule, out, tables, A, B)

    LONG, WIDE, WITHIN, BETWEEN, BASE, PW, AGR, AGRPW, OMNI, GEE, EPH = ({} for _ in range(11))

    for m in METRICS:
        long, wide, dropped = compute_leaning(df, m, A, B)
        LONG[m], WIDE[m] = long, wide
        WITHIN[m] = within_tests(wide, conds)
        BETWEEN[m], PW[m] = between_tests(wide, conds, "delta")
        BASE[m], _ = between_tests(wide, conds, "L_A")
        AGR[m], AGRPW[m], OMNI[m] = agreement(wide, conds, args.n_boot, args.n_perm, rng)
        GEE[m] = gee_model(df, m, B, tables)
        EPH[m] = engagement_phase_tests(long, wide, m, conds, A, B)

        wide.reset_index().to_csv(os.path.join(tables, f"leaning_participant_{m}.csv"), index=False)
        WITHIN[m].to_csv(os.path.join(tables, f"within_phase_test_{m}.csv"), index=False)
        PW[m].to_csv(os.path.join(tables, f"between_delta_pairwise_{m}.csv"), index=False)
        AGR[m].to_csv(os.path.join(tables, f"agreement_{m}.csv"), index=False)
        AGRPW[m].to_csv(os.path.join(tables, f"agreement_pairwise_{m}.csv"), index=False)
        EPH[m][0].to_csv(os.path.join(tables, f"engagement_phase_test_{m}.csv"), index=False)

        print("=" * 90)
        print(f"METRIC: {METRICS[m]}   (participants with both phases: {len(wide)}"
              f"{f', excluded {dropped} without reliable/unreliable articles in one phase' if dropped else ''})")
        print("-" * 90)
        print("1) Leaning phase 1 vs phase 2 within each condition")
        cols = ["condition", "n", "mean_L_phaseA", "mean_L_phaseB", "mean_delta",
                "delta_ci_lo", "delta_ci_hi", "cohen_dz", "p_wilcoxon_holm", "p_ttest_holm"]
        print(WITHIN[m][cols].to_string(index=False, float_format=fmt))
        print(f"\n2) Does Delta differ across conditions?  Kruskal-Wallis H={BETWEEN[m]['kw_H']:.3f}, "
              f"p={BETWEEN[m]['kw_p']:.4g} | ANOVA F={BETWEEN[m]['anova_F']:.3f}, p={BETWEEN[m]['anova_p']:.4g}")
        print(f"   (baseline check, leaning in phase {A:g} across conditions: KW p={BASE[m]['kw_p']:.4g})")
        if not PW[m].empty:
            print(PW[m][["cond_1", "cond_2", "mean_1", "mean_2", "p", "p_holm"]].to_string(index=False, float_format=fmt))
        print("\n3) Agreement of leaning between phases (higher = more stable)")
        print(AGR[m][["condition", "n", "CCC", "CCC_ci_lo", "CCC_ci_hi", "ICC_A1", "spearman_rho",
                      "mean_abs_delta", "pct_same_sign"]].to_string(index=False, float_format=fmt))
        best = AGR[m].dropna(subset=["CCC"])
        if not best.empty:
            b = best.loc[best["CCC"].idxmax()]
            print(f"   -> Highest agreement: condition {b.condition} ({labels[b.condition]}), CCC={b.CCC:.3f}")
        print(f"   Omnibus permutation test (CCC equal across conditions): p={OMNI[m]:.4g}")
        if not AGRPW[m].empty:
            print(AGRPW[m][["cond_1", "cond_2", "diff", "p_perm", "p_perm_holm"]].to_string(index=False, float_format=fmt))
        print(f"\n4) GEE ({'logistic' if m == 'bin' else 'linear'}, article level): {GEE[m][1]}")
        print(f"\n5) Engagement phase {A:g} vs {B:g} by condition and reliability (participant means, paired Wilcoxon)")
        print(EPH[m][0][["condition", "reliability", "n", "mean_phaseA", "mean_phaseB", "mean_diff",
                         "p_wilcoxon", "p_wilcoxon_holm"]].to_string(index=False, float_format=fmt))
        print()

    summary = []
    for m in METRICS:
        s = {"metric": m, "kw_delta_p": BETWEEN[m]["kw_p"], "anova_delta_p": BETWEEN[m]["anova_p"],
             "omnibus_CCC_p": OMNI[m], "gee_3way_p": GEE[m][0]}
        for _, r in AGR[m].iterrows():
            s[f"CCC_cond{r.condition}"] = r.CCC
        summary.append(s)
    pd.DataFrame(summary).to_csv(os.path.join(tables, "summary.csv"), index=False)

    for m in METRICS:
        fig_leaning_by_phase(WIDE[m], m, conds, labels, A, B, out)
        fig_paired(WIDE[m], m, conds, labels, WITHIN[m], A, B, out, rng)
        fig_agreement_scatter(WIDE[m], m, conds, labels, AGR[m], A, B, out, rng)
        fig_delta(WIDE[m], WITHIN[m], BETWEEN[m], m, conds, labels, out, rng)
        fig_engagement_phase_box(EPH[m][0], EPH[m][1], m, conds, labels, A, B, out, rng)
        fig_agreement_forest(AGR[m], OMNI[m], m, conds, labels, out)
    fig_between_conditions(PW, BETWEEN, conds, labels, out)

    supp = args.supp or os.path.join(out, "supplementary")
    supplementary_articles(df, supp)

    print(f"\nFigures saved in: {out}")
    print(f"Supplementary in: {supp}")
    print(f"Tables saved in:  {tables}")


if __name__ == "__main__":
    main()