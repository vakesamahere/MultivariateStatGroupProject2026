#!/usr/bin/env python3
"""23_audit_fixes.py --- intervals for the D-claims in notes/AUDIT_inference.md sections 2-3.

Reads only the committed caches (data_derived/, out/, figures/); writes out/audit_fixes/*.
Nothing under paper/ is modified and src/01-22 are not re-run.  Every interval names its
method and its resampling unit in the table it lands in.

Items: D1/D3/D11/D12 (per-line means + separability), D2 (burst bootstrap of the lag-profile
argmax), D7 (location rate ratios + deviance-difference bootstrap), D10 (per-line recovery
slopes with date-clustered SEs), D13 (rank-1 minus rank-2 gap over the 23 bursts), B11
(profile-likelihood intervals for the two e-folding lengths, plus a paired date bootstrap of
their ratio).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
DATA, DERIVED = ROOT / "data", ROOT / "data_derived"
FIG = ROOT / "figures" / "mechanism"
OUT = ROOT / "out" / "audit_fixes"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20240501
NBOOT_DATE = 5000      # D1, D3, D11, D12
NBOOT_BURST = 5000     # D2, D13
NBOOT_DEV = 1000       # D7
TICK_S, OFF, CARRIER = 30, 99.0, "T3420"
TICK_H = TICK_S / 3600

_LOG: list[str] = []


def say(*lines: object) -> None:
    for ln in lines:
        print(ln)
        _LOG.append(str(ln))


def flush() -> None:
    (OUT / "audit_fixes.txt").write_text("\n".join(_LOG) + "\n")


def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / name, index=False)
    say(f"  -> out/audit_fixes/{name}  ({len(df)} rows)")


def pct(a: np.ndarray, q: float = 2.5) -> float:
    return float(np.nanpercentile(a, q))


def boot_p(diff: np.ndarray) -> float:
    """Two-sided percentile p-value for H0: diff = 0."""
    lo, hi = float(np.mean(diff <= 0)), float(np.mean(diff >= 0))
    return float(min(1.0, 2 * min(lo, hi)))


def cluster_ratio(sum_df: pd.DataFrame, cnt_df: pd.DataFrame, nboot: int, seed: int = SEED):
    """Percentile bootstrap of sum/count per column, resampling whole rows (dates).

    Returns (M, point) where M is (nboot, n_groups) and point the observed ratio.
    """
    S = sum_df.to_numpy(dtype=float)
    N = cnt_df.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, S.shape[0], size=(nboot, S.shape[0]))
    Ms = S[idx].sum(axis=1)
    Mn = N[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        M = np.where(Mn > 0, Ms / Mn, np.nan)
        point = np.where(N.sum(axis=0) > 0, S.sum(axis=0) / N.sum(axis=0), np.nan)
    return M, point


def date_cluster_means(df: pd.DataFrame, group: str, value: str, nboot: int = NBOOT_DATE):
    """Per-group mean with a percentile CI from resampling whole dates."""
    d = df.dropna(subset=[value]).copy()
    d["date"] = pd.to_datetime(d["date"])
    d[group] = d[group].astype(str)
    dates = np.array(sorted(d["date"].unique()))
    groups = sorted(d[group].unique())
    S = (d.groupby(["date", group])[value].sum().unstack(fill_value=0.0)
         .reindex(index=dates, columns=groups, fill_value=0.0))
    N = (d.groupby(["date", group])[value].size().unstack(fill_value=0)
         .reindex(index=dates, columns=groups, fill_value=0))
    M, point = cluster_ratio(S, N, nboot)
    rows = []
    for j, g in enumerate(groups):
        rows.append({"group": g, "n": int(N[g].sum()), "n_dates": int((N[g] > 0).sum()),
                     "mean": float(point[j]), "ci_lo": pct(M[:, j]), "ci_hi": pct(M[:, j], 97.5),
                     "se_boot": float(np.nanstd(M[:, j], ddof=1))})
    return pd.DataFrame(rows), M, groups


# ======================================================================================
# D1, D3, D11, D12 -- per-line means with date-clustered intervals and separability
# ======================================================================================
def section_impact() -> None:
    t0 = time.time()
    say("=" * 88)
    say("D1 / D3 / D11 / D12  per-line means, date-clustered 95% percentile intervals")
    say("=" * 88)
    imp = pd.read_csv(DERIVED / "incident_impact.csv")
    imp["line_id"] = imp.line_id.astype(str)
    imp["delay_per_incident_min"] = imp.knock_on_train_min + imp.own_delay_train_min
    nd = imp.date.nunique()
    say(f"incidents {len(imp)}, dates {nd} (the resampling unit for every interval below), "
        f"draws {NBOOT_DATE}")
    say("method: percentile bootstrap over dates (whole service days resampled with replacement); "
        "a day's incidents enter and leave together.")

    trav = pd.read_pickle(DERIVED / "traversals.pkl")
    trav["line_id"] = trav.line_id.astype(str)
    seg = (trav.groupby(["date", "line_id"]).travel_s
           .agg(["sum", "size"]).reset_index())
    seg["date"] = pd.to_datetime(seg.date)
    seg_dates = np.array(sorted(seg.date.unique()))
    S = seg.pivot(index="date", columns="line_id", values="sum").reindex(index=seg_dates).fillna(0.0)
    N = seg.pivot(index="date", columns="line_id", values="size").reindex(index=seg_dates).fillna(0)
    M_seg, pt_seg = cluster_ratio(S, N, NBOOT_DATE)
    seg_rows = []
    for j, ln in enumerate(S.columns):
        seg_rows.append({"quantity": "segment_running_s", "unit": "s", "line_id": str(ln),
                         "n": int(N[ln].sum()), "n_dates": int((N[ln] > 0).sum()),
                         "mean": float(pt_seg[j]), "ci_lo": pct(M_seg[:, j]),
                         "ci_hi": pct(M_seg[:, j], 97.5),
                         "se_boot": float(np.nanstd(M_seg[:, j], ddof=1))})
    say(f"segment running time: {len(trav):,} traversals over {len(seg_dates)} dates; "
        "resampling unit = service date, all 105 dates (a timetable property, not incident-weighted)")

    quant = [("boundary_headway_s", "s", "block-boundary headway"),
             ("knock_on_train_min", "min", "knock-on delay"),
             ("delay_per_incident_min", "min", "delay per incident"),
             ("n_followers_delayed", "count", "followers delayed"),
             ("clear_after_recovery_s", "s", "recovery time")]
    cells, Mstore = [], {}
    for q, unit, _ in quant:
        tab, M, groups = date_cluster_means(imp, "line_id", q)
        Mstore[q] = (M, groups)
        for _, r in tab.iterrows():
            cells.append({"quantity": q, "unit": unit, "line_id": r.group, "n": r.n,
                          "n_dates": r.n_dates, "mean": r["mean"], "ci_lo": r.ci_lo,
                          "ci_hi": r.ci_hi, "se_boot": r.se_boot})
    for r in seg_rows:
        cells.append(r)
    cells = pd.DataFrame(cells)
    cells["ci_method"] = "percentile bootstrap over dates"
    cells["resample_unit"] = "service date"

    # ---- D1 contrast: Line 2 minus Line 8 on boundary headway
    M, groups = Mstore["boundary_headway_s"]
    j2, j8 = groups.index("2"), groups.index("8")
    diff = M[:, j2] - M[:, j8]
    sub2 = imp.loc[imp.line_id == "2", "boundary_headway_s"].dropna()
    sub8 = imp.loc[imp.line_id == "8", "boundary_headway_s"].dropna()
    welch = stats.ttest_ind(sub2, sub8, equal_var=False)
    d1 = cells[cells.quantity == "boundary_headway_s"].copy()
    d1 = d1[["line_id", "n", "n_dates", "mean", "ci_lo", "ci_hi", "se_boot",
             "ci_method", "resample_unit"]]
    d1.insert(0, "row_type", "line")
    d1["mean_min"] = d1["mean"] / 60
    d1["ci_lo_min"] = d1["ci_lo"] / 60
    d1["ci_hi_min"] = d1["ci_hi"] / 60
    d1["p_boot"] = np.nan
    d1["p_welch"] = np.nan
    d1["overlap_min"] = np.nan
    r2 = d1.set_index("line_id").loc["2"]
    r8 = d1.set_index("line_id").loc["8"]
    overlap = max(0.0, min(r2.ci_hi, r8.ci_hi) - max(r2.ci_lo, r8.ci_lo))
    d1 = pd.concat([d1, pd.DataFrame([{
        "row_type": "contrast", "line_id": "2 minus 8", "n": len(sub2) + len(sub8),
        "n_dates": int(imp[imp.line_id.isin(["2", "8"])].date.nunique()),
        "mean": float(np.mean(diff)), "ci_lo": pct(diff), "ci_hi": pct(diff, 97.5),
        "se_boot": float(np.std(diff, ddof=1)), "ci_method": "percentile bootstrap over dates",
        "resample_unit": "service date", "mean_min": float(np.mean(diff)) / 60,
        "ci_lo_min": pct(diff) / 60, "ci_hi_min": pct(diff, 97.5) / 60,
        "p_boot": boot_p(diff), "p_welch": float(welch.pvalue), "overlap_min": overlap / 60}])],
        ignore_index=True)
    save(d1, "d1_boundary_by_line.csv")
    say(f"D1 boundary headway: Line 2 {r2['mean']/60:.2f} min [{r2.ci_lo/60:.2f}, {r2.ci_hi/60:.2f}], "
        f"Line 8 {r8['mean']/60:.2f} min [{r8.ci_lo/60:.2f}, {r8.ci_hi/60:.2f}]")
    say(f"D1 contrast 2-8 = {np.mean(diff)/60:+.2f} min [{pct(diff)/60:+.2f}, {pct(diff,97.5)/60:+.2f}] "
        f"(p_boot {boot_p(diff):.4f}); Welch p = {welch.pvalue:.4f}; "
        f"intervals overlap by {overlap/60:.2f} min")
    flush()

    # ---- separability verdicts for the paper's five superlatives (+ the "intermediate" claim)
    claims = [
        ("knock_on_train_min", "2", "top", "Line 2 carries the largest knock-on delay per incident"),
        ("delay_per_incident_min", "2", "top", "Line 2 has the largest mean delay per incident"),
        ("n_followers_delayed", "12", "bottom", "Line 12 has the fewest followers per incident"),
        ("clear_after_recovery_s", "12", "bottom", "Line 12 has the shortest recovery"),
        ("segment_running_s", "2", "top", "Line 2 has the longest mean segment running time"),
    ]

    def verdict(q: str, claimed: str, direction: str) -> dict:
        c = cells[cells.quantity == q].set_index("line_id")
        asc = direction == "bottom"
        order = c["mean"].sort_values(ascending=asc).index.tolist()
        rival = order[1] if order[0] == claimed else order[0]
        a, b = c.loc[claimed], c.loc[rival]
        if asc:                       # claim: claimed is the smallest
            gap = float(b.ci_lo - a.ci_hi)
        else:                         # claim: claimed is the largest
            gap = float(a.ci_lo - b.ci_hi)
        return {"claimed_line": claimed, "rival_line": rival,
                "claimed_is_extreme": order[0] == claimed,
                "claimed_bound": float(a.ci_lo if not asc else a.ci_hi),
                "rival_bound": float(b.ci_hi if not asc else b.ci_lo),
                "gap": gap, "separable": bool(gap > 0),
                "overlap": float(max(0.0, min(a.ci_hi, b.ci_hi) - max(a.ci_lo, b.ci_lo)))}

    vrows = []
    for q, claimed, direction, text in claims:
        v = verdict(q, claimed, direction)
        vrows.append({"row_type": "verdict", "quantity": q,
                      "unit": cells.loc[cells.quantity == q, "unit"].iloc[0], "line_id": "",
                      "n": np.nan, "n_dates": np.nan, "mean": np.nan, "ci_lo": np.nan,
                      "ci_hi": np.nan, "se_boot": np.nan,
                      "ci_method": "percentile bootstrap over dates", "resample_unit": "service date",
                      "claim": text, **v})
        say(f"  {q}: claimed {claimed} [{v['claimed_bound']:.2f}] vs {v['rival_line']} "
            f"[{v['rival_bound']:.2f}] -> gap {v['gap']:+.2f}, "
            f"{'SEPARABLE' if v['separable'] else 'overlap ' + format(v['overlap'], '.2f')}"
            f"{'' if v['claimed_is_extreme'] else ' (claim not even rank 1)'}")
    c8 = cells[cells.quantity == "delay_per_incident_min"].set_index("line_id")
    up = float(c8.loc["2", "ci_lo"] - c8.loc["8", "ci_hi"])
    dn = float(c8.loc["8", "ci_lo"] - c8.loc["12", "ci_hi"])
    say(f"  Line 8 'intermediate' on delay per incident: below Line 2 gap {up:+.2f}, "
        f"above Line 12 gap {dn:+.2f}")
    for other, gap, text in [
            ("2", float(c8.loc["2", "ci_lo"] - c8.loc["8", "ci_hi"]),
             "Line 8 is intermediate on delay per incident: below Line 2"),
            ("12", float(c8.loc["8", "ci_lo"] - c8.loc["12", "ci_hi"]),
             "Line 8 is intermediate on delay per incident: above Line 12")]:
        vrows.append({"row_type": "verdict", "quantity": "delay_per_incident_min", "unit": "min",
                      "line_id": "8", "claim": text, "claimed_line": "8", "rival_line": other,
                      "claimed_is_extreme": False,
                      "claimed_bound": float(c8.loc["8", "ci_lo"] if other == "12"
                                             else c8.loc["8", "ci_hi"]),
                      "rival_bound": float(c8.loc["2", "ci_lo"] if other == "2"
                                           else c8.loc["12", "ci_hi"]),
                      "gap": gap, "separable": bool(gap > 0),
                      "overlap": float(max(0.0, c8.loc[["8", other], "ci_hi"].min()
                                           - c8.loc[["8", other], "ci_lo"].max()))})
    d3 = pd.concat([cells.assign(row_type="cell", claim="", claimed_line="", rival_line="",
                                 claimed_is_extreme=np.nan, claimed_bound=np.nan,
                                 rival_bound=np.nan, gap=np.nan, separable=np.nan,
                                 overlap=np.nan),
                    pd.DataFrame(vrows)], ignore_index=True)
    d3 = d3[["row_type", "quantity", "unit", "line_id", "n", "n_dates", "mean", "ci_lo", "ci_hi",
             "se_boot", "ci_method", "resample_unit", "claim", "claimed_line", "rival_line",
             "claimed_is_extreme", "claimed_bound", "rival_bound", "gap", "separable", "overlap"]]
    save(d3, "d3_impact_by_line.csv")
    say("  per-line means (min / s / count):")
    for q, unit, _ in quant:
        r = cells[cells.quantity == q]
        say(f"    {q:24s} " + "  ".join(f"L{x.line_id} {x['mean']:.1f} [{x.ci_lo:.1f}, {x.ci_hi:.1f}]"
                                       for _, x in r.iterrows()))
    r = cells[cells.quantity == "segment_running_s"]
    say(f"    {'segment_running_s':24s} " + "  ".join(
        f"L{x.line_id} {x['mean']:.1f} [{x.ci_lo:.1f}, {x.ci_hi:.1f}]" for _, x in r.iterrows()))
    say(f"  ({time.time() - t0:.0f} s)")
    flush()


# ======================================================================================
# D10 -- per-line recovery slopes with date-clustered SEs, plus the elasticity interval
# ======================================================================================
def section_recovery_slopes() -> None:
    t0 = time.time()
    say("=" * 88)
    say("D10  per-line recovery slopes, date-clustered SEs")
    say("=" * 88)
    imp = pd.read_csv(DERIVED / "incident_impact.csv")
    imp["line_id"] = imp.line_id.astype(str)
    imp["date"] = pd.to_datetime(imp["date"])
    imp["rec_min"] = imp.clear_after_recovery_s / 60
    imp["D"] = imp.knock_on_train_min + imp.own_delay_train_min
    codes = {d: i for i, d in enumerate(sorted(imp.date.unique()))}
    grp = imp.date.map(codes).to_numpy()
    n_cl = len(codes)
    say(f"n = {len(imp)}, clusters (dates) = {n_cl}, cov_type='cluster' on date, "
        f"t with df = {n_cl - 1}; dependent variable in minutes")
    rows = []
    for xvar, label in [("D", "rec_min ~ D * C(line),  D = knock-on + own delay (the paper's D)"),
                        ("own_delay_train_min", "rec_min ~ own_delay * C(line)")]:
        res = smf.ols(f"rec_min ~ {xvar} * C(line_id)", data=imp).fit(
            cov_type="cluster", cov_kwds={"groups": grp})
        pn = list(res.params.index)
        iD = pn.index(xvar)
        inter = [p for p in pn if p.startswith(f"{xvar}:C(line_id)")]
        say(f"  [{label}]  base line = {sorted(imp.line_id.unique())[0]}")
        for ln in ["2", "8", "10", "12"]:
            c = np.zeros(len(pn))
            c[iD] = 1.0
            nm = f"{xvar}:C(line_id)[T.{ln}]"
            if nm in pn:
                c[pn.index(nm)] = 1.0
            tt = res.t_test(c.reshape(1, -1))
            lo, hi = tt.conf_int(alpha=0.05)[0]
            n_ln = int((imp.line_id == ln).sum())
            rows.append({"row_type": "line", "model": label, "line_id": ln, "n": n_ln,
                         "n_dates": int(imp.loc[imp.line_id == ln, "date"].nunique()),
                         "slope": float(c @ res.params.to_numpy()),
                         "se_cluster": float(np.squeeze(tt.sd)), "ci_lo": float(lo),
                         "ci_hi": float(hi), "stat": float(np.squeeze(tt.tvalue)), "df1": 1.0,
                         "df2": float(n_cl - 1), "p": float(np.squeeze(tt.pvalue)),
                         "includes_0p5": np.nan, "includes_1p0": np.nan})
            say(f"    Line {ln:2s}: slope {rows[-1]['slope']:+.2f} "
                f"[{lo:+.2f}, {hi:+.2f}] (SE {np.squeeze(tt.sd):.2f}, "
                f"p {np.squeeze(tt.pvalue):.4f}), n = {n_ln}")
        R = np.zeros((len(inter), len(pn)))
        for i, p in enumerate(inter):
            R[i, pn.index(p)] = 1.0
        ft = res.f_test(R)
        fval, pval = float(np.squeeze(ft.fvalue)), float(np.squeeze(ft.pvalue))
        say(f"    interaction test (the {len(inter)} {xvar}:line terms = 0): "
            f"F({len(inter)}, {n_cl - 1}) = {fval:.2f}, p = {pval:.4f}")
        rows.append({"row_type": "interaction", "model": label, "line_id": "all", "n": len(imp),
                     "n_dates": n_cl, "slope": np.nan, "se_cluster": np.nan, "ci_lo": np.nan,
                     "ci_hi": np.nan, "stat": fval, "df1": float(len(inter)),
                     "df2": float(n_cl - 1), "p": pval, "includes_0p5": np.nan,
                     "includes_1p0": np.nan})
    rec = pd.read_csv(ROOT / "out" / "recovery" / "recovery_models.csv")
    for model, label in [("B2 log-log: log(rec_min) ~ log(D)", "loglog pooled"),
                         ("B2 log-log + line", "loglog + line")]:
        r = rec[(rec.model == model) & (rec.term == "logD")].iloc[0]
        rows.append({"row_type": "elasticity", "model": label, "line_id": "pooled",
                     "n": int(r.n), "n_dates": np.nan, "slope": float(r.estimate),
                     "se_cluster": np.nan, "ci_lo": float(r.ci_lo), "ci_hi": float(r.ci_hi),
                     "stat": np.nan, "df1": np.nan, "df2": np.nan, "p": float(r.p_value),
                     "includes_0p5": bool(r.ci_lo <= 0.5 <= r.ci_hi),
                     "includes_1p0": bool(r.ci_lo <= 1.0 <= r.ci_hi)})
        say(f"  elasticity ({label}): {r.estimate:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}] "
            f"-- includes 0.5: {bool(r.ci_lo <= 0.5 <= r.ci_hi)}, includes 1.0: "
            f"{bool(r.ci_lo <= 1.0 <= r.ci_hi)} (unclustered interval, as published)")
    # the published elasticity interval already uses cluster-robust SEs (the refit below
    # reproduces it exactly), but statsmodels keeps df_resid = 141 for them, so it uses
    # normal rather than t(37) critical values; rebuild it with the small-cluster df
    dl = imp.assign(log_rec=np.log(imp.rec_min), log_D=np.log(imp.D))
    res = smf.ols("log_rec ~ log_D", data=dl).fit(cov_type="cluster", cov_kwds={"groups": grp})
    e0, se = float(res.params["log_D"]), float(res.bse["log_D"])
    q = float(stats.t.ppf(0.975, n_cl - 1))
    lo, hi = e0 - q * se, e0 + q * se
    pub = rec[(rec.model == "B2 log-log: log(rec_min) ~ log(D)") & (rec.term == "logD")].iloc[0]
    say(f"  elasticity refit with cluster SE: {e0:.4f} (SE {se:.4f}); published "
        f"[{pub.ci_lo:.4f}, {pub.ci_hi:.4f}] -- cluster SE reproduces it "
        f"{abs(e0 - pub.estimate) < 1e-6 and abs(se - (pub.ci_hi - pub.ci_lo) / 3.9199) < 2e-3}")
    rows.append({"row_type": "elasticity", "model": "loglog pooled, cluster SE, t(37)",
                 "line_id": "pooled", "n": len(dl), "n_dates": n_cl, "slope": e0,
                 "se_cluster": se, "ci_lo": float(lo), "ci_hi": float(hi),
                 "stat": float(res.tvalues["log_D"]), "df1": 1.0, "df2": float(n_cl - 1),
                 "p": float(res.pvalues["log_D"]), "includes_0p5": bool(lo <= 0.5 <= hi),
                 "includes_1p0": bool(lo <= 1.0 <= hi)})
    say(f"  elasticity (loglog pooled, cluster SE, t({n_cl - 1})): {e0:.3f} [{lo:.3f}, {hi:.3f}] "
        f"-- includes 0.5: {bool(lo <= 0.5 <= hi)}, includes 1.0: {bool(lo <= 1.0 <= hi)}")
    r = rec[(rec.model == "B2 linear: rec_min ~ D") & (rec.term == "D")].iloc[0]
    say(f"  pooled linear slope (unclustered, as published): {r.estimate:.3f} "
        f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]")
    save(pd.DataFrame(rows), "d10_recovery_slopes.csv")
    say(f"  ({time.time() - t0:.0f} s)")
    flush()


# ======================================================================================
# Shared: rebuild the 30-s label grid per date (same construction as src/03, src/08)
# ======================================================================================
def build_grids():
    t0 = time.time()
    say("=" * 88)
    say("grid rebuild: 140 trains x 105 dates x 2,222 ticks of 30 s (read-only on the caches)")
    say("=" * 88)
    trav = pd.read_pickle(DERIVED / "traversals.pkl")
    bd = pd.read_csv(DERIVED / "breakdowns.csv",
                     parse_dates=["onset_ts", "recovery_ts", "date"])
    st = pd.read_csv(DATA / "station_information.csv", encoding="utf-8-sig", dtype=str)

    dw = trav[trav.dwell_s.notna()]
    iv = pd.concat([
        pd.DataFrame({"train_id": dw.train_id.values, "line_id": dw.line_id.values,
                      "date": dw.date.values, "loc": dw.from_station.astype(str).values,
                      "start": (dw.dep_ts - pd.to_timedelta(dw.dwell_s, unit="s")).values,
                      "end": dw.dep_ts.values}),
        pd.DataFrame({"train_id": trav.train_id.values, "line_id": trav.line_id.values,
                      "date": trav.date.values, "loc": trav.segment.astype(str).values,
                      "start": trav.dep_ts.values, "end": trav.arr_ts.values}),
    ], ignore_index=True)
    del dw
    for c in ["train_id", "line_id", "loc"]:
        iv[c] = iv[c].astype(str).astype("category")
    iv["date"] = pd.to_datetime(iv.date)
    labels = list(iv["loc"].cat.categories)
    trains = list(iv.train_id.cat.categories)
    dates = np.array(sorted(iv.date.unique()))
    train_idx = {t: i for i, t in enumerate(trains)}

    def sec_of_day(ts, day):
        return (ts - day).dt.total_seconds().to_numpy()

    t_start_s = float(np.floor(sec_of_day(iv.start, iv.date).min() / 3600) * 3600)
    nt = int(np.ceil((sec_of_day(iv.end, iv.date).max() - t_start_s) / TICK_S)) + 1

    nodes = sorted(set(st.station_id))
    adj = {s: set() for s in nodes}
    for a, b in zip(st.station_id, st.next_station_id):
        if isinstance(b, str):
            adj[a].add(b)
            adj[b].add(a)
    hop = pd.DataFrame(9999, index=nodes, columns=nodes, dtype=np.int32)
    for s in nodes:
        seen, frontier = {s: 0}, [s]
        while frontier:
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in seen:
                        seen[v] = seen[u] + 1
                        nxt.append(v)
            frontier = nxt
        for v, x in seen.items():
            hop.at[s, v] = x
    lab_nodes = [lb.split(" -> ") if " -> " in lb else [lb] for lb in labels]
    sent = len(labels)
    ld = np.full((sent + 1, sent + 1), OFF, dtype=np.float32)
    for i, a in enumerate(lab_nodes):
        for j, b in enumerate(lab_nodes):
            h = hop.loc[a, b].to_numpy().min()
            if len(a) == 2 and len(b) == 2:
                ld[i, j] = 0.0 if set(a) == set(b) else 1.0 + h
            elif len(a) == 2 or len(b) == 2:
                ld[i, j] = 0.5 + h
            else:
                ld[i, j] = h
    say(f"  labels {len(labels)}, trains {len(trains)}, dates {len(dates)}, ticks {nt}, "
        f"t_start {t_start_s:.0f} s")

    iv["di"] = iv.date.map({pd.Timestamp(d): i for i, d in enumerate(dates)})
    iv["code"] = iv["loc"].cat.codes
    iv["a"] = np.ceil((sec_of_day(iv.start, iv.date) - t_start_s) / TICK_S).astype(np.int64)
    iv["b"] = np.ceil((sec_of_day(iv.end, iv.date) - t_start_s) / TICK_S).astype(np.int64)
    bd["ti"] = bd.train_id.map(train_idx)
    bd["di"] = bd.date.map({pd.Timestamp(d): i for i, d in enumerate(dates)})
    bd["tick"] = np.floor((sec_of_day(bd.onset_ts, bd.date) - t_start_s) / TICK_S).astype(np.int64)
    ci = train_idx[CARRIER]

    grids = {}
    for di, day in enumerate(dates):
        sub = iv[iv.di == di]
        n_t = np.maximum(sub.b.to_numpy() - sub.a.to_numpy(), 0)
        rep = np.repeat(np.arange(len(sub)), n_t)
        offs = np.arange(n_t.sum()) - np.repeat(np.cumsum(n_t) - n_t, n_t)
        g = np.full((len(trains), nt), -1, dtype=np.int16)
        g[sub.train_id.cat.codes.to_numpy()[rep], sub.a.to_numpy()[rep] + offs] = \
            sub.code.to_numpy()[rep]
        grids[day] = g

    got = np.array([grids[d][t, k] for d, t, k in
                    zip(bd.date.values, bd.ti.values, bd.tick.values)])
    ok = np.mean([labels[c] == l if c >= 0 else False for c, l in zip(got, bd.location)])
    say(f"  validation: victim label recovered at onset for {ok:.1%} of the {len(bd)} breakdowns")
    say(f"  ({time.time() - t0:.0f} s)")
    flush()
    return dict(labels=labels, trains=trains, dates=dates, nt=nt, grids=grids, ld=ld,
                sent=sent, bd=bd, ci=ci, train_idx=train_idx, t_start_s=t_start_s)


def d_code(d):
    d = np.nan_to_num(np.asarray(d, dtype=np.float64), nan=OFF)
    return np.where(d >= OFF, 7, np.minimum(np.round(d * 2), 6)).astype(np.int64)


# ======================================================================================
# D2 -- burst-level bootstrap of the argmax of the two lag profiles
# ======================================================================================
def section_lag(G: dict) -> None:
    t0 = time.time()
    say("=" * 88)
    say("D2  bootstrap of the lag-profile argmax, resampling units = bursts")
    say("=" * 88)
    scan = pd.read_csv(FIG / "tables" / "carrier_lag_scan.csv")
    bd = G["bd"]
    lags = np.array(sorted(scan.lag_min.unique()))
    ths = sorted(scan.d_max.unique())
    groups = ["Line 10", "Lines 2/8/12"]
    l10 = (bd.line_id.astype(str) == "10").to_numpy()
    gmask = {"Line 10": l10, "Lines 2/8/12": ~l10}

    dist = np.full((len(bd), len(lags)), OFF)
    for li, L in enumerate(lags):
        lt = int(round(L * 60 / TICK_S))
        for i, (day, t, k) in enumerate(zip(bd.date.values, bd.ti.values, bd.tick.values)):
            g = G["grids"][day]
            j = k + lt
            cl = g[G["ci"], j] if 0 <= j < G["nt"] else -1
            vl = g[t, k]
            dist[i, li] = G["ld"][vl if vl >= 0 else G["sent"], cl if cl >= 0 else G["sent"]]
    say(f"  {len(bd)} breakdowns x {len(lags)} lags; carrier {CARRIER}; "
        f"distance in hops on the label graph (0 = same undirected segment)")

    imp = pd.read_csv(DERIVED / "incident_impact.csv",
                      usecols=["incident_id", "in_burst", "burst_id"])
    bmap = dict(zip(imp.incident_id, imp.burst_id))
    inb = dict(zip(imp.incident_id, imp.in_burst))
    cluster = np.array([f"burst {bmap[i]}" if inb[i] else f"lone {i}"
                        for i in bd.incident_id])
    cl_ids = sorted(set(cluster))
    cl_idx = np.array([cl_ids.index(c) for c in cluster])
    n_cl = len(cl_ids)
    say(f"  resampling unit: burst ({sum(inb.values())} events in "
        f"{len(set(cluster)) - sum(not inb[i] for i in bd.incident_id)} bursts, "
        f"{sum(not inb[i] for i in bd.incident_id)} lone events are their own unit) -> {n_cl} units")
    # the exposure is per (group, d_max, lag): the carrier's path is not in service at
    # every tick, so the number of train-hours within d_max hops of it varies with the lag
    hmat = {g: {th: scan[(scan.victim_line == g) & (scan.d_max == th)]
                .set_index("lag_min").loc[lags, "h"].to_numpy() for th in ths}
            for g in groups}
    hv = np.concatenate([hmat[g][th] for g in groups for th in ths])
    say(f"  exposure per (group, d_max, lag): {hv.min():.1f}-{hv.max():.1f} train-h "
        "(held fixed at its observed value across draws: it is a timetable property, "
        "not an outcome of the incident)")
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n_cl, size=(NBOOT_BURST, n_cl))

    rows, prof_rows = [], []
    for grp in groups:
        gm = gmask[grp]
        K = {}
        for th in ths:
            km = np.zeros((n_cl, len(lags)))
            hit = (dist <= th).astype(float)
            for ci_ in range(n_cl):
                m = gm & (cl_idx == ci_)
                if m.any():
                    km[ci_] = hit[m].sum(axis=0)
            K[th] = km
        for th in ths:
            h = hmat[grp][th]                               # (n_lags,)
            kd = K[th][idx].sum(axis=1)                     # (NBOOT, n_lags)
            rate = 1000 * kd / h
            am = np.argmax(rate, axis=1)
            top = rate.max(axis=1)
            tied = (rate == top[:, None]).sum(axis=1)
            kobs = K[th].sum(axis=0)
            prof = pd.DataFrame({"group": grp, "profile": f"d_max={th}", "lag_min": lags,
                                 "victim_line": grp, "d_max": th,
                                 "k": kobs.astype(int), "h": h,
                                 "rate": 1000 * kobs / h, "ci_lo": np.percentile(rate, 2.5, axis=0),
                                 "ci_hi": np.percentile(rate, 97.5, axis=0),
                                 "share_argmax": np.bincount(am, minlength=len(lags)) / len(am)})
            prof_rows.append(prof)
            claim = -0.5 if grp == "Line 10" else 2.0   # scan lag: negative = carrier earlier
            ci_i = int(np.argmin(np.abs(lags - claim)))
            ci_lag = float(lags[ci_i])
            rows.append({"group": grp, "profile": f"d_max={th}", "claimed_peak_lag_min": claim,
                         "mode_lag_min": float(lags[np.bincount(am, minlength=len(lags)).argmax()]),
                         "q025": float(np.percentile(lags[am], 2.5)),
                         "q975": float(np.percentile(lags[am], 97.5)),
                         "share_claimed_peak": float(np.mean(lags[am] == ci_lag)),
                         "share_zero": float(np.mean(lags[am] == 0.0)),
                         "share_tied": float(np.mean(tied > 1)),
                         "k_at_claimed": int(kobs[ci_i]),
                         "rate_at_claimed": float(1000 * kobs[ci_i] / h[ci_i]),
                         "h": float(h[ci_i])})
        # joint: maximise over the three thresholds first, then take the lag argmax
        R = np.stack([1000 * K[th][idx].sum(axis=1) / hmat[grp][th] for th in ths])  # (3,NBOOT,L)
        best = R.max(axis=0)                                # (NBOOT, L)
        am = np.argmax(best, axis=1)
        claim = -0.5 if grp == "Line 10" else 2.0   # scan lag: negative = carrier earlier
        ci_i = int(np.argmin(np.abs(lags - claim)))
        ci_lag = float(lags[ci_i])
        rows.append({"group": grp, "profile": "joint (best d_max)", "claimed_peak_lag_min": claim,
                     "mode_lag_min": float(lags[np.bincount(am, minlength=len(lags)).argmax()]),
                     "q025": float(np.percentile(lags[am], 2.5)),
                     "q975": float(np.percentile(lags[am], 97.5)),
                     "share_claimed_peak": float(np.mean(lags[am] == ci_lag)),
                     "share_zero": float(np.mean(lags[am] == 0.0)),
                     "share_tied": np.nan, "k_at_claimed": np.nan,
                     "rate_at_claimed": np.nan, "h": np.nan})
    prof = pd.concat(prof_rows, ignore_index=True)
    prof.insert(0, "row_type", "profile")
    chk = prof.merge(scan, on=["victim_line", "lag_min", "d_max"], how="left")
    dk = int(np.abs(chk.k_x.to_numpy() - chk.k_y.to_numpy()).max())
    dr = float(np.abs(chk.rate.to_numpy() - chk.rate_per_1000h.to_numpy()).max())
    say(f"  check against carrier_lag_scan.csv: max |dk| = {dk}, "
        f"max |d rate| = {dr:.2e} per 1000 train-h over {len(chk)} cells")
    summ = pd.DataFrame(rows)
    summ.insert(0, "row_type", "argmax_summary")
    out = pd.concat([prof, summ], ignore_index=True)
    out = out[["row_type", "group", "profile", "lag_min", "k", "h", "rate", "ci_lo", "ci_hi",
               "share_argmax", "claimed_peak_lag_min", "mode_lag_min", "q025", "q975",
               "share_claimed_peak", "share_zero", "share_tied", "k_at_claimed",
               "rate_at_claimed"]]
    out["n_draws"] = NBOOT_BURST
    out["ci_method"] = "percentile bootstrap over bursts"
    save(out, "d2_lag_argmax_boot.csv")
    for r in rows:
        say(f"  {r['group']:12s} {r['profile']:20s} argmax mode {r['mode_lag_min']:+.1f} min, "
            f"2.5-97.5% [{r['q025']:+.1f}, {r['q975']:+.1f}], "
            f"share on claimed {r['share_claimed_peak']:.2f}, share on 0 {r['share_zero']:.2f}")
    say(f"  ({time.time() - t0:.0f} s)")
    flush()


# ======================================================================================
# D7 -- location rate ratios and a date-clustered bootstrap of the deviance difference
# ======================================================================================
def section_mediation(G: dict) -> None:
    t0 = time.time()
    say("=" * 88)
    say("D7  location rate ratios (crude vs adjusted) and deviance-difference bootstrap")
    say("=" * 88)
    lm = pd.read_csv(FIG / "tables" / "location_mediation.csv")
    rr = lm.rename(columns={"RR_adj_carrier_distance": "rr_adj",
                            "lo_adj_carrier_distance": "rr_adj_lo",
                            "hi_adj_carrier_distance": "rr_adj_hi"})
    rr = rr[["loc_class", "RR_crude", "lo_crude", "hi_crude", "rr_adj", "rr_adj_lo", "rr_adj_hi"]]
    rr = rr.rename(columns={"RR_crude": "rr_crude", "lo_crude": "rr_crude_lo",
                            "hi_crude": "rr_crude_hi"})
    rr["crude_over_adj"] = rr.rr_crude / rr.rr_adj
    rr["note"] = np.where(rr.rr_crude < 1e-6,
                          "no events in this class: the rate ratio is not identified", "")
    rr["ci_method"] = "Wald on the Poisson GLM (as published, unclustered)"
    rr["resample_unit"] = "none"
    save(rr, "d7_location_rr.csv")
    live = rr[rr.rr_crude >= 1e-6]
    say(f"  {len(rr)} location classes, {len(live)} with events (the paper's six plus the "
        "reference, which is not listed); "
        f"crude RR {live.rr_crude.min():.2f}-{live.rr_crude.max():.2f}, "
        f"adjusted {live.rr_adj.min():.2f}-{live.rr_adj.max():.2f}; "
        f"largest crude-to-adjusted collapse {live.crude_over_adj.max():.1f}x "
        f"({live.loc[live.crude_over_adj.idxmax(), 'loc_class']})")

    locs = pd.read_csv(FIG / "tables" / "location_rates.csv")
    lclass_u = dict(zip(locs.loc_u, locs.hop_class + " / " + locs.kind))
    classes = sorted(set(lclass_u.values()))
    lab_class = np.array([classes.index(lclass_u[" - ".join(sorted(lb.split(" -> ")))
                                        if " -> " in lb else lb]) for lb in G["labels"]] + [-1])
    ref = "off-L10, >=2 hops / segment"
    bd = G["bd"]
    expo = np.zeros((len(G["dates"]), len(classes), 8))
    kev = np.zeros((len(G["dates"]), len(classes), 8))
    di_of = {pd.Timestamp(d): i for i, d in enumerate(G["dates"])}
    for di, day in enumerate(G["dates"]):
        g = G["grids"][day]
        gi = np.where(g < 0, G["sent"], g)
        dc = G["ld"][gi, gi[G["ci"]][None, :]]
        dc[G["ci"]] = np.nan
        svc = g >= 0
        svc[G["ci"]] = False
        key = (lab_class[gi] * 8 + d_code(dc))[svc]
        expo[di] = np.bincount(key, minlength=len(classes) * 8).reshape(len(classes), 8) * TICK_H
    for day, t, k in zip(bd.date.values, bd.ti.values, bd.tick.values):
        g = G["grids"][day]
        d = G["ld"][g[t, k], g[G["ci"], k]]
        kev[di_of[pd.Timestamp(day)], lab_class[g[t, k]], d_code(d)] += 1
    h = expo.sum(axis=0)
    k = kev.sum(axis=0)
    cells = [(i, j) for i in range(len(classes)) for j in range(8) if h[i, j] > 0]
    say(f"  {int(k.sum())} events over {h.sum():,.0f} train-h in {len(cells)} occupied "
        f"(location class x hop-code) cells; reference class '{ref}'")
    loc_lv = [i for i in range(len(classes)) if i != classes.index(ref)]
    dc_lv = [j for j in range(8) if j != 0]
    X = np.zeros((len(cells), 1 + len(loc_lv) + len(dc_lv)))
    X[:, 0] = 1.0
    for r, (i, j) in enumerate(cells):
        if i in loc_lv:
            X[r, 1 + loc_lv.index(i)] = 1.0
        if j in dc_lv:
            X[r, 1 + len(loc_lv) + dc_lv.index(j)] = 1.0
    n_loc = 1 + len(loc_lv)
    Xloc, Xdist, Xboth = X[:, :n_loc], X[:, [0] + list(range(n_loc, X.shape[1]))], X
    kobs = np.array([k[i, j] for i, j in cells], dtype=float)
    hobs = np.array([h[i, j] for i, j in cells], dtype=float)
    off = np.log(hobs)
    fam = sm.families.Poisson()

    def fit(y, Xm, offset):
        return sm.GLM(y, Xm, family=fam, offset=offset).fit()

    def lrs(y):
        a = fit(y, Xloc, off).llf
        b = fit(y, Xdist, off).llf
        c = fit(y, Xboth, off).llf
        return 2 * (c - b), 2 * (c - a)

    lr_loc, lr_dist = lrs(kobs)
    say(f"  full data: LR(location | distance) = {lr_loc:.1f} on {len(loc_lv)} df, "
        f"LR(distance | location) = {lr_dist:.1f} on {len(dc_lv)} df, "
        f"difference = {lr_dist - lr_loc:.1f}")

    dates_ev = np.array(sorted({pd.Timestamp(d) for d in bd.date.values}))
    Kd = np.zeros((len(cells), len(dates_ev)))
    for di, d in enumerate(dates_ev):
        Kd[:, di] = np.array([kev[di_of[pd.Timestamp(d)], i, j] for i, j in cells])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(dates_ev), size=(NBOOT_DEV, len(dates_ev)))
    draws, fails = [], 0
    for b in range(NBOOT_DEV):
        y = Kd[:, idx[b]].sum(axis=1)
        if y.sum() == 0:
            fails += 1
            continue
        try:
            a, c = lrs(y)
        except Exception:
            fails += 1
            continue
        draws.append({"row_type": "draw", "draw": b, "lr_loc_given_dist": a,
                      "lr_dist_given_loc": c, "diff": c - a})
    D = pd.DataFrame(draws)
    say(f"  bootstrap: {len(D)} of {NBOOT_DEV} date-resamples fitted ({fails} dropped); "
        f"resampling unit = service date ({len(dates_ev)} dates with >=1 breakdown), "
        f"exposure held at its observed value")
    diff = D["diff"].to_numpy()
    summ = pd.DataFrame([{
        "row_type": "summary", "draw": -1, "lr_loc_given_dist": lr_loc, "lr_dist_given_loc": lr_dist,
        "diff": lr_dist - lr_loc, "n_draws": len(D), "n_failed": fails,
        "ci_lo": pct(diff), "ci_hi": pct(diff, 97.5), "ci_lo_loc": pct(D.lr_loc_given_dist.to_numpy()),
        "ci_hi_loc": pct(D.lr_loc_given_dist.to_numpy(), 97.5),
        "ci_lo_dist": pct(D.lr_dist_given_loc.to_numpy()),
        "ci_hi_dist": pct(D.lr_dist_given_loc.to_numpy(), 97.5),
        "share_diff_positive": float(np.mean(diff > 0)), "min_diff": float(diff.min())}])
    out = pd.concat([D, summ], ignore_index=True)
    out["ci_method"] = "percentile bootstrap over dates"
    out["resample_unit"] = "service date"
    save(out, "d7_deviance_boot.csv")
    say(f"  difference {lr_dist - lr_loc:.1f} [{pct(diff):.1f}, {pct(diff, 97.5):.1f}]; "
        f"positive in {np.mean(diff > 0):.3f} of draws; "
        f"LR(location|distance) {lr_loc:.1f} [{pct(D.lr_loc_given_dist.to_numpy()):.1f}, "
        f"{pct(D.lr_loc_given_dist.to_numpy(), 97.5):.1f}], "
        f"LR(distance|location) {lr_dist:.1f} [{pct(D.lr_dist_given_loc.to_numpy()):.1f}, "
        f"{pct(D.lr_dist_given_loc.to_numpy(), 97.5):.1f}]")
    say(f"  ({time.time() - t0:.0f} s)")
    flush()


# ======================================================================================
# D13 -- rank-1 minus rank-2 gap over the 23 bursts
# ======================================================================================
def section_rank_gap(G: dict) -> None:
    t0 = time.time()
    say("=" * 88)
    say("D13  rank-1 minus rank-2 mean hop distance, per burst")
    say("=" * 88)
    imp = pd.read_csv(DERIVED / "incident_impact.csv",
                      usecols=["incident_id", "in_burst", "burst_id"])
    bmap = dict(zip(imp.incident_id, imp.burst_id))
    inb = dict(zip(imp.incident_id, imp.in_burst))
    bd = G["bd"]
    ev = [(day, t, k, bmap[i]) for day, t, k, i in
          zip(bd.date.values, bd.ti.values, bd.tick.values, bd.incident_id) if inb[i]]
    bursts = sorted({e[3] for e in ev})
    say(f"  {len(ev)} events in {len(bursts)} bursts; ranking all {len(G['trains'])} trains by "
        "their mean hop distance to the burst's onset locations")
    say("  each event's own victim train is dropped from its own column (it is at distance 0 "
        "by construction); this is the convention that reproduces the paper's 0.75 vs 2.67")
    rows = []
    for b in bursts:
        g = [e for e in ev if e[3] == b]
        M = np.zeros((len(G["trains"]), len(g)))
        for j, (day, t, k, _) in enumerate(g):
            gg = G["grids"][day]
            gi = np.where(gg < 0, G["sent"], gg)
            M[:, j] = G["ld"][gi[:, k], gi[t, k]]
        for j, e in enumerate(g):
            M[e[1], j] = np.nan
        m = np.nanmean(M, axis=1)
        o = np.argsort(m)
        rows.append({"row_type": "burst", "burst_id": b, "n_events": len(g),
                     "rank1_train": G["trains"][o[0]], "d_rank1": float(m[o[0]]),
                     "rank2_train": G["trains"][o[1]], "d_rank2": float(m[o[1]]),
                     "gap_rank1_minus_rank2": float(m[o[0]] - m[o[1]])})
    R = pd.DataFrame(rows)
    gap = R.gap_rank1_minus_rank2.to_numpy()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(gap), size=(NBOOT_BURST, len(gap)))
    med = np.median(gap[idx], axis=1)
    say(f"  T3420 is rank 1 in {int((R.rank1_train == CARRIER).sum())} of {len(R)} bursts; "
        f"median d(rank1) {R.d_rank1.median():.2f}, median d(rank2) {R.d_rank2.median():.2f} hops")
    say(f"  gap (rank1 - rank2): median {np.median(gap):+.2f}, IQR [{np.percentile(gap,25):+.2f}, "
        f"{np.percentile(gap,75):+.2f}], min {gap.min():+.2f} (burst "
        f"{R.loc[R.gap_rank1_minus_rank2.idxmin(), 'burst_id']}), max {gap.max():+.2f}; "
        f"bootstrap 95% CI of the median [{pct(med):+.2f}, {pct(med, 97.5):+.2f}]")
    summ = pd.DataFrame([{
        "row_type": "summary", "burst_id": "all", "n_events": len(ev),
        "rank1_train": "", "d_rank1": R.d_rank1.median(), "rank2_train": "",
        "d_rank2": R.d_rank2.median(), "gap_rank1_minus_rank2": float(np.median(gap)),
        "median": float(np.median(gap)), "q25": float(np.percentile(gap, 25)),
        "q75": float(np.percentile(gap, 75)), "min": float(gap.min()), "max": float(gap.max()),
        "boot_lo": pct(med), "boot_hi": pct(med, 97.5),
        "share_gap_ge_0": float(np.mean(med >= 0)),
        "share_bursts_gap_le_-1": float(np.mean(gap <= -1.0)),
        "n_bursts": len(R), "n_rank1_is_T3420": int((R.rank1_train == CARRIER).sum()),
        "n_draws": NBOOT_BURST}])
    out = pd.concat([R, summ], ignore_index=True)
    out["ci_method"] = "percentile bootstrap over bursts"
    out["resample_unit"] = "burst"
    save(out, "d13_runner_up_gap.csv")
    say(f"  gap >= 0 (ordering lost) in {np.mean(med >= 0):.3f} of bootstrap draws; "
        f"{np.mean(gap <= -1.0):.2f} of the {len(R)} bursts separate the top two by >= 1 hop")
    say(f"  ({time.time() - t0:.0f} s)")
    flush()


# ======================================================================================
# B11 -- profile-likelihood intervals for the two e-folding lengths
# ======================================================================================
def _core_fit(b: pd.DataFrame, efold: float | None = None):
    """Poisson core+background: rate = bg + A*exp(-d/L) per 1000 train-h, offset log(h)."""
    d = b.d_m.to_numpy()
    k = b.events.to_numpy(float)
    h = b.exposure_train_h.to_numpy(float)

    def nll(p):
        mu = h / 1000 * (np.exp(p[0]) + np.exp(p[1] - d / efold))
        return float((mu - k * np.log(mu)).sum())

    best = None
    for s in (-1 / 3000, -1 / 600, -1 / 1200, -1 / 150):
        r = optimize.minimize(nll, [np.log(0.1), np.log(100.0)], method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    return best


def _geo_bins(sfx: str) -> pd.DataFrame:
    """The 13 distance bins with an exposure-weighted mean distance per bin."""
    b = pd.read_csv(ROOT / "out" / "geo" / f"rate_by_distance{sfx}.csv")
    rs = pd.read_csv(ROOT / "out" / "geo" / f"radius_scan{sfx}.csv")
    cum = rs.exposure_in_train_h.to_numpy()
    r = rs.r_m.to_numpy()
    w, c = np.diff(cum), (r[:-1] + r[1:]) / 2
    d = []
    for lo, hi in zip(b.lo_m, b.hi_m):
        m = (c >= lo) & (c < hi)
        d.append(35000.0 if hi >= 1e9 else float((w[m] * c[m]).sum() / w[m].sum()))
    b["d_m"] = d
    return b


def _fit3(d: np.ndarray, h: np.ndarray, k: np.ndarray):
    """Free fit over (bg, L, A) for fixed bin distances and exposures."""
    def nll(p):
        mu = h / 1000 * (np.exp(p[0]) + np.exp(p[2] - d / np.exp(p[1])))
        return float((mu - k * np.log(mu)).sum())

    best = None
    for s in (np.log(600), np.log(1500), np.log(300)):
        r3 = optimize.minimize(nll, [np.log(0.1), s, np.log(100.0)], method="L-BFGS-B")
        if best is None or r3.fun < best.fun:
            best = r3
    return float(np.exp(best.x[1])), best


def _deviance(b: pd.DataFrame, nll: float) -> float:
    k = b.events.to_numpy(float)
    return float(2 * (nll + np.sum(k * np.log(np.where(k > 0, k, 1.0))) - k.sum()))


def section_efold() -> dict[str, float]:
    t0 = time.time()
    say("=" * 88)
    say("B11  profile-likelihood 95% intervals for the two core e-folding lengths")
    say("=" * 88)
    grids, summ = [], []
    for metric, sfx in [("straight", ""), ("network", "_network")]:
        b = _geo_bins(sfx)
        say(f"  {metric}: {len(b)} bins, {int(b.events.sum())} events, "
            f"{b.exposure_train_h.sum():,.0f} train-h; bin distance = exposure-weighted mean of "
            "the 50 m ring centres (last bin, 10 km+, set to 35 km)")
        # free fit over (bg, A, L) to get the MLE and the deviance scale
        dd, kk, hh = b.d_m.to_numpy(), b.events.to_numpy(float), b.exposure_train_h.to_numpy(float)
        mle, best = _fit3(dd, hh, kk)
        dev_min = _deviance(b, best.fun)
        say(f"  MLE e-fold {mle:.1f} m, background {np.exp(best.x[0]):.4f} per 1000 train-h, "
            f"deviance {dev_min:.1f}, AIC {dev_min + 2 * 3:.1f}")
        curve = []
        for L in np.arange(150.0, 6000.1, 10.0):
            f = _core_fit(b, L)
            curve.append({"row_type": "grid", "metric": metric, "efold_m": L,
                          "deviance": _deviance(b, f.fun),
                          "background": float(np.exp(f.x[0])),
                          "amplitude": float(np.exp(f.x[1]))})
        C = pd.DataFrame(curve)
        C["delta_deviance"] = C.deviance - C.deviance.min()
        inside = C[C.delta_deviance <= 3.8415]
        lo, hi = float(inside.efold_m.min()), float(inside.efold_m.max())
        say(f"  profile: delta-deviance <= 3.8415 on [{lo:.0f}, {hi:.0f}] m "
            f"(grid 150-6000 m, step 10 m, {len(C)} points); minimum {C.deviance.min():.1f}")
        grids.append(C)
        summ.append({"row_type": "summary", "metric": metric, "efold_m": mle,
                     "deviance": dev_min, "ci_lo": lo, "ci_hi": hi,
                     "background": float(np.exp(best.x[0])),
                     "amplitude": float(np.exp(best.x[2])),
                     "delta_deviance": 0.0, "n_bins": len(b),
                     "n_events": int(b.events.sum()), "min_deviance": float(C.deviance.min())})
    out = pd.concat([pd.concat(grids, ignore_index=True), pd.DataFrame(summ)], ignore_index=True)
    out["ci_method"] = "profile likelihood (delta-deviance <= 3.8415)"
    out["resample_unit"] = "none (exposure-weighted 50 m histogram)"
    save(out, "b11_efold_profile.csv")
    s = pd.DataFrame(summ).set_index("metric")
    say(f"  straight {s.loc['straight','efold_m']:.0f} m [{s.loc['straight','ci_lo']:.0f}, "
        f"{s.loc['straight','ci_hi']:.0f}]; network {s.loc['network','efold_m']:.0f} m "
        f"[{s.loc['network','ci_lo']:.0f}, {s.loc['network','ci_hi']:.0f}]")
    say(f"  ({time.time() - t0:.0f} s)")
    flush()
    return {m: float(s.loc[m, "efold_m"]) for m in ("straight", "network")}


def section_efold_ratio(ef: dict[str, float]) -> None:
    """Paired date bootstrap of the network/straight e-folding ratio: same 142 events in both."""
    t0 = time.time()
    say("=" * 88)
    say("B11b  paired date bootstrap of the two e-folding lengths (same events, both metrics)")
    say("=" * 88)
    EV = pd.read_csv(ROOT / "out" / "geo" / "events_geo.csv")
    II = pd.read_csv(DERIVED / "incident_impact.csv")[["incident_id", "date"]]
    EV = EV.merge(II, on="incident_id", how="left")
    keep = EV.straight_m.notna() & EV.network_m.notna() & EV.date.notna()
    EV = EV[keep].reset_index(drop=True)
    say(f"  {len(EV)} of 143 events carry both distances and a date; "
        f"{EV.date.nunique()} distinct dates")
    B, cats, obs = {}, {}, {}
    for metric, sfx in (("straight", ""), ("network", "_network")):
        b = _geo_bins(sfx)
        edges = np.append(b.lo_m.to_numpy()[1:], np.inf)
        cat = np.digitize(EV[f"{metric}_m"].to_numpy(float), edges)
        k = np.bincount(cat, minlength=len(b)).astype(float)
        if not np.array_equal(k, b.events.to_numpy(float)):
            raise SystemExit(f"binning check failed for {metric}: {k} vs {b.events.to_numpy()}")
        B[metric], cats[metric], obs[metric] = b, cat, k
    say("  binning check: both metrics reproduce the committed event counts per bin")
    dates = np.array(sorted(EV.date.unique()))
    d_of = {d: i for i, d in enumerate(dates)}
    ev_date = EV.date.map(d_of).to_numpy()
    rng = np.random.default_rng(SEED + 11)
    rows, failed = [], 0
    for b_i in range(NBOOT_DEV):
        draw = rng.choice(len(dates), size=len(dates), replace=True)
        w = np.bincount(draw, minlength=len(dates)).astype(float)
        out = {"row_type": "draw", "boot_id": b_i}
        for metric in ("straight", "network"):
            b = B[metric]
            k = np.bincount(cats[metric], weights=w[ev_date], minlength=len(b))
            L, _ = _fit3(b.d_m.to_numpy(), b.exposure_train_h.to_numpy(float), k)
            out[f"efold_{metric}"] = L
        if not (np.isfinite(out["efold_straight"]) and np.isfinite(out["efold_network"])):
            failed += 1
            continue
        out["ratio_network_over_straight"] = out["efold_network"] / out["efold_straight"]
        rows.append(out)
    D = pd.DataFrame(rows)
    rat = D.ratio_network_over_straight.to_numpy()
    pt = ef["network"] / ef["straight"]
    say(f"  {len(D)} draws used, {failed} failed; ratio of point estimates {pt:.3f}")
    say(f"  ratio 95% CI [{pct(rat):.2f}, {pct(rat, 97.5):.2f}], median {np.median(rat):.2f}; "
        f"network > straight in {np.mean(rat > 1):.3f} of draws")
    say(f"  straight e-fold 95% CI [{pct(D.efold_straight.to_numpy()):.0f}, "
        f"{pct(D.efold_straight.to_numpy(), 97.5):.0f}] m; network "
        f"[{pct(D.efold_network.to_numpy()):.0f}, {pct(D.efold_network.to_numpy(), 97.5):.0f}] m")
    summ = pd.DataFrame([{"row_type": "summary", "boot_id": -1,
                          "efold_straight": ef["straight"], "efold_network": ef["network"],
                          "ratio_network_over_straight": pt,
                          "ci_lo": pct(rat), "ci_hi": pct(rat, 97.5),
                          "median": float(np.median(rat)),
                          "share_ratio_gt_1": float(np.mean(rat > 1)),
                          "n_draws": len(D), "n_failed": failed,
                          "n_events": len(EV), "n_dates": len(dates)}])
    out = pd.concat([D, summ], ignore_index=True)
    out["ci_method"] = "percentile bootstrap over dates (both metrics refit on the same draw)"
    out["resample_unit"] = "service date"
    save(out, "b11_efold_ratio_boot.csv")
    say(f"  ({time.time() - t0:.0f} s)")
    flush()


def main() -> None:
    t0 = time.time()
    say(f"23_audit_fixes.py  {time.strftime('%Y-%m-%d %H:%M')}  seed {SEED}  "
        f"numpy {np.__version__}  pandas {pd.__version__}  statsmodels {sm.__version__}")
    say("all intervals are percentile bootstraps unless the table says otherwise; "
        "every table carries its method and resampling unit")
    section_impact()
    section_recovery_slopes()
    G = build_grids()
    section_lag(G)
    section_mediation(G)
    section_rank_gap(G)
    EF = section_efold()
    section_efold_ratio(EF)
    say("=" * 88)
    say(f"done in {time.time() - t0:.0f} s; tables in out/audit_fixes/, log in "
        "out/audit_fixes/audit_fixes.txt")
    flush()


if __name__ == "__main__":
    main()
