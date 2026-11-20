"""11_recovery.py -- how long does the line stay off schedule after a breakdown, and what sets it?

B1 pins the recovery-time definition and reproduces it. The definition lives in src/04_impact.py:595 and its
distribution in notes/04_impact.md:135; it is quoted, not re-derived:

    last_ts = max timestamp of every attributed delayed event of incident k
    clear_after_recovery_s = (last_ts - recovery_ts).total_seconds()      # 04_impact.py:595
    clear_after_onset_s    = (last_ts - onset_ts).total_seconds()         # 04_impact.py:596

B2 tests the brief's hypothesis recovery ~ accumulated delay / per-trip absorption margin. The margin is built
per line from the timetable itself (h = headway, delta = separation at the downstream block boundary, turnaround
= layover at the terminal before the next trip, seg = mean segment running time), because the running-time
channel of the brief's formula is empty in this timetable: excess_travel_s is >= 0 for every one of the 4.19M
traversals and the median is 0, so scheduled minus minimum running time is exactly 0 on all 310 segments and the
only absorption left is the block-boundary slack and the terminal layover.

B3 screens the other candidate determinants (duration, burst membership, location class, line, peak, how many
incidents that day, queue length, distance to the terminal) univariately and then jointly, with date-clustered
intervals and a Benjamini-Hochberg correction across the univariate screen. B4 reports what is predictable.

Outputs: out/recovery/recovery_table.csv (one row per incident, every covariate),
out/recovery/recovery_models.csv (coefficient, CI, p and fit per model),
out/recovery/margin_by_line.csv (the per-line timetable margins of B2)
Run: python3 src/11_recovery.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA, DERIVED = ROOT / "data", ROOT / "data_derived"
OUT = ROOT / "out" / "recovery"
OUT.mkdir(parents=True, exist_ok=True)

LINES = [2, 8, 10, 12]
COEFS: list[dict] = []      # -> out/recovery/recovery_models.csv


def log(msg: str = "", end: str = "\n") -> None:
    print(msg, end=end, flush=True)


def section(title: str) -> None:
    log(f"\n## {title}")


def q(s: pd.Series, name: str, unit: str = "") -> str:
    """Distribution line in the format of notes/04_impact.md, so B1 is a like-for-like comparison."""
    return (f"{name} n={s.notna().sum()}, mean {s.mean():.1f}, min {s.min():.0f}, p25 {s.quantile(.25):.0f}, "
            f"median {s.median():.0f}, p75 {s.quantile(.75):.1f}, p90 {s.quantile(.9):.1f}, max {s.max():.0f}"
            f"{unit}")


def record(model: str, term: str, est: float, lo: float, hi: float, p: float | None, n: int,
           fit: float | None, fit_name: str, scale: str = "") -> None:
    COEFS.append({"model": model, "term": term, "estimate": est, "ci_lo": lo, "ci_hi": hi, "p_value": p,
                  "n": n, fit_name: float(fit) if fit is not None else np.nan, "fit": fit_name,
                  "scale": scale})


def cluster_ols(y: np.ndarray, x: np.ndarray, names: list[str], groups: np.ndarray, model: str,
                fit_name: str = "R2") -> sm.regression.linear_model.RegressionResultsWrapper:
    """OLS with standard errors clustered by service date; records every coefficient."""
    res = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": groups})
    ci = res.conf_int()
    for i, nm in enumerate(names):
        record(model, nm, float(res.params[i]), float(ci[i, 0]), float(ci[i, 1]), float(res.pvalues[i]),
               int(res.nobs), float(res.rsquared), fit_name)
    return res


def cluster_glm(y: np.ndarray, x: np.ndarray, names: list[str], groups: np.ndarray, model: str,
                family: sm.families.Family) -> sm.genmod.generalized_linear_model.GLMResultsWrapper:
    """GLM with log link and date-clustered SE; records coefficients and deviance explained."""
    res = sm.GLM(y, x, family=family).fit(cov_type="cluster", cov_kwds={"groups": groups})
    ci = res.conf_int()
    for i, nm in enumerate(names):
        record(model, nm, float(res.params[i]), float(ci[i, 0]), float(ci[i, 1]), float(res.pvalues[i]),
               int(res.nobs), float(1 - res.deviance / res.null_deviance), "deviance explained")
    return res


# ================================================================================================
# 0. The recovery table: one row per incident, every covariate the rest of the script uses
# ================================================================================================
imp = pd.read_csv(DERIVED / "incident_impact.csv",
                  parse_dates=["date", "onset_ts", "recovery_ts"])
imp["line_id"] = imp.line_id.astype(int)
imp["rec_s"] = imp.clear_after_recovery_s
imp["rec_min"] = imp.rec_s / 60.0
imp["onset_min"] = imp.clear_after_onset_s / 60.0
imp["D"] = imp.knock_on_train_min + imp.own_delay_train_min          # accumulated delay, train-min
imp["logD"] = np.log(imp.D)
imp["log_rec"] = np.log(imp.rec_min)
imp["n_events_day"] = imp.groupby("date").incident_id.transform("size")

# timetable facts per line, taken from the traversals (scheduled running times and terminal layovers)
tv = pd.read_pickle(DERIVED / "traversals.pkl")
seg_s = tv.groupby(tv.line_id.astype(int)).travel_s.mean()                     # mean segment running time
turn_s = tv.groupby(tv.line_id.astype(int)).turnaround_s.median()               # layover at a terminal
del tv
imp["seg_s"] = imp.line_id.map(seg_s)
imp["queue_min"] = imp.n_followers_delayed * imp.headway_s / 60.0               # one headway per follower
imp["tail_min"] = imp.dist_terminal_st * imp.seg_s / 60.0                       # ride to the terminal
GROUPS = pd.factorize(imp.date)[0]
N_CLUST = imp.date.nunique()
log(f"incidents {len(imp)} on {N_CLUST} dates (date-clustered SEs on {N_CLUST} clusters); "
    f"headway {imp.headway_s.mean():.0f} s, boundary slack {imp.boundary_headway_s.mean():.1f} s, "
    f"seg running {imp.seg_s.mean():.1f} s, dist to terminal {imp.dist_terminal_st.mean():.1f} stations")

# ================================================================================================
# B1. The definition, and the distribution it produces
# ================================================================================================
section("B1 recovery time -- definition and distribution")
log("definition: clear_after_recovery_s = last attributed delayed event - recovery_ts (src/04_impact.py:595); "
    "last_ts is the max timestamp over every delayed event attributed to the incident, the broken train included")
log(q(imp.rec_s, "time back on schedule after recovery", " s"))
log(q(imp.clear_after_onset_s, "from onset", " s"))
log("04_impact.md:135 reports mean 3338.2, min 201, p25 2246, median 3052, p75 4586.5, p90 5408.4, max 8994 s")
rec_ref = {"mean": 3338.18, "min": 201, "p25": 2246, "median": 3052, "p75": 4586.5, "p90": 5408.4, "max": 8994}
got = {"mean": imp.rec_s.mean(), "min": imp.rec_s.min(), "p25": imp.rec_s.quantile(.25),
       "median": imp.rec_s.median(), "p75": imp.rec_s.quantile(.75), "p90": imp.rec_s.quantile(.9),
       "max": imp.rec_s.max()}
dev = max(abs(got[k] - rec_ref[k]) for k in rec_ref)
log(f"reproduction: max absolute deviation from the note {dev:.2f} s -> "
    f"{'exact' if dev < 0.05 else 'MISMATCH'}; range {imp.rec_s.min():.0f}-{imp.rec_s.max():.0f} s = "
    f"{imp.rec_s.min() / 60:.1f}-{imp.rec_s.max() / 60:.1f} min")
record("B1 distribution", "clear_after_recovery_s", float(imp.rec_s.mean()), np.nan, np.nan, None, len(imp),
       None, "", "s; median 3052, p25 2246, p75 4586.5")


# ================================================================================================
# B2. The brief's hypothesis: recovery ~ accumulated delay / per-trip absorption margin
# ================================================================================================
section("B2 hypothesis -- recovery as accumulated delay divided by the per-trip margin")
# Per-line margins straight from the timetable. "margin" = the delay one trip can absorb before the next
# departure is affected, which in this timetable is the block-boundary separation delta (the running-time
# slack is 0 everywhere and the terminal layover is not on the delay path of a follower).
mg = (imp.groupby("line_id")
      .agg(n=("incident_id", "size"), h_s=("headway_s", "mean"), delta_s=("boundary_headway_s", "mean"),
           rec_med_min=("rec_min", "median"), D_med=("D", "median"), seg_s=("seg_s", "first")))
mg["turn_s"] = mg.index.map(turn_s)
mg["h_over_delta"] = mg.h_s / mg.delta_s
mg["margin_train_min"] = mg.delta_s / 60.0
for l in mg.index:
    d = imp[imp.line_id == l]
    fit = sm.OLS(d.rec_min, sm.add_constant(d.D)).fit(cov_type="cluster",
                                                      cov_kwds={"groups": pd.factorize(d.date)[0]})
    mg.loc[l, "slope_D"] = fit.params.iloc[1]
    mg.loc[l, "slope_D_se"] = fit.bse.iloc[1]
    mg.loc[l, "slope_D_r2"] = fit.rsquared
mg["slope_over_pred"] = mg.slope_D / mg.h_over_delta
mg.round(3).to_csv(OUT / "margin_by_line.csv")
log("line  n   h(s)  delta(s)  h/delta  turn(s)  seg(s)  D_med  slope_D(se)   R2   slope/pred")
for l, r in mg.iterrows():
    log(f"{l:4d} {r.n:3.0f} {r.h_s:6.0f} {r.delta_s:8.1f} {r.h_over_delta:8.3f} {r.turn_s:8.0f} "
        f"{r.seg_s:7.1f} {r.D_med:6.1f}  {r.slope_D:5.3f} ({r.slope_D_se:.3f}) {r.slope_D_r2:5.3f} "
        f"{r.slope_over_pred:8.2f}")
pred = float((mg.h_over_delta * mg.n).sum() / mg.n.sum())
log(f"h/delta is nearly the same on every line ({mg.h_over_delta.min():.3f}-{mg.h_over_delta.max():.3f}); "
    f"exposure-weighted prediction for the slope of rec(min) on D(train-min): {pred:.3f}")

# pooled fits
one = np.ones(len(imp))

m_lin = cluster_ols(imp.rec_min.values, np.column_stack([one, imp.D.values]), ["const", "D"], GROUPS,
                    "B2 linear: rec_min ~ D")
log(f"linear: slope {m_lin.params[1]:.3f} [{m_lin.conf_int()[1, 0]:.3f}, {m_lin.conf_int()[1, 1]:.3f}], "
    f"R2 {m_lin.rsquared:.3f}", end="; ")
m_lin_l = cluster_ols(imp.rec_min.values, np.column_stack([one, imp.D.values,
                                                           pd.get_dummies(imp.line_id, drop_first=True).values]),
                      ["const", "D"] + [f"line_{l}" for l in LINES if l != 2], GROUPS,
                      "B2 linear + line: rec_min ~ D + C(line)")
log(f"+ line: slope {m_lin_l.params[1]:.3f} [{m_lin_l.conf_int()[1, 0]:.3f}, "
    f"{m_lin_l.conf_int()[1, 1]:.3f}], R2 {m_lin_l.rsquared:.3f}")
m_sq = cluster_ols(imp.rec_min.values, np.column_stack([one, np.sqrt(imp.D.values)]), ["const", "sqrtD"], GROUPS,
                   "B2 square root: rec_min ~ sqrt(D)")
log(f"rec_min ~ sqrt(D): R2 {m_sq.rsquared:.3f} (queue drain predicts a square-root law)", end="; ")
m_ll = cluster_ols(imp.log_rec.values, np.column_stack([one, imp.logD.values]), ["const", "logD"], GROUPS,
                   "B2 log-log: log(rec_min) ~ log(D)", fit_name="R2 (on log scale)")
log(f"log-log elasticity {m_ll.params[1]:.3f} [{m_ll.conf_int()[1, 0]:.3f}, {m_ll.conf_int()[1, 1]:.3f}], "
    f"R2 {m_ll.rsquared:.3f} (1 = the brief's linear law, 0.5 = queue drain)")
m_ll_l = cluster_ols(imp.log_rec.values, np.column_stack([one, imp.logD.values,
                                                          pd.get_dummies(imp.line_id, drop_first=True).values]),
                     ["const", "logD"] + [f"line_{l}" for l in LINES if l != 2], GROUPS,
                     "B2 log-log + line", fit_name="R2 (on log scale)")
log(f"log-log + line: elasticity {m_ll_l.params[1]:.3f} [{m_ll_l.conf_int()[1, 0]:.3f}, "
    f"{m_ll_l.conf_int()[1, 1]:.3f}], R2 {m_ll_l.rsquared:.3f}")
m_gam = cluster_glm(imp.rec_min.values, np.column_stack([one, imp.D.values]), ["const", "D"], GROUPS,
                    "B2 Gamma GLM (log link): rec_min ~ D", sm.families.Gamma(sm.families.links.Log()))
log(f"Gamma GLM (log link): {m_gam.params[1]:.5f} [{m_gam.conf_int()[1, 0]:.5f}, "
    f"{m_gam.conf_int()[1, 1]:.5f}] per train-min = {100 * m_gam.params[1]:.2f}%, deviance explained "
    f"{1 - m_gam.deviance / m_gam.null_deviance:.3f}")

# Is the pooled slope statistically distinguishable from the predicted h/delta?
b, se = float(m_lin.params[1]), float(m_lin.bse[1])
t_pred = (b - pred) / se
p_pred = 2 * (1 - stats.norm.cdf(abs(t_pred)))
log(f"test slope = h/delta: t = {t_pred:+.2f}, p = {p_pred:.3f} -> the pooled slope is "
    f"{'compatible with' if p_pred > 0.05 else 'incompatible with'} the margin")
record("B2 test: slope = h/delta", "D", b, float(m_lin.conf_int()[1, 0]), float(m_lin.conf_int()[1, 1]),
       p_pred, len(imp), None, "", f"predicted {pred:.3f}")

# The mechanism the hypothesis is standing in for: a follower queue that drains one headway per train, plus
# the ride of the last delayed train to the terminal, where nothing follows it.
m_str = cluster_ols(imp.rec_min.values, np.column_stack([one, imp.queue_min.values, imp.tail_min.values]),
                    ["const", "queue_min", "tail_min"], GROUPS, "B2 structural: rec_min ~ queue + tail")
log(f"structural (queue drain + ride to the terminal): R2 {m_str.rsquared:.3f}, AIC {m_str.aic:.0f} "
    f"(vs {m_lin.aic:.0f} for D); queue {m_str.params[1]:.3f}, tail {m_str.params[2]:.3f}")
m_str_l = cluster_ols(imp.rec_min.values, np.column_stack([one, imp.queue_min.values, imp.tail_min.values,
                                                           pd.get_dummies(imp.line_id, drop_first=True).values]),
                      ["const", "queue_min", "tail_min"] + [f"line_{l}" for l in LINES if l != 2], GROUPS,
                      "B2 structural + line")
log(f"  + line: R2 {m_str_l.rsquared:.3f}, AIC {m_str_l.aic:.0f} (dAIC {m_str_l.aic - m_str.aic:+.0f} for 3 "
    f"terms) -> line identity has little left once queue and tail are in")
m_turn = cluster_ols(imp.rec_min.values, np.column_stack([one, imp.queue_min.values, imp.tail_min.values,
                                                           imp.line_id.map(turn_s).values / 60.0]),
                     ["const", "queue_min", "tail_min", "turn_min"], GROUPS, "B2 structural + terminal layover")
log(f"  + terminal layover: R2 {m_turn.rsquared:.3f}, coef {m_turn.params[3]:.3f} "
    f"[{m_turn.conf_int()[3, 0]:.3f}, {m_turn.conf_int()[3, 1]:.3f}] p {m_turn.pvalues[3]:.3f} -> the layover "
    f"is not on a follower's delay path and does not help")
m_mix = cluster_ols(imp.rec_min.values, np.column_stack([one, imp.D.values, imp.tail_min.values]),
                    ["const", "D", "tail_min"], GROUPS, "B2 mixed: rec_min ~ D + tail")
log(f"rec_min ~ D + tail: R2 {m_mix.rsquared:.3f}, D {m_mix.params[1]:.3f} "
    f"[{m_mix.conf_int()[1, 0]:.3f}, {m_mix.conf_int()[1, 1]:.3f}], tail {m_mix.params[2]:.3f}")
log(f"corr(rec, D) = {imp.rec_min.corr(imp.D):.3f}, corr(rec, tail) = {imp.rec_min.corr(imp.tail_min):.3f}, "
    f"corr(D, tail) = {imp.D.corr(imp.tail_min):.3f} -> D and the tail are nearly independent, and the tail "
    f"carries the recovery time")


# ================================================================================================
# B3. Everything else that could set the recovery time
# ================================================================================================
section("B3 other determinants -- univariate screen, then jointly")
imp["log_dur"] = np.log(imp.duration_s)
imp["burst"] = imp.in_burst.astype(int)
imp["at_station"] = (imp.location_type == "station").astype(int)
imp["peak_i"] = imp.peak.astype(int)
log(f"candidates: log(duration_s) [{imp.duration_s.min():.0f}-{imp.duration_s.max():.0f} s], in_burst "
    f"({imp.burst.sum()}/{len(imp)}), location_type ({imp.at_station.sum()} at a station), line_id, peak "
    f"({imp.peak_i.sum()} in peak), n_events_day ({imp.n_events_day.min()}-{imp.n_events_day.max()}); weekday is "
    f"constant and is dropped")

UNI: list[tuple[str, float]] = []      # (covariate, p -- joint Wald for factors)
SHOW: list[tuple[str, str]] = []       # (covariate, effect string) in the same order


def univariate(label: str, cols: list[np.ndarray], terms: list[str], joint: bool = False) -> None:
    """One covariate on log(recovery); records every term, keeps one p per covariate for the BH correction."""
    x = np.column_stack([one] + cols)
    res = sm.OLS(imp.log_rec.values, x).fit(cov_type="cluster", cov_kwds={"groups": GROUPS})
    ci = res.conf_int()
    for i, nm in enumerate(terms, start=1):
        record("B3 univariate", f"{label}: {nm}", float(np.exp(res.params[i])), float(np.exp(ci[i, 0])),
               float(np.exp(ci[i, 1])), float(res.pvalues[i]), len(imp), float(res.rsquared),
               "R2 (on log scale)", "ratio")
    if joint:
        p_joint = float(res.f_test(np.eye(1 + len(cols))[1:]).pvalue)
        UNI.append((label, p_joint))
        SHOW.append((label, f"joint p={p_joint:.3f}, R2 {res.rsquared:.3f}"))
    else:
        UNI.append((label, float(res.pvalues[1])))
        SHOW.append((label, f"x{np.exp(res.params[1]):.3f} [{np.exp(ci[1, 0]):.3f}, {np.exp(ci[1, 1]):.3f}] "
                            f"p={res.pvalues[1]:.3f}"))


univariate("log(duration_s)", [imp.log_dur.values], ["log_dur"])
univariate("in_burst", [imp.burst.values], ["burst"])
univariate("location_type=station", [imp.at_station.values], ["at_station"])
univariate("peak", [imp.peak_i.values], ["peak"])
univariate("n_events_day", [imp.n_events_day.values], ["n_events_day"])
univariate("dist_terminal_st", [imp.dist_terminal_st.values], ["dist_terminal_st"])
univariate("headway_s", [imp.headway_s.values], ["headway_s"])
uni_line = sm.OLS(imp.log_rec.values, np.column_stack([one, pd.get_dummies(imp.line_id, drop_first=True).values])
                  ).fit(cov_type="cluster", cov_kwds={"groups": GROUPS})
for i, l in enumerate([l for l in LINES if l != 2], start=1):
    record("B3 univariate", f"line_id: {l} vs 2", float(np.exp(uni_line.params[i])),
           float(np.exp(uni_line.conf_int()[i, 0])), float(np.exp(uni_line.conf_int()[i, 1])),
           float(uni_line.pvalues[i]), len(imp), float(uni_line.rsquared), "R2 (on log scale)", "ratio")
p_line = float(uni_line.f_test(np.eye(4)[1:]).pvalue)
UNI.append(("line_id", p_line))
SHOW.append(("line_id", f"joint p={p_line:.3f}, R2 {uni_line.rsquared:.3f}"))

# Benjamini-Hochberg across the eight candidates
names, ps = zip(*UNI)
order = np.argsort(ps)
bh = np.empty(len(ps))
for rank, idx in enumerate(order, start=1):
    bh[idx] = min(1.0, ps[idx] * len(ps) / rank)
bh = np.minimum.accumulate(bh[order][::-1])[::-1]
qvals = np.empty(len(ps))
qvals[order] = bh
log("effect on log(recovery): factor exp(coef), date-clustered CI, BH q over the 8 candidates")
for (lab, eff), qq in zip(SHOW, qvals):
    log(f"  {lab:21s} {eff:38s} q={qq:.3f}")
for n, pv, qv in zip(names, ps, qvals):
    record("B3 univariate (BH)", n, np.nan, np.nan, np.nan, float(qv), len(imp), None, "", f"p = {pv:.4f}")

# Joint model on the minute scale (the scale B2 works on), so the mechanism coefficients stay interpretable
line_d = pd.get_dummies(imp.line_id, drop_first=True).values
line_names = [f"line_{l}" for l in LINES if l != 2]
ctx = np.column_stack([imp.log_dur.values, imp.burst.values, imp.at_station.values, imp.peak_i.values,
                       imp.n_events_day.values, line_d])
ctx_names = ["log_dur", "in_burst", "at_station", "peak", "n_events_day"] + line_names
m_ctx = cluster_ols(imp.rec_min.values, np.column_stack([one, ctx]), ["const"] + ctx_names, GROUPS,
                    "B3 context only (rec_min)")
m_all = cluster_ols(imp.rec_min.values, np.column_stack([one, imp.queue_min.values, imp.tail_min.values, ctx]),
                    ["const", "queue_min", "tail_min"] + ctx_names, GROUPS, "B3 context + mechanism (rec_min)")
log(f"joint model (minutes): context only R2 {m_ctx.rsquared:.3f} AIC {m_ctx.aic:.0f}; context + queue + tail "
    f"R2 {m_all.rsquared:.3f} AIC {m_all.aic:.0f}")
ci_all = m_all.conf_int()
prm, pv_all = m_all.params, m_all.pvalues
log(f"survivors: queue {prm[1]:+.3f} [{ci_all[1, 0]:+.3f}, {ci_all[1, 1]:+.3f}] p={pv_all[1]:.3f}; "
    f"tail {prm[2]:+.3f} [{ci_all[2, 0]:+.3f}, {ci_all[2, 1]:+.3f}] p={pv_all[2]:.3f}")
log("line dummies vs line 2: " + ", ".join(
    f"{nm[5:]} {prm[8 + j]:+.2f} [{ci_all[8 + j, 0]:+.2f}, {ci_all[8 + j, 1]:+.2f}] p={pv_all[8 + j]:.3f}"
    for j, nm in enumerate(line_names)))
log("null once queue and tail are in: " + ", ".join(
    f"{nm} p={pv_all[3 + j]:.3f}" for j, nm in enumerate(ctx_names[:5])))

# Where the queue+tail model still misses, by line, and whether the ride should carry dwell
resid = pd.Series(imp.rec_min.values - m_str.fittedvalues, index=imp.index)
by_line = resid.groupby(imp.line_id).mean()
m_dw = cluster_ols(imp.rec_min.values, np.column_stack([one, imp.queue_min.values,
                                                       imp.dist_terminal_st.values * (imp.seg_s.values + 30) / 60]),
                   ["const", "queue_min", "tail_incl_dwell"], GROUPS, "B3 ride with 30 s dwell per station")
log("residual by line: " + ", ".join(f"{l} {by_line[l]:+.2f}" for l in LINES) +
    f"; ride + 30 s dwell per station R2 {m_dw.rsquared:.3f} AIC {m_dw.aic:.0f} (worse), so the tail is running "
    f"time")

# ================================================================================================
# B4. How much of the recovery time is predictable, and does it survive a held-out date?
# ================================================================================================
section("B4 predictability")


def loo_date(y: np.ndarray, x: np.ndarray, label: str) -> float:
    """Leave-one-date-out R2: fit on 37 dates, predict the 38th, pooled over all incidents."""
    pred = np.empty(len(y))
    for g in np.unique(GROUPS):
        held = GROUPS == g
        beta = np.linalg.lstsq(x[~held], y[~held], rcond=None)[0]
        pred[held] = x[held] @ beta
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    log(f"  {label:34s} in-sample R2 {sm.OLS(y, x).fit().rsquared:.3f}   leave-one-date-out R2 {r2:.3f}")
    record("B4", label, float(r2), np.nan, np.nan, None, len(y), float(r2), "leave-one-date-out R2")
    return r2


x_ctx = np.column_stack([one, ctx])
x_tail = np.column_stack([one, imp.tail_min.values])
x_mech = np.column_stack([one, imp.queue_min.values, imp.tail_min.values])
x_full = np.column_stack([one, imp.queue_min.values, imp.tail_min.values, ctx])
r2_d = loo_date(imp.rec_min.values, np.column_stack([one, imp.D.values]), "accumulated delay D only")
r2_dl = loo_date(imp.rec_min.values, np.column_stack([one, imp.D.values, line_d]), "D + line")
r2_ctx = loo_date(imp.rec_min.values, x_ctx, "context only (duration, burst, ...)")
r2_q = loo_date(imp.rec_min.values, np.column_stack([one, imp.queue_min.values]), "queue drain only")
r2_tail = loo_date(imp.rec_min.values, x_tail, "ride to the terminal only")
r2_mech = loo_date(imp.rec_min.values, x_mech, "queue + tail")
r2_full = loo_date(imp.rec_min.values, x_full, "queue + tail + context")
log("")
log(f"best model: rec_min = {m_str.params[0]:.2f} + {m_str.params[1]:.3f} * followers*headway/60 + "
    f"{m_str.params[2]:.3f} * dist_to_terminal*seg_time/60")
log(f"  in-sample R2 {m_str.rsquared:.3f}, leave-one-date-out R2 {r2_mech:.3f}; the brief's delay-only form "
    f"reaches R2 {m_lin.rsquared:.3f} / {r2_d:.3f}")
log(f"  residual sd {np.sqrt(m_str.mse_resid):.1f} min on a mean recovery of {imp.rec_min.mean():.1f} min; the two "
    f"terms are nearly independent (corr {imp.queue_min.corr(imp.tail_min):.2f})")

# ================================================================================================
# B5. Land the tables
# ================================================================================================
cols = ["incident_id", "date", "line_id", "route_id", "train_id", "direction", "location_type", "from_station",
        "to_station", "dist_terminal_st", "peak", "in_burst", "duration_s", "n_followers_delayed", "headway_s",
        "boundary_headway_s", "seg_s", "queue_min", "tail_min", "knock_on_train_min", "own_delay_train_min", "D",
        "n_events_day", "rec_s", "rec_min", "onset_min"]
tbl = imp[cols].copy()
tbl["turn_s"] = tbl.line_id.map(turn_s)
tbl["h_over_delta"] = tbl.headway_s / tbl.boundary_headway_s
tbl.to_csv(OUT / "recovery_table.csv", index=False)
models = pd.DataFrame(COEFS)
models.to_csv(OUT / "recovery_models.csv", index=False)
log(f"\nwrote {OUT / 'recovery_table.csv'} ({len(tbl)} rows x {tbl.shape[1]} cols), "
    f"{OUT / 'recovery_models.csv'} ({len(models)} rows), {OUT / 'margin_by_line.csv'} ({len(mg)} rows)")

# ================================================================================================
# Verdict
# ================================================================================================
section("Summary")
log("A3 (victim): the victim is an exposure draw, not a roster -- given a train's time in the carrier's zone and")
log("  the day's incident count, train identity carries no information (zone score test z=-0.16, p=0.87); the")
log("  'repeat offenders' are the line-10 corridor and the burst days, and delay received does not concentrate.")
log(f"B2 (recovery = accumulated delay / per-trip margin): magnitude right, mechanism wrong. Pooled slope "
    f"{b:.2f} [{float(m_lin.conf_int()[1, 0]):.2f}, {float(m_lin.conf_int()[1, 1]):.2f}]")
log(f"  vs the timetable prediction h/delta = {pred:.2f} (p={p_pred:.2f}), but R2 is only {m_lin.rsquared:.2f},")
log(f"  the per-line slopes are 0.98/1.21/2.04/-0.00, so the agreement is between-line, and the elasticity is")
log(f"  {m_ll.params[1]:.2f}, between the linear law (1) and queue drain (0.5). The running-time channel of the")
log("  brief's margin formula is empty (excess_travel_s >= 0 on all 4.19M traversals, slack 0 on all 310")
log("  segments), so only the block-boundary slack and the terminal layover could absorb, and the layover does")
log("  not enter. What actually sets the recovery time is queue drain plus the ride to the terminal:")
log(f"  rec_min = {m_str.params[0]:.2f} + {m_str.params[1]:.3f}*followers*headway/60 + "
    f"{m_str.params[2]:.3f}*dist_to_terminal*seg/60, both coefficients ~ 1 as the mechanism predicts.")
log(f"B3 (other determinants): nothing else survives. Duration is unrelated to recovery (x1.19, p=0.46), and so")
log("  are burst membership, peak and the day's incident count (p 0.40-0.84); location class is marginal")
log("  (x1.21, p=0.056) but is absorbed by the tail term and does not survive BH (q=0.15);")
log("  once queue and tail are in, every context term is null (p 0.13-0.88) and line dummies are worth ~3 min.")
log(f"B4 (predictability): R2 {m_str.rsquared:.3f} in sample and {r2_mech:.3f} leave-one-date-out, residual sd")
log(f"  {np.sqrt(m_str.mse_resid):.1f} min on a mean recovery of {imp.rec_min.mean():.1f} min; held out, queue "
    f"alone {r2_q:.3f} and tail alone {r2_tail:.3f}")
log(f"  the brief's delay-only form reaches {m_lin.rsquared:.3f} / {r2_d:.3f}, the context variables alone "
    f"{m_ctx.rsquared:.3f}.")
log(f"B1: the definition reproduces notes/04_impact.md:135 exactly (max deviation {dev:.2f} s), 143 incidents,")
log("  mean 3338.2 s / median 3052 s / p25 2246 / p75 4586.5 / max 8994 s (3.4-149.9 min).")
log(f"Caveats: 38 date clusters among 105 service days, so every interval is too narrow; the data are")
log("  semi-synthetic, so these are accounting identities of the simulation, not causal effects.")
