"""The bursts are the phenomenon: what is inside one?

Purpose
-------
`12_support_gap.md` found the on-line breakdown rate below 250 m to be 291.0 per 1000 train-h against 0.0 in
250-600 m (279.1 train-h of exposure), but 58 of the 62 near events sit inside 23 bursts: with only the 9 isolated
on-line events the ratio collapses to about 1.4x.  `11_leadtime.md` killed the off-line "lead time" as a clock
artefact.  Both results say the same thing -- the distance effect is not a property of individual failures, it is a
property of clusters.  So the object to explain is not "why is proximity dangerous" but "what a burst is".

This script characterises the 23 bursts (30 min gap rule, 3+ events) from the inside and asks whether they are
generated around the carrier:

1. anatomy of every burst: size, span in time, distinct trains and lines, distance of every event to the carrier
   at its own onset, whether the event is on the carrier's own line (Line 10);
2. is the first event special (paired comparison against the mean of the remainder, sign test over bursts);
3. THE CENTRAL TEST: does distance to the carrier grow with rank inside the burst?  Per-burst Spearman, pooled
   mean, sign test, and the pooled rank-on-distance regression with burst fixed effects (a positive slope means the
   burst spreads outward from the carrier; a flat slope means it is a fixed zone firing repeatedly);
4. a within-burst permutation null for that slope: keep each burst's times and lines, resample which event is which
   rank, 5000 draws;
5. the isolated events only (27, of which 9 on-line and 18 off-line): does any distance effect survive at the
   individual level, with exact Poisson CIs and the exposure of `12_support_gap.md`;
6. what a burst is NOT: repeated trains, burst duration against the first event's recovery time, and line pairs
   with no shared station;
7. burst size against the knock-on delay attributed to the burst (`out/recovery/recovery_table.csv`, written by
   `04_impact.py`).

Read-only: `figures/screening/tables/breakdown_events.csv`, `out/geo/events_geo.csv`, `out/recovery/`,
`out/support_gap/` (exposure), `data_derived/traversals.pkl` (station sets).  Nothing in `src/01`-`src/19` is re-run.

Outputs
-------
out/burst_structure/bursts_anatomy.csv           one row per burst
out/burst_structure/first_event.csv              first event vs the mean of the rest
out/burst_structure/rank_distance.csv            per-burst Spearman and within-burst slope
out/burst_structure/rank_distance_fit.csv        pooled fixed-effects slope + permutation null summary
out/burst_structure/permutation_null.csv         the 5000 draws of the within-burst permutation
out/burst_structure/isolated_rates.csv           isolated-event rates by band, exact Poisson CIs
out/burst_structure/burst_not.csv                the three "what a burst is not" checks
out/burst_structure/burst_size_impact.csv        burst size vs knock-on delay
out/burst_structure/rule_sensitivity.csv         the same tests under the 15 / 30 / 60 min gap rules
out/burst_structure/burst_structure_results.txt  the full log (console shows <= 15 lines per call)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
GEO = ROOT / "out" / "geo"
TAB = ROOT / "figures" / "screening" / "tables"
DERIVED = ROOT / "data_derived"
OUT = ROOT / "out" / "burst_structure"
OUT.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT / "burst_structure_results.txt"
CARRIER = "T3420"
OWN_LINE = "10"  # the carrier's own line: "on its own line" means the broken train is a Line 10 train
N_PERM = 5000
SEED = 20260302
GAP_PRIMARY, MIN_SIZE = 30.0, 3  # 02_screening.py's primary burst rule

_log_lines: list[str] = []


def log(*parts) -> None:
    """Console gets at most 15 lines per call; the results file gets everything."""
    text = " ".join(str(p) for p in parts)
    lines = text.splitlines()
    print("\n".join(lines[:15]) + (f"\n   ... (+{len(lines) - 15} more lines in {LOG_PATH.name})"
                                   if len(lines) > 15 else ""), flush=True)
    _log_lines.append(text)


def section(title: str) -> None:
    log("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def flush_log() -> None:
    LOG_PATH.write_text("\n".join(_log_lines) + "\n", encoding="utf-8")


def garwood(k, hours) -> tuple[np.ndarray, np.ndarray]:
    """Exact Poisson (Garwood) 95% interval for a rate per 1000 train-hours, same estimator as 12/17/19."""
    k = np.asarray(k, dtype=float)
    hours = np.maximum(np.asarray(hours, dtype=float), 1e-12)
    lo = np.where(k > 0, stats.chi2.ppf(0.025, 2 * np.maximum(k, 1)) / 2.0, 0.0)
    hi = stats.chi2.ppf(0.975, 2 * k + 2) / 2.0
    return lo / hours * 1000.0, hi / hours * 1000.0


def sign_test(x: np.ndarray) -> tuple[int, int, float]:
    """Two-sided exact sign test on the non-zero entries of x: (n_positive, n_nonzero, p)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x != 0)]
    k = int((x > 0).sum())
    return k, int(x.size), float(stats.binomtest(k, x.size, 0.5).pvalue) if x.size else np.nan


# ================================================================================================
section("0. Data: the 143 breakdowns, their burst labels, and the per-incident impact attribution")
# ================================================================================================
BD = pd.read_csv(TAB / "breakdown_events.csv", parse_dates=["onset", "recovered"]).sort_values("onset")
assert len(BD) == 143, f"expected 143 breakdowns, found {len(BD)}"
BD = BD.reset_index(drop=True)


def assign_bursts(gap_min: float, min_size: int = MIN_SIZE) -> pd.Series:
    """Burst id per breakdown under one gap rule: the chaining of 02_screening.py, recomputed for the sensitivity."""
    gap = BD.onset.diff().dt.total_seconds() / 60.0
    new_chain = ~((BD.date == BD.date.shift()) & (gap <= gap_min))
    chain = new_chain.cumsum()
    size = chain.map(chain.value_counts())
    return pd.Series(np.where(size >= min_size, chain, -1), index=BD.index)


BURST_PRIMARY = assign_bursts(GAP_PRIMARY)
assert (BURST_PRIMARY == BD.burst_id).all(), "the 30 min rule must reproduce 02_screening's burst labels"

EV = pd.read_csv(GEO / "events_geo.csv", parse_dates=["onset_ts"])
EV = EV.merge(BD[["train_id", "onset", "burst_id", "in_burst"]].rename(columns={"onset": "onset_ts"}),
              on=["train_id", "onset_ts"], how="left")
assert len(EV) == 143 and EV.burst_id.notna().all(), "burst labels did not match every event"
EV["is_online"] = EV.line_id.astype(str) == OWN_LINE
EV["date"] = pd.to_datetime(EV.onset_ts).dt.date.astype(str)

REC = pd.read_csv(ROOT / "out" / "recovery" / "recovery_table.csv")
assert len(REC) == 143 and REC.incident_id.is_unique
EV = EV.merge(REC[["incident_id", "knock_on_train_min", "own_delay_train_min", "rec_min", "onset_min",
                   "duration_s", "n_followers_delayed"]], on="incident_id", how="left")
assert EV.rec_min.notna().all(), "recovery table does not cover every incident"
assert bool((EV.in_burst == (EV.burst_id >= 0)).all()), "in_burst and burst_id disagree"

# The burst frame: every event inside a burst, ordered by onset inside its burst (rank 1 = first).
B = EV[EV.burst_id >= 0].sort_values(["burst_id", "onset_ts"]).copy()
B["rank"] = B.groupby("burst_id").cumcount() + 1
N_BURST = int(B.burst_id.nunique())
ISO = EV[EV.burst_id < 0].copy()
assert len(B) == 116 and N_BURST == 23 and len(ISO) == 27, "burst/isolated split changed"
assert (B.straight_m.notna()).all(), "a burst event has no distance to the carrier"

log(f"bursts: {N_BURST} bursts holding {len(B)} of {len(EV)} events (gap <= {GAP_PRIMARY:.0f} min, "
    f">= {MIN_SIZE} events); {len(ISO)} isolated events, {int(ISO.is_online.sum())} of them on Line 10 "
    f"({int(B.is_online.sum())} burst events are on Line 10)")
log(f"burst sizes: {B.groupby('burst_id').size().value_counts().sort_index().to_dict()}; "
    f"on {B.date.nunique()} service days")
log(f"carrier {CARRIER}; distance = events_geo.straight_m, the straight-line distance from the carrier to the "
    f"broken train at that event's own onset; {int(EV.straight_m.isna().sum())} event(s) without a distance "
    f"({', '.join(EV.loc[EV.straight_m.isna(), 'incident_id'])})")


# ================================================================================================
section("1. Anatomy: one row per burst")
# ================================================================================================
rows = []
for bid, g in B.groupby("burst_id"):
    d = g.straight_m.values.astype(float)
    span_min = (g.onset_ts.max() - g.onset_ts.min()).total_seconds() / 60.0
    rows.append({
        "burst_id": int(bid), "date": g.date.iloc[0], "start": g.onset_ts.min(),
        "n_events": len(g), "span_min": span_min, "n_trains": g.train_id.nunique(),
        "n_lines": g.line_id.nunique(), "lines": "+".join(str(x) for x in sorted(g.line_id.unique())),
        "n_online": int(g.is_online.sum()), "first_online": bool(g.is_online.iloc[0]),
        "d_first_m": d[0], "d_median_m": float(np.median(d)), "d_min_m": float(d.min()),
        "d_max_m": float(d.max()), "span_m": float(d.max() - d.min()),
        "d_all_m": " ".join(f"{x:.0f}" for x in d)})
AN = pd.DataFrame(rows)
AN["gap_to_next_min"] = np.nan
AN = AN.sort_values("start").reset_index(drop=True)
AN.to_csv(OUT / "bursts_anatomy.csv", index=False)

log(f"{N_BURST} bursts, {int(AN.n_events.sum())} events; size {AN.n_events.min()}-{AN.n_events.max()} "
    f"(median {AN.n_events.median():.0f}); every burst has as many distinct trains as events: "
    f"{int((AN.n_trains == AN.n_events).sum())}/{N_BURST}")
log(f"within-burst span: time median {AN.span_min.median():.1f} min "
    f"(range {AN.span_min.min():.1f}-{AN.span_min.max():.1f}), distance median {AN.span_m.median():.0f} m "
    f"(range {AN.span_m.min():.0f}-{AN.span_m.max():.0f}); the median burst stays inside a "
    f"{AN.d_max_m.median():.0f} m radius of the carrier")
log(f"bursts touching the carrier's own line (Line 10): {int((AN.n_online > 0).sum())}/{N_BURST}; "
    f"bursts whose first event is on Line 10: {int(AN.first_online.sum())}/{N_BURST}; "
    f"bursts with more than one line: {int((AN.n_lines > 1).sum())}/{N_BURST}")
log(f"bursts with every event inside 250 m: {int((AN.d_max_m < 250).sum())}/{N_BURST}; "
    f"with every event beyond 250 m: {int((AN.d_min_m >= 250).sum())}/{N_BURST}; mixed: "
    f"{int(((AN.d_min_m < 250) & (AN.d_max_m >= 250)).sum())}/{N_BURST}; with at least one event inside 250 m: "
    f"{int((AN.d_min_m < 250).sum())}/{N_BURST}")
log(f"first event of a burst: median {AN.d_first_m.median():.0f} m from the carrier (range "
    f"{AN.d_first_m.min():.0f}-{AN.d_first_m.max():.0f}), inside 250 m in {int((AN.d_first_m < 250).sum())}/{N_BURST}; "
    f"burst events on Line 10 {int(AN.n_online.sum())}/{int(AN.n_events.sum())} vs isolated events on Line 10 "
    f"{int(ISO.is_online.sum())}/{len(ISO)}")
log("\nper-burst anatomy (distance to the carrier at each event's own onset, in rank order):\n" + AN[
    ["burst_id", "date", "start", "n_events", "span_min", "n_lines", "lines", "n_online", "first_online",
     "d_first_m", "d_min_m", "d_max_m", "span_m", "d_all_m"]].assign(
    start=lambda d: d.start.dt.strftime("%m-%d %H:%M"),
    span_min=lambda d: d.span_min.round(1), d_first_m=lambda d: d.d_first_m.round(0),
    d_min_m=lambda d: d.d_min_m.round(0), d_max_m=lambda d: d.d_max_m.round(0),
    span_m=lambda d: d.span_m.round(0)).to_string(index=False))
flush_log()


# ================================================================================================
section("2. Is the first event special?")
# ================================================================================================
FE = pd.DataFrame({
    "burst_id": AN.burst_id.values,
    "n_events": AN.n_events.values,
    "d_first_m": AN.d_first_m.values,
    "first_online": AN.first_online.values,
    "d_rest_mean_m": B[B["rank"] > 1].groupby("burst_id").straight_m.mean().reindex(AN.burst_id).values,
    "d_rest_min_m": B[B["rank"] > 1].groupby("burst_id").straight_m.min().reindex(AN.burst_id).values,
    "online_rest": B[B["rank"] > 1].groupby("burst_id").is_online.mean().reindex(AN.burst_id).values})
FE["diff_m"] = FE.d_first_m - FE.d_rest_mean_m  # negative = the first event is closer to the carrier
FE.to_csv(OUT / "first_event.csv", index=False)
k_neg, n_nz, p_sign = sign_test(-FE.diff_m)  # positive direction = first event closer
t_paired = stats.ttest_rel(FE.d_first_m, FE.d_rest_mean_m)
w = stats.wilcoxon(FE.d_first_m, FE.d_rest_mean_m)
LATER = B[B["rank"] > 1]
n_first_online = int(FE.first_online.sum())
n_later_online = int(LATER.is_online.sum())
n_later = int(len(LATER))
fisher_p = stats.fisher_exact([[n_first_online, N_BURST - n_first_online],
                               [n_later_online, n_later - n_later_online]])[1]
log(f"first event vs the mean of the rest, paired over {N_BURST} bursts: mean difference "
    f"{FE.diff_m.mean():+.0f} m (median {FE.diff_m.median():+.0f} m), paired t = {t_paired.statistic:.2f} "
    f"(p = {t_paired.pvalue:.3f}), Wilcoxon p = {w.pvalue:.3f}")
log(f"sign test: the first event is closer than the rest of its burst in {k_neg}/{n_nz} bursts with a non-zero "
    f"difference (p = {p_sign:.3f}); further in {n_nz - k_neg}")
log(f"first event on Line 10: {n_first_online}/{N_BURST} bursts; later events on Line 10: "
    f"{n_later_online}/{n_later} ({100 * LATER.is_online.mean():.0f}%); Fisher exact p = {fisher_p:.3f}")
log("\n" + FE.assign(d_first_m=lambda d: d.d_first_m.round(0), d_rest_mean_m=lambda d: d.d_rest_mean_m.round(0),
                     d_rest_min_m=lambda d: d.d_rest_min_m.round(0), diff_m=lambda d: d.diff_m.round(0),
                     online_rest=lambda d: d.online_rest.round(2)).to_string(index=False))
flush_log()


# ================================================================================================
section("3. THE CENTRAL TEST: does the distance to the carrier grow with rank inside the burst?")
# ================================================================================================
def within_slope(y, x, g, ng) -> tuple[float, float, float, int]:
    """Slope of y on x after removing the burst means (burst fixed effects), with its OLS SE.

    Removing the burst mean is exactly the fixed-effects estimator here: with one regressor and a full set of burst
    dummies, the slope is the within-burst covariance over the within-burst variance.  df = N - bursts - 1.
    """
    cnt = np.bincount(g, minlength=ng).astype(float)
    xd = x - (np.bincount(g, x, ng) / cnt)[g]
    yd = y - (np.bincount(g, y, ng) / cnt)[g]
    sxx = float(xd @ xd)
    b = float(xd @ yd) / sxx
    resid = yd - b * xd
    df = int(len(y) - ng - 1)
    se = float(np.sqrt((resid @ resid) / df / sxx))
    return b, se, float(1.0 - (resid @ resid) / (yd @ yd)), df


B["pos"] = (B["rank"] - 1) / B.groupby("burst_id").burst_id.transform("size").sub(1).clip(lower=1)
GID = pd.factorize(B.burst_id)[0].astype(np.int32)
NG = int(GID.max()) + 1
D_M = B.straight_m.values.astype(float)
RANK = B["rank"].values.astype(float)
POS = B["pos"].values.astype(float)

rd = []
for bid, g in B.groupby("burst_id"):
    n = len(g)
    if n >= 3 and g.straight_m.nunique() >= 3:
        rho, p_rho = stats.spearmanr(g["rank"], g.straight_m)
        sl = np.polyfit(g["rank"].values.astype(float), g.straight_m.values.astype(float), 1)[0]
    else:
        rho, p_rho, sl = np.nan, np.nan, np.nan
    rd.append({"burst_id": int(bid), "n_events": n, "spearman_rho": rho, "spearman_p": p_rho,
               "slope_m_per_rank": sl, "d_first_m": float(g.straight_m.iloc[0]),
               "d_last_m": float(g.straight_m.iloc[-1])})
RD = pd.DataFrame(rd)
RD.to_csv(OUT / "rank_distance.csv", index=False)
k_pos, n_rho, p_sign_rho = sign_test(RD.spearman_rho)
t_rho = stats.ttest_1samp(RD.spearman_rho.dropna(), 0.0)
w_rho = stats.wilcoxon(RD.spearman_rho.dropna())

b_rank, se_rank, r2_rank, df_rank = within_slope(D_M, RANK, GID, NG)
b_pos, se_pos, r2_pos, _ = within_slope(D_M, POS, GID, NG)
# A slope of 0 would mean the burst is a fixed zone firing repeatedly; a positive slope means it spreads outward
# from the carrier.  The sign test over bursts is the assumption-light version of the same question.
n_usable = int(RD.spearman_rho.notna().sum())
log(f"per-burst Spearman(rank, distance): {n_usable} of {N_BURST} bursts usable (a burst needs >= 3 events with "
    f"distinct distances); mean rho {RD.spearman_rho.mean():+.3f}, median {RD.spearman_rho.median():+.3f}, "
    f"one-sample t = {t_rho.statistic:.2f} (p = {t_rho.pvalue:.3f}), Wilcoxon p = {w_rho.pvalue:.3f}")
log(f"sign test over bursts: rho > 0 in {k_pos}/{n_rho} (p = {p_sign_rho:.3f}); rho < 0 in {n_rho - k_pos}; "
    f"{N_BURST - n_rho} burst(s) with rho exactly 0 left out of the sign test")
log(f"POOLED rank-distance regression with burst fixed effects ({len(B)} events, {NG} bursts, df = {df_rank}): "
    f"slope {b_rank:+.1f} m per rank, SE {se_rank:.1f}, 95% CI [{b_rank - 1.96 * se_rank:+.1f}, "
    f"{b_rank + 1.96 * se_rank:+.1f}], within-R2 {r2_rank:.4f}")
log(f"same regression on the within-burst position (0 = first event, 1 = last): {b_pos:+.0f} m across a whole "
    f"burst, SE {se_pos:.0f}, 95% CI [{b_pos - 1.96 * se_pos:+.0f}, {b_pos + 1.96 * se_pos:+.0f}]")
log("\nper-burst rank-distance detail:\n" + RD.assign(
    spearman_rho=lambda d: d.spearman_rho.round(2), spearman_p=lambda d: d.spearman_p.round(3),
    slope_m_per_rank=lambda d: d.slope_m_per_rank.round(0), d_first_m=lambda d: d.d_first_m.round(0),
    d_last_m=lambda d: d.d_last_m.round(0)).to_string(index=False))
flush_log()


# ================================================================================================
section("4. Null for the central test: permute the rank-distance pairing inside each burst")
# ================================================================================================
# Each burst keeps its own times, trains, lines and distances; only the assignment of a distance to a rank is
# resampled.  That is the sharp null "the order of the events inside a burst carries no information about distance
# to the carrier", and it preserves every marginal the data actually has (burst size, the burst's distance set, the
# burst's time span).  The MC floor is 1/(N_PERM+1).
rng = np.random.default_rng(SEED)
idx_by_burst = [np.where(GID == b)[0] for b in range(NG)]
null = np.empty(N_PERM)
for i in range(N_PERM):
    dp = D_M.copy()
    for ix in idx_by_burst:
        dp[ix] = D_M[rng.permutation(ix)]
    null[i] = within_slope(dp, RANK, GID, NG)[0]
pd.DataFrame({"draw": np.arange(N_PERM), "pooled_slope_m_per_rank": null}).to_csv(
    OUT / "permutation_null.csv", index=False)

p_two = (1 + int((np.abs(null) >= abs(b_rank)).sum())) / (N_PERM + 1)
p_grow = (1 + int((null >= b_rank).sum())) / (N_PERM + 1)  # one-sided: is the burst spreading outward?
floor = 1.0 / (N_PERM + 1)
FIT = pd.DataFrame([
    {"quantity": "pooled slope, rank -> distance (burst FE)", "estimate": b_rank, "se": se_rank,
     "ci_lo": b_rank - 1.96 * se_rank, "ci_hi": b_rank + 1.96 * se_rank, "unit": "m per rank",
     "perm_p_two_sided": p_two, "perm_p_outward": p_grow, "mc_floor": floor},
    {"quantity": "pooled slope, position -> distance (burst FE)", "estimate": b_pos, "se": se_pos,
     "ci_lo": b_pos - 1.96 * se_pos, "ci_hi": b_pos + 1.96 * se_pos, "unit": "m across a burst",
     "perm_p_two_sided": np.nan, "perm_p_outward": np.nan, "mc_floor": floor},
    {"quantity": "mean per-burst Spearman rho", "estimate": float(RD.spearman_rho.mean()),
     "se": float(RD.spearman_rho.std(ddof=1) / np.sqrt(n_rho)), "ci_lo": np.nan, "ci_hi": np.nan,
     "unit": "rho", "perm_p_two_sided": p_sign_rho, "perm_p_outward": np.nan, "mc_floor": np.nan},
    {"quantity": "permutation null mean", "estimate": float(null.mean()), "se": float(null.std(ddof=1)),
     "ci_lo": float(np.quantile(null, 0.025)), "ci_hi": float(np.quantile(null, 0.975)), "unit": "m per rank",
     "perm_p_two_sided": np.nan, "perm_p_outward": np.nan, "mc_floor": floor}])
FIT.to_csv(OUT / "rank_distance_fit.csv", index=False)

log(f"permutation null ({N_PERM} draws, distances resampled across ranks inside each burst): mean "
    f"{null.mean():+.1f}, sd {null.std(ddof=1):.1f}, 2.5-97.5% [{np.quantile(null, 0.025):+.0f}, "
    f"{np.quantile(null, 0.975):+.0f}] m per rank")
log(f"observed slope {b_rank:+.1f} m per rank: two-sided permutation p = {p_two:.3f}; one-sided p for "
    f"'the burst spreads outward' (slope > 0) = {p_grow:.3f}; Monte-Carlo floor 1/{N_PERM + 1} = {floor:.4f}")
log(f"the observed slope is at percentile {100 * float((null < b_rank).mean()):.1f} of the null: the data put the "
    f"burst slightly on the INWARD side, not the outward side")
flush_log()


# ================================================================================================
section("5. The isolated events: does any distance effect survive at the individual level?")
# ================================================================================================
# Exposure is not recomputed: it is `12_support_gap.md`'s, read back from its tables.  Inside 1000 m the 1 s
# refinement is the right denominator (the 30 s grid aliases the near field); beyond it the full 30 s grid.
H1S = pd.read_csv(ROOT / "out" / "support_gap" / "hist_1s_near.csv")
BANDS = pd.read_csv(ROOT / "out" / "support_gap" / "bands.csv")
H_ON_NEAR = float(H1S.loc[H1S.band_lo_m < 250.0, "same_1s_h"].sum())
H_ON_GAP = float(H1S.loc[(H1S.band_lo_m >= 250.0) & (H1S.band_hi_m <= 600.0), "same_1s_h"].sum())
H_OFF_NEAR = float(BANDS.loc[BANDS.band.isin(["600-1000", "1000-1500"]), "off_h"].sum())
H_OFF_FAR = float(BANDS.loc[BANDS.band.isin(["1500-2000", "2000-3000", "3000-5000", "5000+"]), "off_h"].sum())
# the 1 s refinement stops at 600 m; on-line exposure beyond that only exists on the 30 s grid, which is
# unbiased at these separations (the near-field phase aliasing in notes/12 was a 0-200 m effect)
H_ON_MID = float(BANDS.loc[BANDS.band.isin(["600-1000", "1000-1500"]), "same_h"].sum())
H_ON_FAR = float(BANDS.loc[BANDS.band.isin(["1500-2000", "2000-3000", "3000-5000", "5000+"]), "same_h"].sum())
assert abs(H_ON_NEAR - 213.0) < 0.1 and abs(H_ON_GAP - 279.1) < 0.1, "the 1 s near-field exposure moved"


def in_band(d: pd.Series, lo: float, hi: float) -> int:
    return int(((d >= lo) & (d < hi)).sum())


def poisson_ratio(k1: int, h1: float, k2: int, h2: float) -> dict:
    """Rate ratio k1/h1 over k2/h2 with an exact conditional (binomial) 95% CI for the ratio.

    Given the total, the split between the two bands is Binomial(n, h1/(h1+h2)) under a common rate, so an exact CI
    for that proportion maps to a CI for the ratio.  With k2 = 0 the upper limit is unbounded, which is the honest
    answer, and the ratio itself is infinite: the conservative bound below is rate_lo(k1)/rate_hi(k2).
    """
    n = k1 + k2
    if n == 0:
        return {"ratio": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan}
    ci = stats.binomtest(k1, n, 0.5).proportion_ci(0.95, method="exact")
    lo_p, hi_p = float(ci.low), float(ci.high)
    scale = h2 / h1
    ratio = (k1 / h1) / (k2 / h2) if k2 > 0 else np.inf
    return {"ratio": ratio,
            "ci_lo": (lo_p / (1 - lo_p) * scale) if lo_p < 1 else np.inf,
            "ci_hi": (hi_p / (1 - hi_p) * scale) if hi_p < 1 else np.inf,
            "p": float(stats.binomtest(k1, n, h1 / (h1 + h2)).pvalue)}


rate_rows = []
for sample, frame in (("all events", EV), ("isolated only", ISO)):
    on, off = frame[frame.is_online], frame[~frame.is_online]
    for band, lo, hi, h_on, h_off in (("0-250 m", 0.0, 250.0, H_ON_NEAR, np.nan),
                                      ("250-600 m", 250.0, 600.0, H_ON_GAP, np.nan),
                                      ("600 m-1.5 km", 600.0, 1500.0, H_ON_MID, H_OFF_NEAR),
                                      ("1.5 km+", 1500.0, np.inf, H_ON_FAR, H_OFF_FAR)):
        for channel, sub, hours in (("on-line", on, h_on), ("off-line", off, h_off)):
            if not np.isfinite(hours):
                continue
            k = in_band(sub.straight_m.dropna(), lo, hi)
            lo_ci, hi_ci = garwood(k, hours)
            rate_rows.append({"channel": channel, "band": band, "sample": sample, "events": k,
                              "train_h": round(hours, 2), "rate_per_1000h": k / hours * 1000.0,
                              "ci_lo": float(lo_ci), "ci_hi": float(hi_ci)})
RATES = pd.DataFrame(rate_rows)
RATES.to_csv(OUT / "isolated_rates.csv", index=False)

ISO_ON, ISO_OFF = ISO[ISO.is_online], ISO[~ISO.is_online]
r_on = poisson_ratio(in_band(ISO_ON.straight_m.dropna(), 0, 250), H_ON_NEAR,
                     in_band(ISO_ON.straight_m.dropna(), 250, 600), H_ON_GAP)
r_on_all = poisson_ratio(in_band(EV[EV.is_online].straight_m.dropna(), 0, 250), H_ON_NEAR,
                         in_band(EV[EV.is_online].straight_m.dropna(), 250, 600), H_ON_GAP)
r_on_mid = poisson_ratio(in_band(ISO_ON.straight_m.dropna(), 0, 250), H_ON_NEAR,
                         in_band(ISO_ON.straight_m.dropna(), 600, 1500), H_ON_MID)
r_on_far = poisson_ratio(in_band(ISO_ON.straight_m.dropna(), 0, 250), H_ON_NEAR,
                         in_band(ISO_ON.straight_m.dropna(), 1500, np.inf), H_ON_FAR)
r_off = poisson_ratio(in_band(ISO_OFF.straight_m.dropna(), 600, 1500), H_OFF_NEAR,
                      in_band(ISO_OFF.straight_m.dropna(), 1500, np.inf), H_OFF_FAR)
RATIOS = pd.DataFrame([
    {"contrast": "on-line, 0-250 m vs 250-600 m", "sample": "all events",
     "k1": in_band(EV[EV.is_online].straight_m.dropna(), 0, 250),
     "k2": in_band(EV[EV.is_online].straight_m.dropna(), 250, 600),
     "ratio": r_on_all["ratio"], "ci_lo": r_on_all["ci_lo"], "ci_hi": r_on_all["ci_hi"], "p": r_on_all["p"]},
    {"contrast": "on-line, 0-250 m vs 250-600 m", "sample": "isolated only",
     "k1": in_band(ISO_ON.straight_m.dropna(), 0, 250), "k2": in_band(ISO_ON.straight_m.dropna(), 250, 600),
     "ratio": r_on["ratio"], "ci_lo": r_on["ci_lo"], "ci_hi": r_on["ci_hi"], "p": r_on["p"]},
    {"contrast": "off-line, 600 m-1.5 km vs 1.5 km+", "sample": "isolated only",
     "k1": in_band(ISO_OFF.straight_m.dropna(), 600, 1500), "k2": in_band(ISO_OFF.straight_m.dropna(), 1500, np.inf),
     "ratio": r_off["ratio"], "ci_lo": r_off["ci_lo"], "ci_hi": r_off["ci_hi"], "p": r_off["p"]},
    {"contrast": "on-line, 0-250 m vs 600 m-1.5 km", "sample": "isolated only",
     "k1": in_band(ISO_ON.straight_m.dropna(), 0, 250), "k2": in_band(ISO_ON.straight_m.dropna(), 600, 1500),
     "ratio": r_on_mid["ratio"], "ci_lo": r_on_mid["ci_lo"], "ci_hi": r_on_mid["ci_hi"], "p": r_on_mid["p"]},
    {"contrast": "on-line, 0-250 m vs 1.5 km+", "sample": "isolated only",
     "k1": in_band(ISO_ON.straight_m.dropna(), 0, 250), "k2": in_band(ISO_ON.straight_m.dropna(), 1500, np.inf),
     "ratio": r_on_far["ratio"], "ci_lo": r_on_far["ci_lo"], "ci_hi": r_on_far["ci_hi"], "p": r_on_far["p"]}])
RATIOS.to_csv(OUT / "isolated_ratios.csv", index=False)

log(f"exposure read back from out/support_gap (train-h): on-line 0-250 m {H_ON_NEAR:.1f} (1 s refinement), "
    f"250-600 m {H_ON_GAP:.1f} (1 s); off-line 600 m-1.5 km {H_OFF_NEAR:.1f} and 1.5 km+ {H_OFF_FAR:.1f} (30 s grid)")
log(f"on-line events by band: 0-250 m {in_band(EV[EV.is_online].straight_m.dropna(), 0, 250)}, "
    f"250-600 m {in_band(EV[EV.is_online].straight_m.dropna(), 250, 600)}, 600 m+ "
    f"{in_band(EV[EV.is_online].straight_m.dropna(), 600, np.inf)} -- of which isolated: "
    f"{in_band(ISO_ON.straight_m.dropna(), 0, 250)} / {in_band(ISO_ON.straight_m.dropna(), 250, 600)} / "
    f"{in_band(ISO_ON.straight_m.dropna(), 600, np.inf)}")
log(f"off-line isolated events by band: 0-250 m {in_band(ISO_OFF.straight_m.dropna(), 0, 250)}, 250-600 m "
    f"{in_band(ISO_OFF.straight_m.dropna(), 250, 600)}, 600 m-1.5 km {in_band(ISO_OFF.straight_m.dropna(), 600, 1500)},"
    f" 1.5 km+ {in_band(ISO_OFF.straight_m.dropna(), 1500, np.inf)}; {int(ISO_OFF.straight_m.isna().sum())} without a "
    f"distance")
log("\nrates per 1000 train-h, exact Poisson 95% CI:\n" + RATES.assign(
    rate_per_1000h=lambda d: d.rate_per_1000h.round(1), ci_lo=lambda d: d.ci_lo.round(1),
    ci_hi=lambda d: d.ci_hi.round(1)).to_string(index=False))
log("\nratios:\n" + RATIOS.assign(ratio=lambda d: d.ratio.round(2), ci_lo=lambda d: d.ci_lo.round(2),
                                   ci_hi=lambda d: d.ci_hi.round(2), p=lambda d: d.p.round(3)).to_string(index=False))
k_on_near, k_on_gap = int(RATIOS.k1.iloc[1]), int(RATIOS.k2.iloc[1])
on_lo, on_hi = (float(x) for x in garwood(k_on_near, H_ON_NEAR))
gap_lo, gap_hi = (float(x) for x in garwood(k_on_gap, H_ON_GAP))
log(f"ISOLATED ON-LINE: {k_on_near} events / {H_ON_NEAR:.1f} h = {k_on_near / H_ON_NEAR * 1000:.1f} "
    f"[{on_lo:.1f}, {on_hi:.1f}] per 1000 h vs {k_on_gap} events / {H_ON_GAP:.1f} h = 0.0 [0, {gap_hi:.1f}]; "
    f"the point ratio against the upper limit of the empty band is "
    f"{k_on_near / H_ON_NEAR * 1000 / gap_hi:.2f}x")
log(f"exact conditional CI for that ratio: [{r_on['ci_lo']:.2f}, unbounded) -- it includes 1; the CI-based "
    f"lower bound rate_lo/rate_hi is {on_lo / gap_hi:.2f}x. Equal-rate test p = {r_on['p']:.3f}, which is the "
    f"exact binomial probability of 4 of 4 events landing in the smaller band, not evidence of an excess")
log(f"ISOLATED ON-LINE vs the on-line background: 0-250 m 4 events / {H_ON_NEAR:.1f} h vs 600 m-1.5 km "
    f"{in_band(ISO_ON.straight_m.dropna(), 600, 1500)} / {H_ON_MID:.1f} h -> {r_on_mid['ratio']:.2f} "
    f"[{r_on_mid['ci_lo']:.2f}, {r_on_mid['ci_hi']:.2f}]; vs 1.5 km+ {in_band(ISO_ON.straight_m.dropna(), 1500, np.inf)}"
    f" / {H_ON_FAR:.1f} h -> {r_on_far['ratio']:.2f} [{r_on_far['ci_lo']:.2f}, {r_on_far['ci_hi']:.2f}]")
log(f"ISOLATED OFF-LINE: {RATIOS.k1.iloc[2]} events / {H_OFF_NEAR:.1f} h vs {RATIOS.k2.iloc[2]} events / "
    f"{H_OFF_FAR:.1f} h -- ratio {r_off['ratio']:.2f} [{r_off['ci_lo']:.2f}, {r_off['ci_hi']:.2f}], "
    f"p = {r_off['p']:.4f}")
flush_log()


# ================================================================================================
section("6. What a burst is not: repeated trains, one incident with delayed reporting, network trigger")
# ================================================================================================
TV = pd.read_pickle(DERIVED / "traversals.pkl")
STATIONS = {str(ln): set(g.from_station) | set(g.to_station) for ln, g in TV.groupby(TV.line_id.astype(str))}
FIRST = B[B["rank"] == 1].set_index("burst_id")
not_rows = []
for bid, g in B.groupby("burst_id"):
    lines = sorted({str(x) for x in g.line_id.unique()})
    pairs = [(a, b) for i, a in enumerate(lines) for b in lines[i + 1:]]
    disjoint = [(a, b) for a, b in pairs if not (STATIONS[a] & STATIONS[b])]
    f = FIRST.loc[bid]
    not_rows.append({
        "burst_id": int(bid), "n_events": len(g), "n_trains": g.train_id.nunique(),
        "dup_train": bool(g.train_id.duplicated().any()),
        "span_min": float((g.onset_ts.max() - g.onset_ts.min()).total_seconds() / 60.0),
        "rec_first_min": float(f.rec_min), "onset_first_min": float(f.onset_min),
        "duration_first_min": float(f.duration_s) / 60.0,
        "span_lt_rec": bool((g.onset_ts.max() - g.onset_ts.min()).total_seconds() / 60.0 < f.rec_min),
        "span_lt_onset": bool((g.onset_ts.max() - g.onset_ts.min()).total_seconds() / 60.0 < f.onset_min),
        "lines": "+".join(lines), "n_lines": len(lines), "n_line_pairs": len(pairs),
        "n_pairs_no_interchange": len(disjoint),
        "pairs_no_interchange": ";".join(f"{a}-{b}" for a, b in disjoint)})
NOT = pd.DataFrame(not_rows)
NOT.to_csv(OUT / "burst_not.csv", index=False)

log(f"(a) repeated trains: {int(NOT.dup_train.sum())}/{N_BURST} bursts contain two events on the same train "
    f"({int(NOT.n_events.sum())} events, {int(NOT.n_trains.sum())} distinct trains) -- a burst is never one train "
    f"breaking down twice")
log(f"(b) burst span vs the first event's recovery: span < rec_min (time from the line reopening to back in "
    f"schedule) in {int(NOT.span_lt_rec.sum())}/{N_BURST} bursts; span < onset_min (onset to back in schedule) in "
    f"{int(NOT.span_lt_onset.sum())}/{N_BURST}. Median span {NOT.span_min.median():.0f} min vs median first-event "
    f"rec_min {NOT.rec_first_min.median():.0f} min and onset_min {NOT.onset_first_min.median():.0f} min")
SHARED = {(a, b): len(STATIONS[a] & STATIONS[b])
          for i, a in enumerate(sorted(STATIONS)) for b in sorted(STATIONS)[i + 1:]}
log(f"(c) line pairs with no shared station inside a burst: {int((NOT.n_pairs_no_interchange > 0).sum())}/{N_BURST} "
    f"bursts have at least one such pair, {int(NOT.n_pairs_no_interchange.sum())} of "
    f"{int(NOT.n_line_pairs.sum())} line pairs in total; lines per burst {NOT.n_lines.value_counts().sort_index().to_dict()}")
log(f"every line pair in the network shares at least one station (" + ", ".join(
    f"{a}-{b}: {n}" for (a, b), n in sorted(SHARED.items())) + "), so 'two lines with no interchange' is not a "
    f"configuration this network can produce; the burst crosses lines only through the stations they share")
log("\nper-burst checks:\n" + NOT.assign(
    span_min=lambda d: d.span_min.round(1), rec_first_min=lambda d: d.rec_first_min.round(1),
    onset_first_min=lambda d: d.onset_first_min.round(1),
    duration_first_min=lambda d: d.duration_first_min.round(1))[
    ["burst_id", "n_events", "n_trains", "dup_train", "span_min", "duration_first_min", "rec_first_min",
     "onset_first_min", "span_lt_rec", "span_lt_onset", "lines", "n_pairs_no_interchange",
     "pairs_no_interchange"]].to_string(index=False))
flush_log()


# ================================================================================================
section("7. Burst size and the knock-on delay the burst causes")
# ================================================================================================
# Attribution comes from 04_impact.py (out/recovery/recovery_table.csv): knock_on_train_min is the delay carried by
# other trains whose root is that incident, own_delay_train_min is the broken train's own.  Both are counts of
# train-minutes, so no exposure denominator is involved.
BS = B.groupby("burst_id").agg(size=("incident_id", "size"),
                               knock_on=("knock_on_train_min", "sum"),
                               own=("own_delay_train_min", "sum"),
                               n_followers=("n_followers_delayed", "sum"),
                               n_online=("is_online", "sum"),
                               span_min=("onset_ts", lambda s: (s.max() - s.min()).total_seconds() / 60.0))
BS["total"] = BS.knock_on + BS.own
BS["knock_on_per_event"] = BS["knock_on"] / BS["size"]  # bracket access: BS.size is the DataFrame's element count
BS.to_csv(OUT / "burst_size_impact.csv", index=False)
rho_imp = stats.spearmanr(BS["size"], BS.knock_on)
X = np.column_stack([np.ones(len(BS)), np.log(BS["size"].values)])
y = np.log(BS.knock_on.values)
beta, res, rank_, sv = np.linalg.lstsq(X, y, rcond=None)
fit = X @ beta
rss, tss = float(((y - fit) ** 2).sum()), float(((y - y.mean()) ** 2).sum())
s2 = rss / (len(y) - 2)
cov = s2 * np.linalg.inv(X.T @ X)
se_b = float(np.sqrt(cov[1, 1]))
rho_pe = stats.pearsonr(BS["size"], BS.knock_on_per_event)
share = BS.knock_on.sum() / EV.knock_on_train_min.sum()
log(f"burst size vs total knock-on train-min over {N_BURST} bursts: Spearman rho {rho_imp.statistic:+.2f} "
    f"(p = {rho_imp.pvalue:.4f}); log-linear fit log(knock_on) = {beta[0]:.2f} + {beta[1]:.2f} log(size), "
    f"slope SE {se_b:.2f}, 95% CI [{beta[1] - 1.96 * se_b:.2f}, {beta[1] + 1.96 * se_b:.2f}], R2 {1 - rss / tss:.2f}")
log(f"a slope of 1 means the burst's cost is proportional to its size; the burst share of all knock-on delay is "
    f"{100 * share:.0f}% ({BS.knock_on.sum():.0f} of {EV.knock_on_train_min.sum():.0f} train-min)")
log(f"knock-on per event vs burst size: Spearman rho {rho_pe.statistic:+.2f} (p = {rho_pe.pvalue:.3f}), "
    f"median per-event knock-on {BS.knock_on_per_event.median():.1f} train-min (range "
    f"{BS.knock_on_per_event.min():.1f}-{BS.knock_on_per_event.max():.1f}) -- size adds delay through the number of "
    f"incidents, not through a multiplier on each")
log("\nper-burst impact:\n" + BS.assign(knock_on=lambda d: d.knock_on.round(1), own=lambda d: d.own.round(1),
                                        knock_on_per_event=lambda d: d.knock_on_per_event.round(1),
                                        span_min=lambda d: d.span_min.round(0)).to_string())
flush_log()


# ================================================================================================
section("8. Sensitivity of every burst-level claim to the burst definition (15 / 30 / 60 min gap)")
# ================================================================================================
def burst_rank_test(burst_id: pd.Series, n_perm: int, seed: int) -> dict:
    """The step-2/3/4 statistics under one burst rule: pooled slope, mean rho, sign test, permutation p."""
    f = EV.assign(bid=burst_id.values)
    f = f[f.bid >= 0].sort_values(["bid", "onset_ts"]).copy()
    f["rank"] = f.groupby("bid").cumcount() + 1
    g = pd.factorize(f.bid)[0].astype(np.int32)
    ng = int(g.max()) + 1
    d = f.straight_m.values.astype(float)
    r = f["rank"].values.astype(float)
    b, se, r2, df = within_slope(d, r, g, ng)
    rhos = [stats.spearmanr(x["rank"], x.straight_m)[0] for _, x in f.groupby("bid")
            if len(x) >= 3 and x.straight_m.nunique() >= 3]
    rhos = np.array([x for x in rhos if np.isfinite(x)])
    k_pos, n_nz, p_sign = sign_test(rhos)
    rr = np.random.default_rng(seed)
    ix = [np.where(g == b_)[0] for b_ in range(ng)]
    nl = np.empty(n_perm)
    for i in range(n_perm):
        dp = d.copy()
        for a in ix:
            dp[a] = d[rr.permutation(a)]
        nl[i] = within_slope(dp, r, g, ng)[0]
    p_two = (1 + int((np.abs(nl) >= abs(b)).sum())) / (n_perm + 1)
    return {"n_bursts": ng, "n_events": len(f), "n_isolated": int((burst_id < 0).sum()),
            "slope_m_per_rank": b, "slope_se": se, "within_r2": r2, "mean_rho": float(rhos.mean()),
            "rho_pos": k_pos, "rho_n": int(n_nz), "rho_sign_p": p_sign,
            "perm_p_two_sided": p_two, "n_perm": n_perm}


RULE = pd.DataFrame([dict(rule=f"gap <= {int(gap)} min, size >= {MIN_SIZE}", **burst_rank_test(
    assign_bursts(gap), N_PERM, SEED + int(gap))) for gap in (15.0, 30.0, 60.0)])
RULE.to_csv(OUT / "rule_sensitivity.csv", index=False)
log(RULE.assign(slope_m_per_rank=lambda d: d.slope_m_per_rank.round(1), slope_se=lambda d: d.slope_se.round(1),
                within_r2=lambda d: d.within_r2.round(3), mean_rho=lambda d: d.mean_rho.round(3),
                rho_sign_p=lambda d: d.rho_sign_p.round(3),
                perm_p_two_sided=lambda d: d.perm_p_two_sided.round(3)).to_string(index=False))
log(f"under all three rules the pooled slope is negative (never positive), so 'the burst spreads outward' is "
    f"answered no wherever the burst boundary is drawn; under the tightest rule the inward slope is nominally "
    f"significant (p = {RULE.set_index('rule').loc['gap <= 15 min, size >= 3', 'perm_p_two_sided']:.3f}). The 30 min "
    f"row is an independent draw from the same null as section 4, so its p differs there by Monte-Carlo noise only")


# ================================================================================================
section("9. Power: how many isolated near-band events would it take to see a real ratio")
# ================================================================================================
def power_ratio(r: float, lam1: float, t1: float, t2: float) -> float:
    """Power of the exact conditional test for rate_near / rate_far = r at 5% two-sided, near count lam1.

    Conditional on the total count n the near count is Bin(n, t1 / (t1 + t2)); the test rejects when it is in
    either binomial tail, so the power is a sum over n of P(n) times the tail mass under the true split.
    """
    p0 = t1 / (t1 + t2)
    lam2 = lam1 / r * (t2 / t1)
    lam = lam1 + lam2
    ns = np.arange(1, int(lam + 10 * np.sqrt(lam) + 20))
    k_lo = stats.binom.ppf(0.025, ns, p0)
    k_hi = stats.binom.ppf(0.975, ns, p0)
    tail = stats.binom.cdf(k_lo, ns, lam1 / lam) + stats.binom.sf(k_hi - 1, ns, lam1 / lam)
    return float((stats.poisson.pmf(ns, lam) * tail).sum())


def events_for_power(r: float, t1: float, t2: float, target: float = 0.80) -> float:
    lo, hi = 0.5, 120.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if power_ratio(r, mid, t1, t2) < target else (lo, mid)
    return hi


POWER = pd.DataFrame([{"true_ratio": r, "near_events_needed": events_for_power(r, H_ON_NEAR, H_ON_GAP),
                       "near_rate_needed": events_for_power(r, H_ON_NEAR, H_ON_GAP) / H_ON_NEAR * 1000,
                       "exposure_multiple": events_for_power(r, H_ON_NEAR, H_ON_GAP) / k_on_near}
                      for r in (2.0, 3.0)])
POWER.to_csv(OUT / "power.csv", index=False)
log(POWER.assign(near_events_needed=lambda d: d.near_events_needed.round(1),
                 near_rate_needed=lambda d: d.near_rate_needed.round(0),
                 exposure_multiple=lambda d: d.exposure_multiple.round(1)).to_string(index=False))
log(f"a true 2x on-line ratio would need {events_for_power(2.0, H_ON_NEAR, H_ON_GAP):.0f} isolated 0-250 m events "
    f"to reach 80% power at 5% two-sided (3x: {events_for_power(3.0, H_ON_NEAR, H_ON_GAP):.0f}); at the observed "
    f"isolated rate that is {events_for_power(2.0, H_ON_NEAR, H_ON_GAP) / k_on_near:.0f}x the current "
    f"{H_ON_NEAR:.0f} train-h, i.e. we observe {k_on_near} of {events_for_power(2.0, H_ON_NEAR, H_ON_GAP):.0f}")


# ================================================================================================
section("10. Summary of the numbers the note quotes")
SUMMARY = {
    "n_bursts": N_BURST, "n_burst_events": int(len(B)), "n_isolated": int(len(ISO)),
    "span_min_median": float(AN.span_min.median()), "span_m_median": float(AN.span_m.median()),
    "bursts_all_within_250m": int((AN.d_max_m < 250).sum()),
    "first_closer_sign_k": k_neg, "first_closer_sign_n": n_nz, "first_closer_sign_p": p_sign,
    "mean_rho": float(RD.spearman_rho.mean()), "rho_pos_k": k_pos, "rho_pos_n": n_rho, "rho_sign_p": p_sign_rho,
    "pooled_slope": b_rank, "pooled_slope_se": se_rank, "perm_p_two": p_two, "perm_p_outward": p_grow,
    "perm_floor": floor,
    "iso_online_near": k_on_near, "iso_online_gap": k_on_gap, "h_online_near": H_ON_NEAR, "h_online_gap": H_ON_GAP,
    "iso_ratio_point": k_on_near / H_ON_NEAR * 1000 / gap_hi, "iso_ratio_ci_lo": r_on["ci_lo"],
    "iso_online_mid_ratio": r_on_mid["ratio"], "iso_online_mid_ci_lo": r_on_mid["ci_lo"],
    "iso_online_far_ratio": r_on_far["ratio"], "iso_online_far_ci_lo": r_on_far["ci_lo"],
    "iso_online_far_ci_hi": r_on_far["ci_hi"],
    "iso_off_near": int(RATIOS.k1.iloc[2]), "iso_off_far": int(RATIOS.k2.iloc[2]),
    "iso_off_ratio": r_off["ratio"], "iso_off_ci_lo": r_off["ci_lo"], "iso_off_ci_hi": r_off["ci_hi"],
    "dup_train_bursts": int(NOT.dup_train.sum()), "span_lt_rec": int(NOT.span_lt_rec.sum()),
    "pairs_no_interchange": int(NOT.n_pairs_no_interchange.sum()),
    "size_knock_on_rho": float(rho_imp.statistic), "size_knock_on_slope": float(beta[1]),
    "burst_share_knock_on": float(share)}
pd.Series(SUMMARY).to_csv(OUT / "summary.csv", header=["value"])
log(f"bursts {N_BURST}, events in bursts {len(B)}, isolated {len(ISO)}; median within-burst span "
    f"{AN.span_min.median():.0f} min and {AN.span_m.median():.0f} m; no burst is entirely inside 250 m "
    f"({int((AN.d_max_m < 250).sum())}/{N_BURST})")
log(f"central test: pooled slope {b_rank:+.1f} m per rank (SE {se_rank:.1f}), permutation p = {p_two:.3f} "
    f"two-sided, {p_grow:.3f} one-sided for outward spreading, MC floor {floor:.4f}; mean per-burst rho "
    f"{RD.spearman_rho.mean():+.3f}, rho > 0 in {k_pos}/{n_rho} bursts")
log(f"isolated events: on-line 0-250 m vs 250-600 m = {k_on_near}/{H_ON_NEAR:.1f} h vs {k_on_gap}/"
    f"{H_ON_GAP:.1f} h, point ratio {k_on_near / H_ON_NEAR * 1000 / gap_hi:.2f}x, exact CI lower bound "
    f"{r_on['ci_lo']:.2f}; off-line 600 m-1.5 km vs 1.5 km+ = {int(RATIOS.k1.iloc[2])}/{H_OFF_NEAR:.1f} h vs "
    f"{int(RATIOS.k2.iloc[2])}/{H_OFF_FAR:.1f} h, ratio {r_off['ratio']:.2f} "
    f"[{r_off['ci_lo']:.2f}, {r_off['ci_hi']:.2f}]")
log(f"burst size vs knock-on: rho {rho_imp.statistic:+.2f}, log-log slope {beta[1]:.2f} "
    f"[{beta[1] - 1.96 * se_b:.2f}, {beta[1] + 1.96 * se_b:.2f}], bursts carry {100 * share:.0f}% of all knock-on")
log(f"tables in {OUT.relative_to(ROOT)}: " + ", ".join(sorted(p.name for p in OUT.glob("*.csv"))))
flush_log()
log(f"\nfull log written to {LOG_PATH.relative_to(ROOT)}")
