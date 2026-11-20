"""22 Peak ANOVA -- one standard variance decomposition of the victim-to-carrier distance.

Purpose
-------
The distance from the victim train to the carrier T3420 at the moment of the breakdown is multi-modal: Line 10
victims sit at 48-222 m, Line 8 near 1 km, Lines 2 and 12 near 1.5-2 km.  The paper needs ONE standard variance
decomposition saying how much of that spread is between-line geometry (each line's own station spacing) versus
within-line variation, and whether anything else -- where the victim is on its run (station or segment) or what the
carrier is doing -- adds explanatory power.  This is the last analysis of the project; nothing beyond this
decomposition is attempted.

Response: log10(straight_m), i.e. log10 of the physical distance in metres at the onset, for the 142 events with a
finite distance (the 1 remaining event is the "carrier off" case: not in service, so it has no position).

Sections
 1. The response: raw moments, log10 moments, Shapiro-Wilk on both, and the 14 events beyond 5 km reported both
    kept and excluded.  The log is justified by the skew, not assumed.
 2. One-way ANOVA per factor (line_id, victim_kind, carrier_kind): SS, df, F, p, eta-squared, plus group means with
    95% CIs on the log scale back-transformed to metres.
 3. THE MODEL: log10(distance) ~ C(line_id) + C(victim_kind) + C(carrier_kind) + line x victim_kind, type-II SS,
    partial eta-squared, R2 / adjusted R2, and the nested F sequence line -> victim_kind -> carrier_kind ->
    interaction with the incremental R2 of every block.
 4. GEOMETRY CHECK: the victim's own line's median segment length (one number per line) as the only regressor of
    log10(distance); the coefficient, its CI, the R2, and how much of the between-line SS that single number
    absorbs.  This is the "the peaks are station spacing" sentence, with a number.
 5. ROBUSTNESS: the same model on the 26 isolated events only, Kruskal-Wallis per factor, and a 5000-shuffle
    permutation F-test for the full model -- parametric and permutation p side by side.
 6. What the decomposition cannot show: line and geometry are collinear in this design.

Notes on the factor names
-------------------------
`victim_kind` is `location_type` of out/geo/events_geo.csv (station / segment): a breakdown is reported either at a
station (the train is dwelling) or between two stations.  It has two levels, not the three the brief anticipated --
the data has no separate "terminal" victim class.  `carrier_kind` is `carrier_state` (segment / station / terminal /
off); among the 142 finite-distance events only segment, station and terminal occur, because "off" means the carrier
had no position at all.

Read-only inputs: out/geo/events_geo.csv, out/geo/segment_lengths.csv, figures/screening/tables/breakdown_events.csv
(burst labels only).  Nothing in src/01-src/21 is re-run and paper/, data/, data_derived/ are not touched.

Outputs
-------
out/peak_anova/anova_oneway.csv        one-way table per factor and sample
out/peak_anova/group_means.csv         group means, log-scale 95% CIs, back-transformed metres
out/peak_anova/anova_model.csv         type-II ANOVA of the full model
out/peak_anova/nested_sequence.csv     incremental R2 of line -> victim_kind -> carrier_kind -> interaction
out/peak_anova/geometry_check.csv      median segment length per line, covariate regression, variance accounting
out/peak_anova/robustness.csv          isolated-only refit, Kruskal-Wallis, permutation F
out/peak_anova/peak_anova.txt          the full log (console shows <= 15 lines per call)
figures/peak_anova/fig_peak_anova.png  (a) log10 distance by line, (b) variance partition

Run: python3 src/22_peak_anova.py   (~10 s; reads the caches only)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.anova import anova_lm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GEO = ROOT / "out" / "geo"
TAB = ROOT / "figures" / "screening" / "tables"
OUT = ROOT / "out" / "peak_anova"
FIG = ROOT / "figures" / "peak_anova"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT / "peak_anova.txt"

FAR_M = 5000.0  # "beyond 5 km": reported both kept and excluded
N_PERM = 5000
SEED = 20260302
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


def fmt_p(p: float) -> str:
    """p-values span 1e-30 to 1; never print 0.000."""
    if not np.isfinite(p):
        return "n/a"
    return f"{p:.3g}" if p >= 1e-4 else f"{p:.2e}"


def eta2_table(df: pd.DataFrame, y: str, factor: str) -> dict:
    """Classic one-way ANOVA of y on factor: SS_between, SS_within, df, F, p, eta-squared."""
    g = df.groupby(factor, observed=True)[y]
    k, n = g.ngroups, len(df)
    grand = df[y].mean()
    ss_b = float((g.size() * (g.mean() - grand) ** 2).sum())
    ss_w = float(df[y].var(ddof=0) * n) - ss_b
    df_b, df_w = k - 1, n - k
    ms_b, ms_w = ss_b / df_b, ss_w / df_w
    f = ms_b / ms_w
    return {"levels": k, "n": n, "df_between": df_b, "df_within": df_w, "ss_between": ss_b,
            "ss_within": ss_w, "ms_between": ms_b, "ms_within": ms_w, "F": f,
            "p": float(stats.f.sf(f, df_b, df_w)), "eta2": ss_b / (ss_b + ss_w)}


def group_means(df: pd.DataFrame, y: str, factor: str, df_within: int) -> pd.DataFrame:
    """Group means on the log scale with t 95% CIs, back-transformed to metres (geometric mean)."""
    rows = []
    for level, s in df.groupby(factor, observed=True)[y]:
        n = len(s)
        m, se = float(s.mean()), float(s.std(ddof=1) / np.sqrt(n))
        half = float(stats.t.ppf(0.975, df_within)) * se
        rows.append({"factor": factor, "level": str(level), "n": n, "mean_log10": m, "se_log10": se,
                     "ci_lo_log10": m - half, "ci_hi_log10": m + half,
                     "geo_mean_m": 10.0 ** m, "ci_lo_m": 10.0 ** (m - half), "ci_hi_m": 10.0 ** (m + half)})
    return pd.DataFrame(rows)


# ================================================================================================
section("0. Data: the 142 finite distances, the two factors, the burst labels, the per-line spacing")
# ================================================================================================
BD = pd.read_csv(TAB / "breakdown_events.csv", parse_dates=["onset"])
EV = pd.read_csv(GEO / "events_geo.csv", parse_dates=["onset_ts"])
assert len(EV) == 143, f"expected 143 events, found {len(EV)}"
EV = EV.merge(BD[["train_id", "onset", "burst_id", "in_burst"]].rename(columns={"onset": "onset_ts"}),
              on=["train_id", "onset_ts"], how="left")
assert EV.burst_id.notna().all(), "burst labels did not match every event"

N_ALL = len(EV)
EV = EV[np.isfinite(EV.straight_m)].copy()  # the carrier-off event has no position -> no distance
EV["y"] = np.log10(EV.straight_m)
EV["line"] = EV.line_id.astype(str)
EV["victim_kind"] = EV.location_type  # station / segment
EV["carrier_kind"] = EV.carrier_state  # segment / station / terminal (off never survives the filter)
EV["isolated"] = EV.burst_id < 0
assert len(EV) == 142, f"expected 142 finite distances, found {len(EV)}"
assert set(EV.victim_kind) == {"station", "segment"}, f"unexpected victim_kind levels {set(EV.victim_kind)}"
assert set(EV.carrier_kind) == {"segment", "station", "terminal"}, \
    f"unexpected carrier_kind levels {set(EV.carrier_kind)}"

SEG = pd.read_csv(GEO / "segment_lengths.csv")
SPACING = SEG.groupby("line_id").length_m.median().rename("median_seg_m")
assert set(SPACING.index.astype(str)) >= set(EV.line), "a line has no segment lengths"
EV["median_seg_m"] = EV.line.map(lambda s: float(SPACING.loc[int(s)]))
EV["log_spacing"] = np.log10(EV.median_seg_m)
assert EV.log_spacing.notna().all()

FAR = EV.straight_m > FAR_M
SAMPLES = {"all": EV, "excl_gt5km": EV[~FAR]}
assert len(SAMPLES["all"]) == 142 and len(SAMPLES["excl_gt5km"]) == 142 - int(FAR.sum())

log(f"events with a finite distance: {len(EV)} of {N_ALL} (dropped: the carrier-off event, no position); "
    f"{int(EV.isolated.sum())} isolated, {int((~EV.isolated).sum())} inside the 23 bursts")
log(f"factor levels -- line_id {EV.line.value_counts().sort_index().to_dict()}; "
    f"victim_kind {EV.victim_kind.value_counts().to_dict()}; "
    f"carrier_kind {EV.carrier_kind.value_counts().to_dict()}")
log(f"per-line median segment length (m): "
    f"{ {str(k): round(float(v), 1) for k, v in SPACING.items()} }")

# ================================================================================================
section("1. The response: raw is heavily skewed, log10 is the analysis scale")
# ================================================================================================
RESP = []
for name, v, scale in (("straight_m", EV.straight_m, "m"), ("log10(straight_m)", EV.y, "log10 m")):
    sh_w, sh_p = stats.shapiro(v)
    RESP.append({"response": name, "unit": scale, "n": len(v), "mean": float(np.mean(v)),
                 "sd": float(np.std(v, ddof=1)), "median": float(np.median(v)), "min": float(np.min(v)),
                 "max": float(np.max(v)), "skew": float(stats.skew(v)), "excess_kurtosis": float(stats.kurtosis(v)),
                 "shapiro_W": float(sh_w), "shapiro_p": float(sh_p)})
RESP = pd.DataFrame(RESP)
log(RESP.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
log(f"raw spans {EV.straight_m.min():.0f} m to {EV.straight_m.max():,.0f} m (ratio "
    f"{EV.straight_m.max() / EV.straight_m.min():,.0f}x, skew {stats.skew(EV.straight_m):.2f}); on log10 the spread is "
    f"{EV.y.min():.2f} to {EV.y.max():.2f} decades, skew {stats.skew(EV.y):.2f}")
log(f"Shapiro-Wilk rejects normality on BOTH scales (raw p={fmt_p(RESP.shapiro_p[0])}, "
    f"log p={fmt_p(RESP.shapiro_p[1])}): the log removes the 610x scale of the raw distances but not the residual "
    f"shape, which is why "
    f"section 5 adds rank-based and permutation tests rather than relying on the F p-values alone")

log(f"events beyond {FAR_M / 1000:.0f} km: {int(FAR.sum())} of {len(EV)} "
    f"(largest {EV.straight_m.max():,.0f} m = {EV.y.max():.2f} decades); kept in the primary sample, dropped in "
    f"the 'excl_gt5km' sensitivity sample (n={len(SAMPLES['excl_gt5km'])})")

# ================================================================================================
section("2. One-way decompositions: how much of the log-distance spread does each factor carry on its own?")
# ================================================================================================
FACTORS = ["line", "victim_kind", "carrier_kind"]
ONEWAY, MEANS = [], []
for sample, frame in SAMPLES.items():
    for factor in FACTORS:
        row = {"sample": sample, "factor": factor, **eta2_table(frame, "y", factor)}
        ONEWAY.append(row)
        gm = group_means(frame, "y", factor, int(row["df_within"])).assign(sample=sample)
        MEANS.append(gm)
ONEWAY = pd.DataFrame(ONEWAY)
MEANS = pd.concat(MEANS, ignore_index=True)[
    ["sample", "factor", "level", "n", "mean_log10", "se_log10", "ci_lo_log10", "ci_hi_log10",
     "geo_mean_m", "ci_lo_m", "ci_hi_m"]]

for sample in SAMPLES:
    sub = ONEWAY[ONEWAY["sample"] == sample]
    log(f"\n[{sample}] one-way ANOVA of log10(straight_m), n={int(sub.n.iloc[0])}:")
    for _, r in sub.iterrows():
        log(f"  {r['factor']:<12} k={int(r.levels)}  SS_between={r.ss_between:7.3f}  SS_within={r.ss_within:7.3f}  "
            f"df=({int(r.df_between)},{int(r.df_within)})  F={r.F:7.2f}  p={fmt_p(r.p):>9}  eta2={r.eta2:.3f}")

PRIMARY = ONEWAY[(ONEWAY["sample"] == "all")].set_index("factor")
log(f"\non the full sample the single factor explains: line_id eta2={PRIMARY.eta2['line']:.3f}, "
    f"victim_kind eta2={PRIMARY.eta2['victim_kind']:.3f}, carrier_kind eta2={PRIMARY.eta2['carrier_kind']:.3f} "
    f"of the total sum of squares; the three factors are not orthogonal, so these do not add up (section 3)")
log("\ngroup means, log scale, back-transformed to metres (geometric mean [95% CI]), full sample:")
for _, r in MEANS[MEANS["sample"] == "all"].iterrows():
    log(f"  {r['factor']:<12} {r['level']:<9} n={int(r.n):>3}  log10={r.mean_log10:6.3f} "
        f"[{r.ci_lo_log10:6.3f}, {r.ci_hi_log10:6.3f}]  -> {r.geo_mean_m:8.0f} m "
        f"[{r.ci_lo_m:7.0f}, {r.ci_hi_m:8.0f}]")

# ================================================================================================
section("3. The model that matters: type-II ANOVA, partial eta2, and the nested R2 sequence")
# ================================================================================================
FULL_F = "y ~ C(line) + C(victim_kind) + C(carrier_kind) + C(line):C(victim_kind)"
STEPS = [("line", "y ~ C(line)"),
         ("victim_kind", "y ~ C(line) + C(victim_kind)"),
         ("carrier_kind", "y ~ C(line) + C(victim_kind) + C(carrier_kind)"),
         ("line x victim_kind", FULL_F)]

MODEL, NESTED, FITS = [], [], {}
for sample, frame in SAMPLES.items():
    fit = smf.ols(FULL_F, frame).fit()
    FITS[sample] = fit
    tab = anova_lm(fit, typ=2)
    ss_res, df_res = float(tab.loc["Residual", "sum_sq"]), float(tab.loc["Residual", "df"])
    ss_tot = float(tab["sum_sq"].sum())
    for term in tab.index:
        ss = float(tab.loc[term, "sum_sq"])
        MODEL.append({"sample": sample, "term": term, "ss_type2": ss, "df": float(tab.loc[term, "df"]),
                      "F": float(tab.loc[term, "F"]) if term != "Residual" else np.nan,
                      "p": float(tab.loc[term, "PR(>F)"]) if term != "Residual" else np.nan,
                      "partial_eta2": ss / (ss + ss_res) if term != "Residual" else np.nan,
                      "share_of_total_ss": ss / ss_tot})
    prev_sse, prev_df = float(((frame.y - frame.y.mean()) ** 2).sum()), 0
    for step, formula in STEPS:
        f = smf.ols(formula, frame).fit()
        sse, df_m, df_r = float(f.ssr), int(f.df_model), int(f.df_resid)
        df_add = df_m - prev_df
        f_change = ((prev_sse - sse) / df_add) / (sse / df_r)
        NESTED.append({"sample": sample, "step": step, "df_added": df_add, "ss_added": prev_sse - sse,
                       "r2": float(f.rsquared), "r2_adj": float(f.rsquared_adj),
                       "delta_r2": float(f.rsquared) - (0.0 if step == "line" else NESTED[-1]["r2"]),
                       "F_change": f_change, "p_change": float(stats.f.sf(f_change, df_add, df_r)),
                       "df_resid": df_r})
        prev_sse, prev_df = sse, df_m
MODEL, NESTED = pd.DataFrame(MODEL), pd.DataFrame(NESTED)

for sample in SAMPLES:
    m = MODEL[MODEL["sample"] == sample]
    n = NESTED[NESTED["sample"] == sample]
    log(f"\n[{sample}] type-II ANOVA, log10(straight_m) ~ line_id + victim_kind + carrier_kind + line_id x "
        f"victim_kind  (n={len(SAMPLES[sample])}, R2={n.r2.iloc[-1]:.3f}, adj R2={n.r2_adj.iloc[-1]:.3f}, "
        f"residual df={int(n.df_resid.iloc[-1])}):")
    for _, r in m.iterrows():
        if r["term"] == "Residual":
            log(f"  {'Residual':<22} SS={r.ss_type2:7.3f}  df={int(r.df):>3}  share={r.share_of_total_ss:.3f}")
        else:
            log(f"  {r['term']:<22} SS={r.ss_type2:7.3f}  df={int(r.df):>3}  F={r.F:7.2f}  p={fmt_p(r.p):>9}  "
                f"partial eta2={r.partial_eta2:.3f}  share={r.share_of_total_ss:.3f}")
    log(f"  nested sequence (order matters -- later blocks get what the earlier ones leave):")
    for _, r in n.iterrows():
        log(f"    +{r.step:<20} df+{int(r.df_added):>2}  R2={r.r2:.3f}  dR2={r.delta_r2:+.3f}  "
            f"F({int(r.df_added)},{int(r.df_resid)})={r.F_change:7.2f}  p={fmt_p(r.p_change):>9}")

TERM = EV[EV.carrier_kind == "terminal"]
log(f"all {len(TERM)} carrier-terminal events sit beyond 5 km (min {TERM.straight_m.min():,.0f} m), so the "
    f"carrier_kind effect is mostly 'the carrier is standing at a line terminus, far from the victim': its share of "
    f"the total SS falls from {MODEL[(MODEL['sample'] == 'all') & (MODEL.term == 'C(carrier_kind)')].share_of_total_ss.iloc[0]:.3f} "
    f"to {MODEL[(MODEL['sample'] == 'excl_gt5km') & (MODEL.term == 'C(carrier_kind)')].share_of_total_ss.iloc[0]:.3f} "
    f"when the 5 km tail is trimmed (terminal then has no events left)")

n_all = NESTED[NESTED["sample"] == "all"]
log(f"\nline_id alone explains {n_all.r2.iloc[0]:.3f} of the variance of log10(distance) "
    f"({n_all.r2.iloc[0] * 100:.1f}%); adding victim_kind, carrier_kind and the interaction takes R2 to "
    f"{n_all.r2.iloc[-1]:.3f}, i.e. {n_all.r2.iloc[-1] - n_all.r2.iloc[0]:+.3f} for everything else combined")

# ================================================================================================
section("4. Geometry check: is the between-line structure just station spacing?")
# ================================================================================================
GEOCHK = []
line_tab = (EV.groupby("line", observed=True)
            .agg(n=("y", "size"), median_seg_m=("median_seg_m", "first"), geo_mean_m=("y", lambda s: 10.0 ** s.mean()))
            .reset_index().sort_values("median_seg_m"))
for _, r in line_tab.iterrows():
    GEOCHK.append({"quantity": "line_median_segment_length_m", "group": f"line {r.line}", "value": r.median_seg_m,
                   "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan, "note": f"n_events={int(r.n)}"})
    GEOCHK.append({"quantity": "line_geo_mean_distance_m", "group": f"line {r.line}", "value": r.geo_mean_m,
                   "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan, "note": f"n_events={int(r.n)}"})

geo_fit = smf.ols("y ~ log_spacing", EV).fit()
b, se_b = float(geo_fit.params["log_spacing"]), float(geo_fit.bse["log_spacing"])
ci_b = geo_fit.conf_int().loc["log_spacing"]
p_b = float(geo_fit.pvalues["log_spacing"])
r2_geo = float(geo_fit.rsquared)
tss = float(((EV.y - EV.y.mean()) ** 2).sum())
ss_geo = r2_geo * tss
ss_line = float(PRIMARY.ss_between["line"])
r2_line = float(NESTED[(NESTED["sample"] == "all") & (NESTED.step == "line")].r2.iloc[0])
r2_line_kinds = float(NESTED[(NESTED["sample"] == "all") & (NESTED.step == "carrier_kind")].r2.iloc[0])
r2_geo_kinds = float(smf.ols("y ~ log_spacing + C(victim_kind) + C(carrier_kind)", EV).fit().rsquared)
doubling = 10.0 ** (b * np.log10(2.0))
rho_geo = float(stats.spearmanr(line_tab.median_seg_m, line_tab.geo_mean_m).statistic)

GEOCHK += [
    {"quantity": "regression_slope", "group": "log10(median seg len)", "value": b, "ci_lo": float(ci_b.iloc[0]),
     "ci_hi": float(ci_b.iloc[1]), "p": p_b, "note": f"se={se_b:.3f}; log10(dist) per decade of spacing"},
    {"quantity": "distance_factor_per_spacing_doubling", "group": "log10(median seg len)", "value": doubling,
     "ci_lo": 10.0 ** (float(ci_b.iloc[0]) * np.log10(2.0)), "ci_hi": 10.0 ** (float(ci_b.iloc[1]) * np.log10(2.0)),
     "p": np.nan, "note": "back-transformed multiplicative effect of doubling the line's median segment"},
    {"quantity": "r2_covariate_only", "group": "log10(median seg len)", "value": r2_geo, "ci_lo": np.nan,
     "ci_hi": np.nan, "p": p_b, "note": "log10(distance) ~ log10(median segment length), n=142"},
    {"quantity": "r2_line_factor", "group": "C(line_id)", "value": r2_line, "ci_lo": np.nan, "ci_hi": np.nan,
     "p": float(PRIMARY.p["line"]), "note": "3 df: one free mean per line"},
    {"quantity": "between_line_ss", "group": "C(line_id)", "value": ss_line, "ci_lo": np.nan, "ci_hi": np.nan,
     "p": np.nan, "note": "SS_between of the one-way line table"},
    {"quantity": "covariate_ss", "group": "log10(median seg len)", "value": ss_geo, "ci_lo": np.nan, "ci_hi": np.nan,
     "p": np.nan, "note": "1 df: the same SS spent by one number per line"},
    {"quantity": "covariate_share_of_between_line_ss", "group": "log10(median seg len)", "value": ss_geo / ss_line,
     "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan, "note": "how much of the line effect station spacing absorbs"},
    {"quantity": "r2_covariate_plus_kinds", "group": "log10(median seg len) + victim + carrier",
     "value": r2_geo_kinds, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan,
     "note": "4 df: spacing, victim_kind, carrier_kind"},
    {"quantity": "r2_line_plus_kinds", "group": "C(line_id) + victim + carrier", "value": r2_line_kinds,
     "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan, "note": "6 df: the same model with the line factor instead"},
    {"quantity": "spacing_order_concordance", "group": "4 line means", "value": rho_geo, "ci_lo": np.nan,
     "ci_hi": np.nan, "p": 1.0 / 24.0,
     "note": "Spearman rho of median spacing vs geometric-mean distance across the 4 lines; exact one-sided p "
             "= 1/4! = 0.042"},
]
GEOCHK = pd.DataFrame(GEOCHK)

log("\nper line, the single spacing number against the distance peak it is supposed to explain:")
for _, r in line_tab.iterrows():
    log(f"  line {r.line:<3} median segment {r.median_seg_m:7.0f} m  ->  geometric-mean distance "
        f"{r.geo_mean_m:7.0f} m  (n={int(r.n)})")
log(f"\nlog10(distance) ~ log10(median segment length): slope {b:+.3f} [{float(ci_b.iloc[0]):+.3f}, "
    f"{float(ci_b.iloc[1]):+.3f}], p={fmt_p(p_b)}, R2={r2_geo:.3f}; a doubling of a line's median segment length "
    f"multiplies its victims' distance by {doubling:.2f}x [{10.0 ** (float(ci_b.iloc[0]) * np.log10(2.0)):.2f}, "
    f"{10.0 ** (float(ci_b.iloc[1]) * np.log10(2.0)):.2f}]")
log(f"station spacing is one number per line, so it can only reach {r2_geo:.3f} of the total variance against the "
    f"line factor's {r2_line:.3f}: it absorbs {ss_geo / ss_line * 100:.0f}% of the between-line SS "
    f"({ss_geo:.2f} of {ss_line:.2f})")
log(f"spacing + victim_kind + carrier_kind reaches R2={r2_geo_kinds:.3f}; the same three-part model with the line "
    f"factor instead reaches R2={r2_line_kinds:.3f}")
log(f"the four line means order exactly as their median spacings do (Spearman rho={rho_geo:+.2f}, exact one-sided "
    f"p=1/4!={1 / 24:.3f}), but the x range is only {line_tab.median_seg_m.min():.0f}-{line_tab.median_seg_m.max():.0f} "
    f"m ({line_tab.median_seg_m.max() / line_tab.median_seg_m.min():.2f}x) against a distance range of "
    f"{line_tab.geo_mean_m.min():.0f}-{line_tab.geo_mean_m.max():.0f} m "
    f"({line_tab.geo_mean_m.max() / line_tab.geo_mean_m.min():.1f}x): spacing orders the peaks, it does not set their "
    f"magnitude, and the {doubling:.0f}x-per-doubling slope is an extrapolation across four points")

# ================================================================================================
section("5. Robustness: isolated events only, Kruskal-Wallis, and a 5000-shuffle permutation F")
# ================================================================================================
def perm_f_test(frame: pd.DataFrame, formula: str, n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """Shuffle the response, keep the design fixed, recompute the overall F of the same model."""
    fit = smf.ols(formula, frame).fit()
    X, y = np.asarray(fit.model.exog, dtype=float), np.asarray(fit.model.endog, dtype=float)
    n = len(y)
    df_slope = int(np.linalg.matrix_rank(X)) - 1
    df_resid = n - df_slope - 1

    def f_stat(yv: np.ndarray) -> float:
        beta = np.linalg.lstsq(X, yv, rcond=None)[0]
        resid = yv - X @ beta
        r2 = 1.0 - float(resid @ resid) / float(((yv - yv.mean()) ** 2).sum())
        return (r2 / df_slope) / ((1.0 - r2) / df_resid)

    f_obs = f_stat(y)
    rng = np.random.default_rng(seed)
    null = np.array([f_stat(rng.permutation(y)) for _ in range(n_perm)])
    return {"F": f_obs, "df1": df_slope, "df2": df_resid, "p_param": float(stats.f.sf(f_obs, df_slope, df_resid)),
            "p_perm": float((1 + int((null >= f_obs).sum())) / (1 + n_perm)), "null": null}


def kw_test(frame: pd.DataFrame, factor: str, n_perm: int = N_PERM, seed: int = SEED) -> dict:
    """Kruskal-Wallis on log10(distance) by factor, with a label-shuffle p next to the chi-square p."""
    groups = [g.to_numpy() for _, g in frame.groupby(factor, observed=True)["y"]]
    h_obs, p_chi = stats.kruskal(*groups)
    rng = np.random.default_rng(seed)
    y = frame.y.to_numpy()
    codes = pd.factorize(frame[factor], sort=True)[0]
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(codes)
        null[i] = stats.kruskal(*[y[perm == c] for c in np.unique(codes)]).statistic
    return {"H": float(h_obs), "df": len(groups) - 1, "p_param": float(p_chi),
            "p_perm": float((1 + int((null >= h_obs).sum())) / (1 + n_perm)), "null": null}


ROBUST = []
ISO = EV[EV.isolated].copy()
N_FAR = int(FAR.sum())
N_FAR_BURST = int((FAR & ~EV.isolated).sum())
log(f"the isolated sample is not a clean middle band: {N_FAR} events sit beyond 5 km and {N_FAR - N_FAR_BURST} of "
    f"them are isolated ({N_FAR_BURST} inside a burst), so the isolated refit is {N_FAR} far + {len(ISO) - N_FAR} "
    f"near/mid events and its within-line variance is carried by that tail")

iso_cells = ISO.groupby(["line", "victim_kind"], observed=True).size().unstack(fill_value=0)
iso_rank_ok = bool((iso_cells > 0).all().all())  # every line needs both victim kinds for the interaction
ISO_F = FULL_F if iso_rank_ok else "y ~ C(line) + C(victim_kind) + C(carrier_kind)"
ISO_NOTE = ("full model" if iso_rank_ok else
            "additive model -- lines 10 and 8 have no station victim among the isolated events, so the interaction is "
            "not estimable at n=26")
iso_fit = smf.ols(ISO_F, ISO).fit()
iso_X = np.asarray(iso_fit.model.exog, dtype=float)
assert np.linalg.matrix_rank(iso_X) == iso_X.shape[1], "the isolated refit is rank-deficient"
iso_tab = anova_lm(iso_fit, typ=2)
iso_res = float(iso_tab.loc["Residual", "sum_sq"])
for term in iso_tab.index:
    if term == "Residual":
        continue
    ss = float(iso_tab.loc[term, "sum_sq"])
    ROBUST.append({"sample": "isolated (n=26)", "test": "anova_type2", "term": term,
                   "statistic": float(iso_tab.loc[term, "F"]), "df": float(iso_tab.loc[term, "df"]),
                   "p_parametric": float(iso_tab.loc[term, "PR(>F)"]), "p_permutation": np.nan, "n": len(ISO),
                   "note": f"partial eta2={ss / (ss + iso_res):.3f}; {ISO_NOTE}"})

for sample, frame in SAMPLES.items():
    for factor in FACTORS:
        r = kw_test(frame, factor)
        ROBUST.append({"sample": sample, "test": "kruskal_wallis", "term": factor, "statistic": r["H"],
                       "df": r["df"], "p_parametric": r["p_param"], "p_permutation": r["p_perm"], "n": len(frame),
                       "note": "rank test on log10(distance)"})
    for label, formula in (("full model", FULL_F), ("line only", "y ~ C(line)")):
        r = perm_f_test(frame, formula)
        ROBUST.append({"sample": sample, "test": "permutation_F", "term": label, "statistic": r["F"],
                       "df": r["df1"], "p_parametric": r["p_param"], "p_permutation": r["p_perm"], "n": len(frame),
                       "note": f"F({r['df1']},{r['df2']}), {N_PERM} response shuffles"})
ROBUST = pd.DataFrame(ROBUST)

log(f"isolated events only: n={len(ISO)} of the 27 isolated events (one is the carrier-off case, no distance). "
    f"Model used: {ISO_NOTE}; R2={iso_fit.rsquared:.3f}, adj R2={iso_fit.rsquared_adj:.3f}")
for _, r in ROBUST[ROBUST["sample"] == "isolated (n=26)"].iterrows():
    log(f"  {r.term:<22} F={r.statistic:7.2f}  df={int(r.df):>2}  p={fmt_p(r.p_parametric)}")
log("\nparametric vs permutation (5000 shuffles):")
for _, r in ROBUST[ROBUST["test"] != "anova_type2"].iterrows():
    stat = f"{r.statistic:7.2f}"
    log(f"  [{r['sample']:<10}] {r.test:<16} {r.term:<12} {stat}  df={int(r.df):>2}  "
        f"p_param={fmt_p(r.p_parametric):>9}  p_perm={fmt_p(r.p_permutation):>9}")

# ================================================================================================
section("6. What this decomposition does not show")
# ================================================================================================
log("A variance decomposition says where the spread lives, not why a victim fails. The line factor is large and the\n"
    "two behavioural factors are small, but that is a statement about the geography of the 143 breakdowns, not about a\n"
    "mechanism: a line with a wide distance distribution can be wide because its geometry puts trains far from the\n"
    "carrier, or because its trains behave differently, and in this design the line, its station spacing, its fleet\n"
    "and its timetable are collinear -- one number per line cannot separate them. The geometry check in section 4 is\n"
    "therefore an upper bound on the spacing story, not a test of it, and nothing here identifies a causal channel.")

# ================================================================================================
section("7. Outputs")
# ================================================================================================
ONEWAY.to_csv(OUT / "anova_oneway.csv", index=False)
MEANS.to_csv(OUT / "group_means.csv", index=False)
MODEL.to_csv(OUT / "anova_model.csv", index=False)
NESTED.to_csv(OUT / "nested_sequence.csv", index=False)
GEOCHK.to_csv(OUT / "geometry_check.csv", index=False)
ROBUST.to_csv(OUT / "robustness.csv", index=False)

m_all = MODEL[(MODEL["sample"] == "all") & (MODEL.term != "Residual")].set_index("term")
share = {k: float(m_all.share_of_total_ss[k]) for k in m_all.index}
share_resid = float(MODEL[(MODEL["sample"] == "all") & (MODEL.term == "Residual")].share_of_total_ss.iloc[0])
order = EV.groupby("line", observed=True).y.median().sort_values().index.tolist()

fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), gridspec_kw={"width_ratios": [1.35, 1.0]})
ax = axes[0]
rng = np.random.default_rng(SEED)
COLOURS = {"segment": "#1f77b4", "station": "#d62728"}
for i, line in enumerate(order):
    s = EV[EV.line == line]
    ax.boxplot(s.y, positions=[i], widths=0.5, showfliers=False, patch_artist=True,
               boxprops={"facecolor": "#eeeeee", "edgecolor": "#444444"},
               medianprops={"color": "#111111", "lw": 1.6}, whiskerprops={"color": "#444444"},
               capprops={"color": "#444444"})
    for kind, colour in COLOURS.items():
        k = s[s.victim_kind == kind]
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(k)), k.y, s=26, color=colour, alpha=0.8,
                   edgecolor="white", linewidth=0.4, label=f"victim {kind}" if i == 0 else None, zorder=3)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([f"Line {ln}\nn={len(EV[EV.line == ln])}, seg {EV[EV.line == ln].median_seg_m.iloc[0]:.0f} m"
                    for ln in order], fontsize=9)
ax.set_ylabel("distance to carrier at onset  (log$_{10}$ m)")
ax.set_title(f"(a) log$_{{10}}$ straight-line distance by line (n={len(EV)})", fontsize=10.5)
ticks = [50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000]
ax.set_yticks(np.log10(ticks))
ax.set_yticklabels([f"{t/1000:.2f} km" if t >= 1000 else f"{t} m" for t in ticks])
ax.set_ylim(1.25, 4.6)
ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6)
ax.legend(fontsize=8, loc="lower right", frameon=False)

ax = axes[1]
bars = [("line_id", share["C(line)"]), ("victim_kind", share["C(victim_kind)"]),
        ("carrier_kind", share["C(carrier_kind)"]), ("residual", share_resid)]
cols = ["#4c72b0", "#dd8452", "#55a868", "#bbbbbb"]
ax.bar([b[0] for b in bars], [b[1] for b in bars], color=cols, edgecolor="white")
for i, (label, v) in enumerate(bars):
    ax.annotate(f"{v * 100:.1f}%", (i, v), ha="center", va="bottom", fontsize=9)
ax.set_ylim(0, max(v for _, v in bars) * 1.22)
ax.set_ylabel("share of total sum of squares")
ax.set_title("(b) type-II variance partition of the full model", fontsize=10.5)
ax.annotate(f"line x victim_kind: {share['C(line):C(victim_kind)'] * 100:.1f}% (not drawn)\n"
            f"R$^2$ = {NESTED[NESTED['sample'] == 'all'].r2.iloc[-1]:.3f}",
            (0.02, 0.94), xycoords="axes fraction", va="top", fontsize=8, color="#444444")
ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6)
fig.tight_layout()
fig.savefig(FIG / "fig_peak_anova.png", dpi=200)
plt.close(fig)

flush_log()
log(f"wrote {OUT.name}/{{anova_oneway, group_means, anova_model, nested_sequence, geometry_check, robustness}}.csv "
    f"and {LOG_PATH.name}; figure {FIG.name}/fig_peak_anova.png")
