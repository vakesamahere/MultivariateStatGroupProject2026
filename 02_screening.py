"""02 Exploratory screening of breakdown drivers (Shanghai Metro semi-synthetic data).

Reproducible screening pass. Reads the small CSVs in data/ plus a fixed-size head sample of
normal_operations.csv (first service date only) for rough exposure proxies. The full exposure
correction is done in a later step; rates here are "naive" (every train assumed in service all
105 service dates) unless stated otherwise.

Outputs
- figures/screening/*.png              figures
- figures/screening/tables/*.csv       small result tables
- figures/screening/screening_results.txt  full numeric log (interpretation lives in notes/02_screening.md)

Run: python src/02_screening.py   (about a minute; seeds fixed)
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
# Sparse type x batch / batch x route interaction fits (and their bootstrap refits) are rank-deficient by
# design; LR statistics remain valid, so silence the repeated statsmodels warning.
warnings.filterwarnings("ignore", message="The design matrix is rank-deficient")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures" / "screening"
TAB = FIG / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
LOG_PATH = FIG / "screening_results.txt"

SEED = 20260302
N_PERM = 5000
N_SERVICE_DAYS = 105
STUDY_START = pd.Timestamp("2026-03-02")
BURST_GAP_MIN = 30  # consecutive network-wide breakdowns <= this many minutes apart share a burst
BURST_MIN_SIZE = 3  # a burst needs at least this many breakdowns
NORMAL_OPS_SAMPLE_ROWS = 120_000  # covers the whole first service date (2026-03-02, ~79.7k rows)
HOT_C = 30.0
HEAVY_RAIN_MM = 10.0
COLD_C = 10.0

rng = np.random.default_rng(SEED)
sns.set_theme(style="whitegrid", context="paper")
_log_lines: list[str] = []


def log(*parts) -> None:
    text = " ".join(str(p) for p in parts)
    print(text)
    _log_lines.append(text)


def section(title: str) -> None:
    log("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def flush_log() -> None:
    LOG_PATH.write_text("\n".join(_log_lines) + "\n", encoding="utf-8")


def read_csv(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(DATA / name, encoding="utf-8-sig", dtype=str, **kw)


def savefig(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def poisson_ci(k, exposure, level=0.95):
    """Exact (Garwood) CI for a Poisson rate k/exposure."""
    k = np.asarray(k, dtype=float)
    a = (1 - level) / 2
    lo = np.where(k > 0, stats.chi2.ppf(a, 2 * k) / 2, 0.0)
    hi = stats.chi2.ppf(1 - a, 2 * k + 2) / 2
    return lo / exposure, hi / exposure


def rate_table(df: pd.DataFrame, factor: str, count: str = "n_bd", exposure: str = "train_days",
               per: float = 1000.0) -> pd.DataFrame:
    g = df.groupby(factor, observed=True).agg(n_trains=("train_id", "size"), events=(count, "sum"),
                                              exposure=(exposure, "sum"))
    g["rate"] = g.events / g.exposure * per
    lo, hi = poisson_ci(g.events.values, g.exposure.values)
    g["rate_lo"], g["rate_hi"] = lo * per, hi * per
    return g


def group_chisq(counts: np.ndarray, groups: np.ndarray, weights: np.ndarray) -> float:
    """Pearson statistic of group totals vs expectation proportional to group exposure."""
    tot = counts.sum()
    df = pd.DataFrame({"c": counts, "g": groups, "w": weights}).groupby("g").sum()
    exp = tot * df.w / df.w.sum()
    return float(((df.c - exp) ** 2 / exp).sum())


def label_permutation_p(counts, groups, weights, n_perm=N_PERM) -> tuple[float, float]:
    """Permute group labels across units (trains); keeps each train's count intact."""
    counts, groups, weights = map(np.asarray, (counts, groups, weights))
    obs = group_chisq(counts, groups, weights)
    perm = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(len(groups))
        perm[i] = group_chisq(counts, groups[idx], weights)
    return obs, (1 + np.sum(perm >= obs - 1e-12)) / (n_perm + 1)


def glm_rate_ratios(formula: str, data: pd.DataFrame, offset_col: str = "log_exposure",
                    cov_type: str = "nonrobust", cov_kwds=None) -> pd.DataFrame:
    model = smf.glm(formula, data=data, family=sm.families.Poisson(), offset=data[offset_col])
    res = model.fit(cov_type=cov_type, cov_kwds=cov_kwds) if cov_kwds else model.fit(cov_type=cov_type)
    ci = res.conf_int()
    out = pd.DataFrame({"RR": np.exp(res.params), "lo": np.exp(ci[0]), "hi": np.exp(ci[1]), "p": res.pvalues})
    return out, res


def lr_test(full, reduced) -> tuple[float, int, float]:
    stat = 2 * (full.llf - reduced.llf)
    ddf = int(round(full.df_model - reduced.df_model))
    return stat, ddf, stats.chi2.sf(stat, ddf)


# ------------------------------------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------------------------------------
section("1. Data load and integrity checks")
trains = read_csv("train_information.csv")
trains["service_start_date"] = pd.to_datetime(trains.service_start_date)
trains["age_years"] = (STUDY_START - trains.service_start_date).dt.days / 365.25
trains["age_band"] = pd.cut(trains.age_years, [0, 3, 7, 11, 16], labels=["<3y", "3-7y", "7-11y", "11-15y"])

stations = read_csv("station_information.csv")
station_lines = stations.groupby("station_id").line_id.apply(lambda s: ",".join(sorted(s.unique())))
station_name = stations.drop_duplicates("station_id").set_index("station_id").station_name_english

weather = read_csv("temperature.csv").merge(read_csv("rainfall.csv"), on="date", how="outer")
weather["temp_c"] = weather.mean_temperature_c.astype(float)
weather["rain_mm"] = weather.rainfall_mm.astype(float)
weather = weather[["date", "temp_c", "rain_mm"]].sort_values("date").reset_index(drop=True)
service_dates = weather.date.tolist()

cleaning = read_csv("cleaning_records.csv")
cleaning["completed_at"] = pd.to_datetime(cleaning.completed_at)
cleaning["date"] = cleaning.completed_at.dt.strftime("%Y-%m-%d")

bd_raw = read_csv("breakdown_recovery.csv")
bd_raw["ts"] = pd.to_datetime(bd_raw.timestamp)

log(f"trains={len(trains)}, station rows={len(stations)} ({stations.station_id.nunique()} nodes), "
    f"weather days={len(weather)}, cleaning rows={len(cleaning)}, breakdown/recovery rows={len(bd_raw)}")
log("event counts:", bd_raw.event_description.value_counts().to_dict())
per_train_day = cleaning.groupby(["train_id", "date"]).size()
log(f"cleanings per train-day: {per_train_day.value_counts().to_dict()}  "
    f"(train-days covered {len(per_train_day)} = {trains.shape[0]} x {N_SERVICE_DAYS})")
log("cleaning completion hours:", cleaning.completed_at.dt.hour.value_counts().sort_index().to_dict())
assert set(cleaning.date) == set(service_dates)


def pair_breakdowns(raw: pd.DataFrame) -> pd.DataFrame:
    """Pair each BREAKDOWN with the train's next RECOVERED event."""
    raw = raw.sort_values(["train_id", "ts", "event_description"]).reset_index(drop=True)
    on = raw.iloc[::2].reset_index(drop=True)
    off = raw.iloc[1::2].reset_index(drop=True)
    assert (on.event_description == "BREAKDOWN").all() and (off.event_description == "RECOVERED").all()
    assert (on.train_id.values == off.train_id.values).all()
    out = pd.DataFrame({
        "train_id": on.train_id, "onset": on.ts, "recovered": off.ts.values, "location": on.station,
        "recovery_location": off.station.values,
    })
    out["duration_min"] = (out.recovered - out.onset).dt.total_seconds() / 60
    return out


bd = pair_breakdowns(bd_raw)
log(f"paired breakdowns={len(bd)}; recovery at same location label: {(bd.location == bd.recovery_location).mean():.3f}")
bd["at_station"] = ~bd.location.str.contains("->")
bd["from_station"] = bd.location.str.split(" -> ").str[0]
bd["to_station"] = bd.location.str.split(" -> ").str[1]
bd["date"] = bd.onset.dt.strftime("%Y-%m-%d")
bd["hour"] = bd.onset.dt.hour
bd["weekday"] = bd.onset.dt.dayofweek
bd["month"] = bd.onset.dt.month
bd = bd.merge(trains, on="train_id", how="left").merge(weather, on="date", how="left")
same_day_clean = cleaning.rename(columns={"completed_at": "cleaned_at"})[
    ["train_id", "date", "cleaned_at", "cleaning_type", "cleaning_team", "cleaning_product_code"]]
bd = bd.merge(same_day_clean, on=["train_id", "date"], how="left")
bd["hours_since_cleaning"] = (bd.onset - bd.cleaned_at).dt.total_seconds() / 3600
assert bd.hours_since_cleaning.gt(0).all(), "breakdown before that day's cleaning"

# Network-wide bursts: sort all breakdowns by onset, chain those within BURST_GAP_MIN on the same date.
bd = bd.sort_values("onset").reset_index(drop=True)
gap_min = bd.onset.diff().dt.total_seconds() / 60
new_chain = ~((bd.date == bd.date.shift()) & (gap_min <= BURST_GAP_MIN))
bd["chain_id"] = new_chain.cumsum()
chain_size = bd.groupby("chain_id").chain_id.transform("size")
bd["in_burst"] = chain_size >= BURST_MIN_SIZE
bd["burst_id"] = np.where(bd.in_burst, bd.chain_id, -1)
log(f"breakdowns in bursts (>= {BURST_MIN_SIZE} events chained by gaps <= {BURST_GAP_MIN} min): "
    f"{bd.in_burst.sum()}/{len(bd)} in {bd.loc[bd.in_burst, 'burst_id'].nunique()} bursts")
bd.to_csv(TAB / "breakdown_events.csv", index=False)

# Per-train table (naive exposure: every train in service on all 105 dates)
per_train = trains.copy()
per_train["n_bd"] = per_train.train_id.map(bd.train_id.value_counts()).fillna(0).astype(int)
per_train["n_bd_burst"] = per_train.train_id.map(bd[bd.in_burst].train_id.value_counts()).fillna(0).astype(int)
per_train["n_bd_isolated"] = per_train.n_bd - per_train.n_bd_burst
per_train["train_days"] = N_SERVICE_DAYS
per_train["log_exposure"] = np.log(per_train.train_days)
per_train.to_csv(TAB / "per_train_counts.csv", index=False)
flush_log()


# ------------------------------------------------------------------------------------------------
# 2. Fleet composition: confounding between train attributes
# ------------------------------------------------------------------------------------------------
section("2. Fleet composition (confounding between type, batch, line/route, age)")
for a, b_ in [("train_type", "batch_number"), ("train_type", "line_id"), ("batch_number", "route_id"),
              ("train_type", "route_id")]:
    ct = pd.crosstab(trains[a], trains[b_])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    cramer_v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    log(f"\n{a} x {b_}: chi2={chi2:.2f} df={dof} p={p:.3f} Cramer V={cramer_v:.2f} (asymptotic; sparse cells)")
    log(ct.to_string())
for f in ["train_type", "batch_number", "route_id"]:
    groups = [g.age_years.values for _, g in trains.groupby(f)]
    h, p = stats.kruskal(*groups)
    log(f"age by {f}: Kruskal H={h:.2f} p={p:.3f}; medians "
        f"{trains.groupby(f).age_years.median().round(1).to_dict()}")
flush_log()


# ------------------------------------------------------------------------------------------------
# 3. Naive per-train breakdown rates by train attributes
# ------------------------------------------------------------------------------------------------
section("3. Naive per-train rates (breakdowns per 1000 train-days; every train assumed in service 105 days)")
pval_registry: list[tuple[str, float]] = []  # (test label, p) for a global BH adjustment


def register(label: str, p: float) -> None:
    pval_registry.append((label, float(p)))


rate_tables = {}
for f in ["train_type", "batch_number", "line_id", "route_id", "age_band"]:
    rt = rate_table(per_train, f)
    rate_tables[f] = rt
    obs, p_perm = label_permutation_p(per_train.n_bd.values, per_train[f].astype(str).values,
                                      per_train.train_days.values)
    register(f"per-train rate by {f} (label permutation)", p_perm)
    log(f"\n{f}: Pearson stat={obs:.2f}, permutation p={p_perm:.4f}")
    log(rt.round(2).to_string())
    # Isolated (non-burst) breakdowns only: are the factor effects the same outside bursts?
    obs_i, p_i = label_permutation_p(per_train.n_bd_isolated.values, per_train[f].astype(str).values,
                                     per_train.train_days.values)
    log(f"  isolated-only events: stat={obs_i:.2f}, permutation p={p_i:.4f}; "
        f"counts {per_train.groupby(f, observed=True).n_bd_isolated.sum().to_dict()}")
    rt.to_csv(TAB / f"rate_by_{f}.csv")

# Line-10 contrast (route 10 + 10B) vs the other lines
per_train["is_line10"] = (per_train.line_id == "10").astype(int)
rr, _ = glm_rate_ratios("n_bd ~ is_line10", per_train)
log(f"\nLine 10 vs other lines: RR={rr.loc['is_line10', 'RR']:.2f} "
    f"[{rr.loc['is_line10', 'lo']:.2f}, {rr.loc['is_line10', 'hi']:.2f}], Wald p={rr.loc['is_line10', 'p']:.2g}")
rr10b, _ = glm_rate_ratios("n_bd ~ C(route_id, Treatment('10'))", per_train[per_train.line_id == "10"])
log(f"Within line 10, route 10B vs 10: RR={rr10b.iloc[1]['RR']:.2f} [{rr10b.iloc[1]['lo']:.2f}, "
    f"{rr10b.iloc[1]['hi']:.2f}], p={rr10b.iloc[1]['p']:.3f}")

# Age as continuous
rr_age, _ = glm_rate_ratios("n_bd ~ age_years", per_train)
log(f"Age slope (per year): RR={rr_age.loc['age_years', 'RR']:.3f} [{rr_age.loc['age_years', 'lo']:.3f}, "
    f"{rr_age.loc['age_years', 'hi']:.3f}], p={rr_age.loc['age_years', 'p']:.3f}")
rho, p_rho = stats.spearmanr(per_train.age_years, per_train.n_bd)
log(f"Spearman(age, count)={rho:.3f} p={p_rho:.3f}")
register("per-train count vs age (Poisson slope)", rr_age.loc["age_years", "p"])

# Multivariable Poisson GLM with LR tests (drop-one)
section("3b. Multivariable per-train Poisson GLM: route + type + batch + age, drop-one LR tests")
full_formula = "n_bd ~ C(route_id) + C(train_type) + C(batch_number) + age_years"
full = smf.glm(full_formula, data=per_train, family=sm.families.Poisson(), offset=per_train.log_exposure).fit()
log(f"full model deviance={full.deviance:.2f} on {full.df_resid:.0f} df; Pearson chi2/df={full.pearson_chi2 / full.df_resid:.3f}")
for term in ["C(route_id)", "C(train_type)", "C(batch_number)", "age_years"]:
    reduced_formula = full_formula.replace(f" + {term}", "").replace(f"{term} + ", "")
    red = smf.glm(reduced_formula, data=per_train, family=sm.families.Poisson(),
                  offset=per_train.log_exposure).fit()
    s, d, p = lr_test(full, red)
    log(f"  drop {term:18s}: LR={s:6.2f} df={d} p={p:.4f}")
    register(f"multivariable GLM drop {term}", p)
ci = full.conf_int()
mv = pd.DataFrame({"RR": np.exp(full.params), "lo": np.exp(ci[0]), "hi": np.exp(ci[1]), "p": full.pvalues})
log(mv.round(3).to_string())
mv.to_csv(TAB / "per_train_multivariable_glm.csv")

# Interaction type x batch (sparse: 24 cells, 2-10 trains each)
section("3c. Interactions among train attributes (sparse; screening only)")
for a, b_ in [("train_type", "batch_number"), ("train_type", "route_id"), ("batch_number", "route_id"),
              ("train_type", "age_years")]:
    ta = f"C({a})" if a != "age_years" else a
    tb = f"C({b_})" if b_ != "age_years" else b_
    add = smf.glm(f"n_bd ~ {ta} + {tb}", data=per_train, family=sm.families.Poisson(),
                  offset=per_train.log_exposure).fit()
    inter = smf.glm(f"n_bd ~ {ta} * {tb}", data=per_train, family=sm.families.Poisson(),
                    offset=per_train.log_exposure).fit()
    s, d, p = lr_test(inter, add)
    # Permutation LR p-value: permute the joint (a, b) label pair across trains keeps margins fixed
    # only approximately, so instead use parametric bootstrap from the additive fit.
    mu = add.fittedvalues.values
    boot = []
    for _ in range(500):
        y = rng.poisson(mu)
        dfb = per_train.assign(n_bd=y)
        try:
            a_b = smf.glm(f"n_bd ~ {ta} + {tb}", data=dfb, family=sm.families.Poisson(),
                          offset=dfb.log_exposure).fit()
            i_b = smf.glm(f"n_bd ~ {ta} * {tb}", data=dfb, family=sm.families.Poisson(),
                          offset=dfb.log_exposure).fit()
            boot.append(2 * (i_b.llf - a_b.llf))
        except Exception:  # separation in a sparse bootstrap draw
            continue
    p_boot = (1 + np.sum(np.array(boot) >= s)) / (len(boot) + 1)
    log(f"{a} x {b_}: LR={s:.2f} df={d} asymptotic p={p:.4f}; parametric-bootstrap p={p_boot:.4f} (B={len(boot)})")
    register(f"interaction {a} x {b_} (bootstrap LR)", p_boot)
tb_counts = per_train.pivot_table(index="train_type", columns="batch_number", values="n_bd", aggfunc="sum")
tb_trains = per_train.pivot_table(index="train_type", columns="batch_number", values="n_bd", aggfunc="size")
tb_rate = tb_counts / (tb_trains * N_SERVICE_DAYS) * 1000
log("\ntype x batch breakdown counts:\n" + tb_counts.to_string())
log("type x batch naive rate per 1000 train-days:\n" + tb_rate.round(1).to_string())

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, f in zip(axes, ["train_type", "batch_number", "route_id"]):
    rt = rate_tables[f]
    ax.errorbar(range(len(rt)), rt.rate, yerr=[rt.rate - rt.rate_lo, rt.rate_hi - rt.rate], fmt="o", capsize=3)
    ax.axhline(per_train.n_bd.sum() / (len(per_train) * N_SERVICE_DAYS) * 1000, color="grey", ls="--", lw=1)
    ax.set_xticks(range(len(rt)), [f"{i}\n(n={n})" for i, n in zip(rt.index, rt.n_trains)])
    ax.set_title(f"by {f}")
    ax.set_ylabel("breakdowns / 1000 train-days (naive)")
savefig(fig, "rates_by_train_attribute.png")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
sns.heatmap(tb_rate, annot=tb_counts.astype(int), fmt="d", cmap="viridis", ax=axes[0],
            cbar_kws={"label": "rate /1000 train-days"})
axes[0].set_title("type x batch (annot = breakdown count)")
sns.stripplot(data=per_train, x="age_years", y="route_id", hue="n_bd", palette="rocket_r", ax=axes[1], size=6)
axes[1].set_title("train age by route, coloured by breakdown count")
savefig(fig, "type_batch_heatmap_age.png")
flush_log()


# ------------------------------------------------------------------------------------------------
# 4. Location: where do breakdowns happen, relative to where trains spend time?
# ------------------------------------------------------------------------------------------------
section("4. Location (exposure proxy = train-seconds per station/segment on 2026-03-02, from a head sample "
        "of normal_operations.csv)")


def poisson_rr_exact(k1, e1, k0, e0, level=0.95):
    """Rate ratio (k1/e1)/(k0/e0) with exact conditional (binomial) CI and two-sided p for RR=1."""
    n = k1 + k0
    bt = stats.binomtest(int(k1), int(n), e1 / (e1 + e0))
    lo_p, hi_p = bt.proportion_ci(confidence_level=level, method="exact")
    to_rr = lambda p: (p / (1 - p)) * (e0 / e1) if p < 1 else np.inf  # noqa: E731
    point = (k1 / e1) / (k0 / e0) if k0 > 0 else np.inf
    return point, to_rr(lo_p), to_rr(hi_p), bt.pvalue


ops = read_csv("normal_operations.csv", nrows=NORMAL_OPS_SAMPLE_ROWS)
ops["ts"] = pd.to_datetime(ops.timestamp)
ops = ops[ops.ts.dt.strftime("%Y-%m-%d") == service_dates[0]].sort_values(["train_id", "ts"])
assert ops.ts.max() < pd.Timestamp(service_dates[1]), "sample does not cover the full first service date"
ops["next_station"] = ops.groupby("train_id").station.shift(-1)
ops["next_event"] = ops.groupby("train_id").event_description.shift(-1)
ops["next_ts"] = ops.groupby("train_id").ts.shift(-1)
ops["secs"] = (ops.next_ts - ops.ts).dt.total_seconds()
dwell = ops[(ops.event_description == "ARRIVED") & (ops.next_event == "DEPARTED")]
runs = ops[ops.event_description.isin(["DEPARTED", "DEPARTED_ORIGIN"]) & (ops.next_event == "ARRIVED")]
exposure_loc = pd.concat([
    dwell.groupby("station").secs.sum().rename("secs").reset_index().rename(columns={"station": "location"}),
    runs.assign(location=runs.station + " -> " + runs.next_station).groupby("location").secs.sum().reset_index(),
], ignore_index=True)
exposure_loc["at_station"] = ~exposure_loc.location.str.contains("->")
log(f"day-1 sample rows={len(ops)}, trains={ops.train_id.nunique()}, dwell mean={dwell.secs.mean():.1f}s, "
    f"run mean={runs.secs.mean():.1f}s; locations with exposure={len(exposure_loc)}")
unmatched = set(bd.location) - set(exposure_loc.location)
log(f"breakdown location labels absent from day-1 exposure: {sorted(unmatched)}")

# Graph distance (hops) from each node to the nearest Line-10 node
adjacency: dict[str, set[str]] = {}
for _, row in stations.iterrows():
    adjacency.setdefault(row.station_id, set())
    if isinstance(row.next_station_id, str):
        adjacency[row.station_id].add(row.next_station_id)
        adjacency.setdefault(row.next_station_id, set()).add(row.station_id)
line10_nodes = set(stations.loc[stations.line_id == "10", "station_id"])
hops_to_line10 = {s: 0 for s in line10_nodes}
frontier = list(line10_nodes)
while frontier:
    nxt = []
    for u in frontier:
        for v in adjacency[u]:
            if v not in hops_to_line10:
                hops_to_line10[v] = hops_to_line10[u] + 1
                nxt.append(v)
    frontier = nxt


def location_hops(loc: str) -> int:
    return min(hops_to_line10[s] for s in loc.split(" -> "))


def corridor_class(h: int) -> str:
    return {0: "0: touches Line-10 node", 1: "1: one hop from Line 10"}.get(h, "2+: two or more hops")


for df_ in (exposure_loc, bd):
    col = "location"
    df_["hops_to_line10"] = df_[col].map(location_hops)
    df_["corridor"] = df_.hops_to_line10.map(corridor_class)

cor = pd.DataFrame({
    "breakdowns": bd.groupby("corridor").size(),
    "burst": bd[bd.in_burst].groupby("corridor").size(),
    "isolated": bd[~bd.in_burst].groupby("corridor").size(),
    "exposure_share": exposure_loc.groupby("corridor").secs.sum() / exposure_loc.secs.sum(),
}).fillna(0)
cor["expected"] = cor.exposure_share * len(bd)
cor["obs_over_exp"] = cor.breakdowns / cor.expected
log("\nBreakdowns by proximity to Line 10 (all lines' trains):\n" + cor.round(3).to_string())
gof = stats.chisquare(cor.breakdowns, cor.expected)
log(f"GOF vs exposure: chi2={gof.statistic:.1f}, p={gof.pvalue:.2g}")
register("location: proximity-to-Line-10 classes vs exposure (GOF)", gof.pvalue)
e_near = exposure_loc.loc[exposure_loc.hops_to_line10 <= 1, "secs"].sum()
e_far = exposure_loc.loc[exposure_loc.hops_to_line10 >= 2, "secs"].sum()
k_near, k_far = (bd.hops_to_line10 <= 1).sum(), (bd.hops_to_line10 >= 2).sum()
rr_, lo_, hi_, p_ = poisson_rr_exact(k_near, e_near, k_far, e_far)
log(f"Within <=1 hop of Line 10 vs >=2 hops: {k_near} vs {k_far} events; RR={rr_:.1f} [{lo_:.1f}, {hi_:.1f}], p={p_:.2g}")
for lab, sub in [("isolated only", bd[~bd.in_burst]), ("burst only", bd[bd.in_burst])]:
    kn, kf = (sub.hops_to_line10 <= 1).sum(), (sub.hops_to_line10 >= 2).sum()
    rr_s = poisson_rr_exact(kn, e_near, kf, e_far)
    log(f"  {lab}: {kn} vs {kf}; RR={rr_s[0]:.1f} [{rr_s[1]:.1f}, {rr_s[2]:.1f}], p={rr_s[3]:.2g}")

# By the breakdown train's own line: where does each line's fleet break down vs where it runs?
runs_line = runs.merge(trains[["train_id", "line_id"]], on="train_id")
dwell_line = dwell.merge(trains[["train_id", "line_id"]], on="train_id")
exp_by_line = pd.concat([
    dwell_line.assign(location=dwell_line.station),
    runs_line.assign(location=runs_line.station + " -> " + runs_line.next_station),
])
exp_by_line["near"] = exp_by_line.location.map(location_hops) <= 1
tab_rows = []
for line, g in bd.groupby("line_id"):
    e = exp_by_line[exp_by_line.line_id == line]
    tab_rows.append({"train_line": line, "breakdowns": len(g), "near_line10": int((g.hops_to_line10 <= 1).sum()),
                     "share_near_obs": (g.hops_to_line10 <= 1).mean(),
                     "share_time_near": e.loc[e.near, "secs"].sum() / e.secs.sum()})
by_line_loc = pd.DataFrame(tab_rows).set_index("train_line")
log("\nBy train's own line: share of breakdowns vs share of running time within 1 hop of Line 10\n"
    + by_line_loc.round(3).to_string())

# Line-level rates corrected by day-1 time-in-service share (fleet-hours per line)
fleet_secs = exp_by_line.groupby("line_id").secs.sum()
line_rates = pd.DataFrame({"breakdowns": bd.groupby("line_id").size(), "day1_train_hours": fleet_secs / 3600})
line_rates["per_1000_day1_train_hours_x105"] = line_rates.breakdowns / (line_rates.day1_train_hours * N_SERVICE_DAYS) * 1000
log("\nPer-line breakdowns per 1000 train-hours (day-1 hours scaled x105; rough):\n" + line_rates.round(3).to_string())

# At station vs between stations
t_station = exposure_loc.loc[exposure_loc.at_station, "secs"].sum() / exposure_loc.secs.sum()
k_st = int(bd.at_station.sum())
bt = stats.binomtest(k_st, len(bd), t_station)
ci_st = bt.proportion_ci(method="exact")
log(f"\nAt-station breakdowns {k_st}/{len(bd)} = {k_st / len(bd):.3f} [{ci_st.low:.3f}, {ci_st.high:.3f}] "
    f"vs share of train time dwelling at stations {t_station:.3f}; binomial p={bt.pvalue:.3g}")
near_exp = exposure_loc[exposure_loc.hops_to_line10 <= 1]
t_station_near = near_exp.loc[near_exp.at_station, "secs"].sum() / near_exp.secs.sum()
sub = bd[bd.hops_to_line10 <= 1]
bt2 = stats.binomtest(int(sub.at_station.sum()), len(sub), t_station_near)
log(f"  restricted to <=1 hop of Line 10: {sub.at_station.sum()}/{len(sub)} vs time share {t_station_near:.3f}, p={bt2.pvalue:.3g}")
register("at-station vs between-station share vs dwell-time share", bt.pvalue)

# Per-node table (each breakdown credited to its from-station; segments counted once)
node_rows = []
for node in sorted(adjacency):
    e = exposure_loc[exposure_loc.location.str.split(" -> ").str[0] == node].secs.sum()
    k = int((bd.from_station == node).sum())
    node_rows.append({"station_id": node, "name": station_name.get(node, ""), "lines": station_lines.get(node, ""),
                      "hops_to_line10": hops_to_line10[node], "breakdowns_from_node": k, "exposure_secs": e})
nodes_df = pd.DataFrame(node_rows)
nodes_df["expected"] = nodes_df.exposure_secs / nodes_df.exposure_secs.sum() * len(bd)
nodes_df["p_upper"] = stats.poisson.sf(nodes_df.breakdowns_from_node - 1, nodes_df.expected)
nodes_df["q_bh"] = multipletests(nodes_df.p_upper, method="fdr_bh")[1]
nodes_df = nodes_df.sort_values("p_upper")
nodes_df.to_csv(TAB / "breakdowns_by_station_node.csv", index=False)
log("\nTop station nodes (breakdowns whose location starts at the node; Poisson upper-tail vs exposure, BH q):\n"
    + nodes_df.head(15).round(4).to_string(index=False))
inside = nodes_df[nodes_df.hops_to_line10 == 0]
gof_in = stats.chisquare(inside.breakdowns_from_node,
                         inside.exposure_secs / inside.exposure_secs.sum() * inside.breakdowns_from_node.sum())
log(f"Heterogeneity among Line-10 nodes only (GOF vs exposure, {len(inside)} nodes): chi2={gof_in.statistic:.1f}, "
    f"p={gof_in.pvalue:.3g} (small expected counts; indicative)")

# Figure: route strip charts of breakdown counts per node
fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharey=True)
for ax, (route, g) in zip(axes, stations.assign(seq=stations.sequence.astype(int)).sort_values("seq").groupby("route_id")):
    touch = pd.concat([bd.from_station, bd.to_station.dropna()]).value_counts()
    vals = g.station_id.map(touch).fillna(0).values
    colors = ["tab:red" if hops_to_line10[s] == 0 else ("tab:orange" if hops_to_line10[s] == 1 else "tab:grey")
              for s in g.station_id]
    ax.bar(range(len(g)), vals, color=colors)
    ax.set_xticks(range(len(g)), g.station_id, rotation=90, fontsize=6)
    ax.set_title(f"route {route}: breakdowns touching each node (red=Line-10 node, orange=1 hop, grey=2+)", fontsize=8)
savefig(fig, "location_by_route_node.png")
flush_log()

# Within the Line-10 corridor (<=1 hop), compare the trains of each line per corridor train-hour
near_hours = exp_by_line[exp_by_line.near].groupby("line_id").secs.sum() / 3600 * N_SERVICE_DAYS
k_line_near = bd[bd.hops_to_line10 <= 1].groupby("line_id").size()
zone = pd.DataFrame({"breakdowns_in_corridor": k_line_near, "corridor_train_hours_est": near_hours})
zone["rate_per_1000h"] = zone.breakdowns_in_corridor / zone.corridor_train_hours_est * 1000
lo, hi = poisson_ci(zone.breakdowns_in_corridor.values, zone.corridor_train_hours_est.values)
zone["lo"], zone["hi"] = lo * 1000, hi * 1000
log("\nWithin-corridor rates by the train's own line (corridor exposure from day-1 x105):\n" + zone.round(3).to_string())
k10, e10 = zone.loc["10", "breakdowns_in_corridor"], zone.loc["10", "corridor_train_hours_est"]
ko, eo = zone.drop("10").breakdowns_in_corridor.sum(), zone.drop("10").corridor_train_hours_est.sum()
rr_z = poisson_rr_exact(k10, e10, ko, eo)
log(f"Line-10 trains vs other lines' trains, inside corridor: RR={rr_z[0]:.2f} [{rr_z[1]:.2f}, {rr_z[2]:.2f}], p={rr_z[3]:.3f}")
gof_z = stats.chisquare(zone.breakdowns_in_corridor,
                        zone.corridor_train_hours_est / zone.corridor_train_hours_est.sum() * zone.breakdowns_in_corridor.sum())
log(f"Equal within-corridor rate across the 4 lines' fleets: chi2={gof_z.statistic:.2f}, p={gof_z.pvalue:.3f}")
register("within-corridor rate differs by train line", gof_z.pvalue)
zone.to_csv(TAB / "within_corridor_rate_by_line.csv")
flush_log()


# ------------------------------------------------------------------------------------------------
# 5. Temporal structure: bursts, date, weekday, month, hour
# ------------------------------------------------------------------------------------------------
section("5. Temporal structure")
daily = weather.copy()
daily["n_bd"] = daily.date.map(bd.groupby("date").size()).fillna(0).astype(int)
daily["n_isolated"] = daily.date.map(bd[~bd.in_burst].groupby("date").size()).fillna(0).astype(int)
daily["n_bursts"] = daily.date.map(bd[bd.in_burst].groupby("date").burst_id.nunique()).fillna(0).astype(int)
daily["n_in_bursts"] = daily.n_bd - daily.n_isolated
daily["dt"] = pd.to_datetime(daily.date)
daily["weekday"] = daily.dt.dt.dayofweek
daily["month"] = daily.dt.dt.month
daily["day_index"] = np.arange(len(daily))
log(f"daily counts: mean={daily.n_bd.mean():.3f}, var={daily.n_bd.var():.3f}, "
    f"index of dispersion={daily.n_bd.var() / daily.n_bd.mean():.2f}; days with 0: {(daily.n_bd == 0).sum()}")
log(f"isolated daily: mean={daily.n_isolated.mean():.3f}, var={daily.n_isolated.var():.3f} (Poisson-like?)")
log(f"bursts per day: mean={daily.n_bursts.mean():.3f}, var={daily.n_bursts.var():.3f}; "
    f"burst-days={int((daily.n_bursts > 0).sum())}")
log("top days:\n" + daily.sort_values("n_bd", ascending=False).head(12)[
    ["date", "temp_c", "rain_mm", "weekday", "n_bd", "n_bursts", "n_isolated"]].to_string(index=False))

bursts = bd[bd.in_burst].groupby("burst_id").agg(
    date=("date", "first"), start=("onset", "min"), end=("onset", "max"), size=("train_id", "size"),
    n_trains=("train_id", "nunique"), n_lines=("line_id", "nunique"), n_routes=("route_id", "nunique"),
    n_types=("train_type", "nunique"), n_batches=("batch_number", "nunique"))
bursts["span_min"] = (bursts.end - bursts.start).dt.total_seconds() / 60
bursts["start_hour"] = bursts.start.dt.hour
bursts = bursts.merge(weather, on="date", how="left")
bursts.to_csv(TAB / "bursts.csv", index=False)
log("\nBursts:\n" + bursts.drop(columns=["end"]).to_string(index=False))

# Null for burst structure: events placed uniformly at random over service time (05:30-23:30) of all
# service dates (homogeneous Poisson process given total). Compare the share of breakdowns in bursts.
def burst_share(onsets_min_by_day: list[np.ndarray]) -> float:
    in_b = 0
    total = 0
    for x in onsets_min_by_day:
        x = np.sort(x)
        total += len(x)
        if len(x) == 0:
            continue
        chain = np.concatenate([[0], np.cumsum(np.diff(x) > BURST_GAP_MIN)])
        sizes = np.bincount(chain)
        in_b += sizes[sizes >= BURST_MIN_SIZE].sum()
    return in_b / total


obs_share = bd.in_burst.mean()
sim = []
for _ in range(2000):
    days = rng.integers(0, N_SERVICE_DAYS, len(bd))
    mins = rng.uniform(5.5 * 60, 23.5 * 60, len(bd))
    sim.append(burst_share([mins[days == d] for d in range(N_SERVICE_DAYS)]))
sim = np.array(sim)
log(f"\nShare of breakdowns in bursts: observed {obs_share:.3f}; homogeneous-Poisson null mean {sim.mean():.3f}, "
    f"95th pct {np.quantile(sim, 0.95):.3f}, p={(1 + (sim >= obs_share).sum()) / (len(sim) + 1):.4f}")
# Conditional on each day's count: are events clustered in time within the day?
sim_c = []
day_counts = daily.n_bd.values
for _ in range(2000):
    sim_c.append(burst_share([rng.uniform(5.5 * 60, 23.5 * 60, k) for k in day_counts]))
sim_c = np.array(sim_c)
log(f"  conditional on daily counts: null mean {sim_c.mean():.3f}, p={(1 + (sim_c >= obs_share).sum()) / (len(sim_c) + 1):.4f}")
register("temporal clustering (burst share vs homogeneous Poisson)", (1 + (sim >= obs_share).sum()) / (len(sim) + 1))

# Sensitivity of the burst definition
for gap_m, min_size in [(15, 3), (30, 3), (60, 3), (30, 2)]:
    x_by_day = [(g_.onset.dt.hour * 60 + g_.onset.dt.minute + g_.onset.dt.second / 60).sort_values().values
                for _, g_ in bd.groupby("date")]
    n_in, n_b = 0, 0
    for x in x_by_day:
        chain = np.concatenate([[0], np.cumsum(np.diff(x) > gap_m)])
        sizes = np.bincount(chain)
        n_in += sizes[sizes >= min_size].sum()
        n_b += (sizes >= min_size).sum()
    log(f"burst definition gap<={gap_m} min, size>={min_size}: {n_b} bursts containing {n_in}/{len(bd)} breakdowns")

# Inter-event gaps within a day
within = bd.sort_values("onset")
gaps = within.groupby("date").onset.diff().dt.total_seconds().dropna() / 60
log(f"within-day gaps between consecutive network breakdowns (min): median={gaps.median():.1f}, "
    f"share<=15min={(gaps <= 15).mean():.2f}")

# Weekday / month: day-level Poisson GLMs (counts of bursts and of all breakdowns); permutation over days
for outcome in ["n_bd", "n_bursts", "n_isolated"]:
    for f in ["weekday", "month"]:
        m1 = smf.glm(f"{outcome} ~ C({f})", data=daily, family=sm.families.Poisson()).fit()
        m0 = smf.glm(f"{outcome} ~ 1", data=daily, family=sm.families.Poisson()).fit()
        lr_obs = 2 * (m1.llf - m0.llf)
        perm = []
        for _ in range(2000):
            dfp = daily.assign(**{f: rng.permutation(daily[f].values)})
            perm.append(2 * (smf.glm(f"{outcome} ~ C({f})", data=dfp, family=sm.families.Poisson()).fit().llf - m0.llf))
        p_perm = (1 + np.sum(np.array(perm) >= lr_obs)) / (len(perm) + 1)
        means = daily.groupby(f)[outcome].mean().round(2).to_dict()
        log(f"{outcome} by {f}: LR={lr_obs:.2f} df={len(means) - 1}, asymptotic p={stats.chi2.sf(lr_obs, len(means) - 1):.4f}, "
            f"day-permutation p={p_perm:.4f}; means {means}")
        if outcome in ("n_bursts", "n_isolated"):
            register(f"{outcome} by {f} (day permutation)", p_perm)

# Hour of day with exposure: active trains per hour on day 1 (proxy)
ops_hour = ops.assign(hour=ops.ts.dt.hour).groupby("hour").train_id.nunique()
hourly = pd.DataFrame({"breakdowns": bd.groupby("hour").size(), "isolated": bd[~bd.in_burst].groupby("hour").size(),
                       "burst_starts": bursts.groupby("start_hour").size(), "active_trains_day1": ops_hour}).fillna(0)
hourly = hourly.loc[5:23]
hourly["expected"] = hourly.active_trains_day1 / hourly.active_trains_day1.sum() * len(bd)
log("\nHour of day (expected = proportional to active trains in that hour on day 1):\n" + hourly.round(2).to_string())
mask = hourly.expected > 0
g_h = stats.chisquare(hourly.breakdowns[mask], hourly.expected[mask] * hourly.breakdowns.sum() / hourly.expected[mask].sum())
exp_iso = hourly.active_trains_day1 / hourly.active_trains_day1.sum() * hourly.isolated.sum()
g_hi = stats.chisquare(hourly.isolated[mask], exp_iso[mask] * hourly.isolated[mask].sum() / exp_iso[mask].sum())
log(f"hour GOF all events chi2={g_h.statistic:.1f} p={g_h.pvalue:.3g} (invalid as independent: bursts); "
    f"isolated-only chi2={g_hi.statistic:.1f} p={g_hi.pvalue:.3g} (tiny expected counts)")
# Burst start hours vs uniform over 06-22
bs = bursts.start_hour.values
ks = stats.kstest((bursts.start.dt.hour * 60 + bursts.start.dt.minute - 5.5 * 60) / (18 * 60), "uniform")
log(f"burst start times vs uniform 05:30-23:30: KS D={ks.statistic:.3f}, p={ks.pvalue:.3f}")
register("burst start time-of-day vs uniform (KS)", ks.pvalue)

fig, axes = plt.subplots(3, 1, figsize=(12, 9))
ax = axes[0]
ax.bar(daily.dt, daily.n_isolated, color="tab:blue", label="isolated")
ax.bar(daily.dt, daily.n_in_bursts, bottom=daily.n_isolated, color="tab:red", label="in burst")
ax.set_ylabel("breakdowns/day")
ax2 = ax.twinx()
ax2.plot(daily.dt, daily.temp_c, color="tab:orange", lw=1, label="mean temp (C)")
ax2.set_ylabel("temp C")
ax.legend(loc="upper left")
ax.set_title("daily breakdowns (stacked: isolated / in network-wide bursts) with temperature")
ax = axes[1]
ax.bar(daily.dt, daily.rain_mm, color="tab:cyan")
ax.set_ylabel("rain mm")
for d in bursts.date.unique():
    ax.axvline(pd.Timestamp(d), color="tab:red", alpha=0.3)
ax.set_title("daily rainfall; red lines = burst days")
ax = axes[2]
ax.bar(hourly.index - 0.2, hourly.breakdowns, width=0.4, label="observed")
ax.bar(hourly.index + 0.2, hourly.expected, width=0.4, label="expected by active trains")
ax.set_xlabel("hour of day")
ax.legend()
savefig(fig, "temporal_daily_hourly.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
sns.barplot(data=daily, x="weekday", y="n_bursts", ax=axes[0], errorbar=("ci", 95))
axes[0].set_title("bursts per day by weekday (0=Mon)")
sns.barplot(data=daily, x="month", y="n_bursts", ax=axes[1], errorbar=("ci", 95))
axes[1].set_title("bursts per day by month")
savefig(fig, "weekday_month_bursts.png")
flush_log()


# ------------------------------------------------------------------------------------------------
# 6. Weather: day-level counts vs temperature and rainfall (incl. lags, thresholds, anomalies)
# ------------------------------------------------------------------------------------------------
section("6. Weather vs day-level counts")
daily["temp_anom"] = daily.temp_c - daily.temp_c.rolling(15, center=True, min_periods=5).mean()
daily["temp_change"] = daily.temp_c.diff()
daily["rain_lag1"] = daily.rain_mm.shift(1)
daily["temp_lag1"] = daily.temp_c.shift(1)
daily["rain_3d"] = daily.rain_mm.rolling(3, min_periods=1).sum()
daily["burst_day"] = (daily.n_bursts > 0).astype(int)
weather_vars = ["temp_c", "rain_mm", "temp_anom", "temp_change", "rain_lag1", "temp_lag1", "rain_3d"]


def circular_shift_p(x: np.ndarray, y: np.ndarray, n=N_PERM) -> float:
    """Spearman correlation p-value preserving autocorrelation of both series (circular shifts of x)."""
    ok = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[ok], y[ok]
    obs = abs(stats.spearmanr(x, y).statistic)
    shifts = rng.integers(5, len(x) - 5, n)
    null = np.array([abs(stats.spearmanr(np.roll(x, s), y).statistic) for s in shifts])
    return (1 + np.sum(null >= obs)) / (n + 1)


rows = []
for outcome in ["n_bd", "n_bursts", "n_isolated", "burst_day"]:
    for v in weather_vars:
        x, yv = daily[v].values.astype(float), daily[outcome].values.astype(float)
        ok = ~np.isnan(x)
        rho = stats.spearmanr(x[ok], yv[ok]).statistic
        rows.append({"outcome": outcome, "weather": v, "spearman": rho, "p_circular_shift": circular_shift_p(x, yv)})
wtab = pd.DataFrame(rows)
log(wtab.round(3).to_string(index=False))
wtab.to_csv(TAB / "weather_day_level_correlations.csv", index=False)
for _, rw in wtab[wtab.outcome.isin(["n_bursts", "n_isolated"])].iterrows():
    register(f"weather: {rw.outcome} vs {rw.weather} (Spearman, circular shift)", rw.p_circular_shift)

# Threshold contrasts (day-level rate ratio of bursts and of isolated events) with exact CIs
log("\nThreshold contrasts (days above vs at/below threshold); exact conditional RR on day counts:")
thr_rows = []
for v, thresholds in [("rain_mm", [0.0, 5.0, 10.0, 20.0, 30.0]), ("temp_c", [25.0, 28.0, 30.0, 32.0]),
                      ("temp_anom", [1.0, 2.0]), ("rain_lag1", [10.0])]:
    for t in thresholds:
        hi_days = daily[v] > t
        lo_days = (daily[v] <= t)
        for outcome in ["n_bursts", "n_isolated", "n_bd"]:
            k1, k0 = daily.loc[hi_days, outcome].sum(), daily.loc[lo_days, outcome].sum()
            e1, e0 = hi_days.sum(), lo_days.sum()
            if k1 + k0 == 0 or e1 == 0:
                continue
            rr_t = poisson_rr_exact(k1, e1, k0, e0)
            thr_rows.append({"var": v, "threshold": t, "outcome": outcome, "days_above": int(e1),
                             "events_above": int(k1), "events_below": int(k0), "RR": rr_t[0], "lo": rr_t[1],
                             "hi": rr_t[2], "p": rr_t[3]})
thr = pd.DataFrame(thr_rows)
log(thr.round(3).to_string(index=False))
log("note: n_bd p-values ignore burst clustering (anti-conservative); use n_bursts / n_isolated for inference.")
thr.to_csv(TAB / "weather_threshold_contrasts.csv", index=False)

# Day-level Poisson / logistic models for bursts with smooth-ish temperature and rainfall
for formula in ["n_bursts ~ temp_c + I(temp_c**2) + np.log1p(rain_mm)",
                "n_bursts ~ temp_anom + np.log1p(rain_mm) + np.log1p(rain_lag1)",
                "n_isolated ~ temp_c + I(temp_c**2) + np.log1p(rain_mm)"]:
    m = smf.glm(formula, data=daily.dropna(subset=["rain_lag1"]) if "lag1" in formula else daily,
                family=sm.families.Poisson()).fit()
    m0 = smf.glm(formula.split("~")[0] + "~ 1", data=m.model.data.frame, family=sm.families.Poisson()).fit()
    s, d, p = lr_test(m, m0)
    log(f"\n{formula}: LR={s:.2f} df={d} p={p:.3f}")
    log(pd.DataFrame({"coef": m.params, "p": m.pvalues}).round(4).to_string())
# Negative binomial for the total daily count (overdispersed by bursts)
nb = smf.glm("n_bd ~ temp_c + I(temp_c**2) + np.log1p(rain_mm)", data=daily,
             family=sm.families.NegativeBinomial(alpha=3.0)).fit()
log("\nNB(alpha=3) total daily count: " + pd.DataFrame({"coef": nb.params, "p": nb.pvalues}).round(4).to_string())

# Burst size vs weather (given a burst happened)
for v in ["temp_c", "rain_mm", "temp_anom"]:
    rho_b = stats.spearmanr(bursts[v], bursts["size"]) if v in bursts else None
    if v == "temp_anom":
        bursts["temp_anom"] = bursts.date.map(daily.set_index("date").temp_anom)
        rho_b = stats.spearmanr(bursts.temp_anom, bursts["size"], nan_policy="omit")
    log(f"burst size vs {v}: Spearman {rho_b.statistic:.3f}, p={rho_b.pvalue:.3f} (n={len(bursts)})")
    register(f"burst size vs {v}", rho_b.pvalue)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, v in zip(axes, ["temp_c", "rain_mm", "temp_anom"]):
    jitter = rng.uniform(-0.15, 0.15, len(daily))
    sc = ax.scatter(daily[v], daily.n_bd + jitter, c=daily.n_bursts, cmap="coolwarm", s=18)
    ax.set_xlabel(v)
    ax.set_ylabel("breakdowns/day")
    if v == "rain_mm":
        ax.set_xscale("symlog", linthresh=1)
fig.colorbar(sc, ax=axes[-1], label="bursts that day")
axes[0].set_title("daily breakdowns vs weather (colour = bursts)")
savefig(fig, "weather_scatter.png")
flush_log()


# ------------------------------------------------------------------------------------------------
# 7. Train-day panel: cleaning, product switching, train attributes, weather interactions
# ------------------------------------------------------------------------------------------------
section("7. Train-day panel (14,700 train-days; outcome = breakdowns that day)")
panel = cleaning.sort_values(["train_id", "completed_at"]).copy()
g = panel.groupby("train_id")
panel["prev_product"] = g.cleaning_product_code.shift(1)
panel["prev_team"] = g.cleaning_team.shift(1)
panel["prev_type"] = g.cleaning_type.shift(1)
panel["product_switch"] = (panel.cleaning_product_code != panel.prev_product).astype(float)
panel.loc[panel.prev_product.isna(), "product_switch"] = np.nan
panel["team_switch"] = (panel.cleaning_team != panel.prev_team).astype(float)
panel.loc[panel.prev_team.isna(), "team_switch"] = np.nan
panel["is_deep"] = (panel.cleaning_type == "DEEP").astype(int)
panel["deep_epoch"] = g.is_deep.cumsum()
panel["day_idx"] = g.cumcount()
epoch_start = panel[panel.is_deep == 1].groupby(["train_id", "deep_epoch"]).day_idx.min()
panel["days_since_deep"] = panel.day_idx - panel.set_index(["train_id", "deep_epoch"]).index.map(epoch_start).values
panel.loc[panel.deep_epoch == 0, "days_since_deep"] = np.nan
panel["dsd_band"] = pd.cut(panel.days_since_deep, [-1, 0, 2, 5, 10, 200], labels=["0", "1-2", "3-5", "6-10", ">10"])
run_id = (panel.cleaning_product_code != panel.prev_product).cumsum()
panel["same_product_run"] = panel.groupby(["train_id", run_id]).cumcount() + 1
panel["clean_hour"] = panel.completed_at.dt.hour + panel.completed_at.dt.minute / 60
y_all = bd.groupby(["train_id", "date"]).size().rename("y")
y_burst = bd[bd.in_burst].groupby(["train_id", "date"]).size().rename("y_burst")
panel = panel.join(y_all, on=["train_id", "date"]).join(y_burst, on=["train_id", "date"]).fillna({"y": 0, "y_burst": 0})
panel["y_isolated"] = panel.y - panel.y_burst
panel = panel.merge(trains, on="train_id", how="left").merge(daily[["date", "temp_c", "rain_mm", "temp_anom",
                                                                     "n_bursts", "weekday", "month"]], on="date")
panel["hot"] = (panel.temp_c > HOT_C).astype(int)
panel["cold"] = (panel.temp_c <= COLD_C).astype(int)
panel["heavy_rain"] = (panel.rain_mm > HEAVY_RAIN_MM).astype(int)
panel["burst_day"] = (panel.n_bursts > 0).astype(int)
assert panel.y.sum() == len(bd)
panel.to_csv(TAB / "train_day_panel.csv", index=False)
log(f"panel rows={len(panel)}, breakdowns={int(panel.y.sum())}; train-days with >1 breakdown: {(panel.y > 1).sum()}")

log("\nOne-way rates per 1000 train-days (naive; no adjustment):")
for f in ["cleaning_product_code", "cleaning_team", "cleaning_type", "prev_product", "prev_team", "prev_type",
          "product_switch", "team_switch", "dsd_band"]:
    t = panel.groupby(f, observed=True).agg(train_days=("y", "size"), events=("y", "sum"),
                                            burst_events=("y_burst", "sum"), isolated=("y_isolated", "sum"))
    t["rate"] = t.events / t.train_days * 1000
    lo, hi = poisson_ci(t.events.values, t.train_days.values)
    t["lo"], t["hi"] = lo * 1000, hi * 1000
    log(f"\n{f}:\n" + t.round(2).to_string())

trans = panel.dropna(subset=["prev_product"]).pivot_table(index="prev_product", columns="cleaning_product_code",
                                                          values="y", aggfunc=["sum", "size"])
trans_rate = trans["sum"] / trans["size"] * 1000
log("\nproduct transition (rows = previous day, cols = today): events\n" + trans["sum"].astype(int).to_string()
    + "\nrate per 1000 train-days\n" + trans_rate.round(1).to_string())
log("P(product switch) = " + f"{panel.product_switch.mean():.3f} (2/3 expected if products iid uniform)")
log("product x line share:\n" + pd.crosstab(panel.line_id, panel.cleaning_product_code, normalize="index").round(3).to_string())
log("team x line share:\n" + pd.crosstab(panel.line_id, panel.cleaning_team, normalize="index").round(3).to_string())
log("product x month share:\n" + pd.crosstab(panel.month, panel.cleaning_product_code, normalize="index").round(3).to_string())
for a, b_ in [("cleaning_product_code", "cleaning_team"), ("cleaning_product_code", "line_id"),
              ("cleaning_team", "line_id"), ("cleaning_product_code", "month"), ("cleaning_type", "cleaning_team"),
              ("cleaning_product_code", "train_type")]:
    ct = pd.crosstab(panel[a], panel[b_])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    log(f"confounding check {a} x {b_}: chi2={chi2:.1f} df={dof} p={p:.3f}")

# Conditional (date fixed-effect) Poisson models. Dates with no breakdown carry no information in the
# conditional likelihood, so we fit on breakdown dates only (avoids infinite date effects).
event_dates = set(bd.date)
pe = panel[panel.date.isin(event_dates) & panel.prev_product.notna()].copy()
log(f"\nConditional models on {pe.date.nunique()} breakdown dates, {len(pe)} train-days:")
base_terms = "C(date) + C(route_id)"
cond_specs = {
    "cleaning product (today)": "C(cleaning_product_code)",
    "cleaning team (today)": "C(cleaning_team)",
    "cleaning type (today)": "C(cleaning_type)",
    "previous-day product": "C(prev_product)",
    "previous-day team": "C(prev_team)",
    "previous-day type": "C(prev_type)",
    "product switch": "product_switch",
    "team switch": "team_switch",
    "days since deep (band)": "C(dsd_band)",
    "train type": "C(train_type)",
    "batch": "C(batch_number)",
    "age (years)": "age_years",
}
pe_dsd = pe.dropna(subset=["dsd_band"])
cond_rows = []
for label, term in cond_specs.items():
    data_ = pe_dsd if "dsd" in term else pe
    m1 = smf.glm(f"y ~ {term} + {base_terms}", data=data_, family=sm.families.Poisson()).fit()
    m0 = smf.glm(f"y ~ {base_terms}", data=data_, family=sm.families.Poisson()).fit()
    s, d, p = lr_test(m1, m0)
    eff = {k: np.exp(v) for k, v in m1.params.items() if k.startswith(term.split("(")[-1].rstrip(")")) or
           k.startswith(term)}
    cis = m1.conf_int()
    eff_str = "; ".join(f"{k.replace(term, '').strip('[]T.') or k}: RR {v:.2f} [{np.exp(cis.loc[k, 0]):.2f}, "
                        f"{np.exp(cis.loc[k, 1]):.2f}]" for k, v in eff.items())
    cond_rows.append({"factor": label, "LR": s, "df": d, "p": p, "effects": eff_str})
    register(f"train-day conditional: {label}", p)
cond = pd.DataFrame(cond_rows)
log(cond.round(4).to_string(index=False))
cond.to_csv(TAB / "train_day_conditional_tests.csv", index=False)

# Joint model: product switch adjusted for everything else
joint = smf.glm("y ~ product_switch + C(cleaning_product_code) + C(prev_product) + C(cleaning_team) + "
                "C(cleaning_type) + C(train_type) + C(batch_number) + age_years + " + base_terms,
                data=pe, family=sm.families.Poisson()).fit()
ci_j = joint.conf_int()
log(f"\nJoint conditional model: product_switch RR={np.exp(joint.params['product_switch']):.2f} "
    f"[{np.exp(ci_j.loc['product_switch', 0]):.2f}, {np.exp(ci_j.loc['product_switch', 1]):.2f}], "
    f"p={joint.pvalues['product_switch']:.4f}")
for outcome in ["y_burst", "y_isolated"]:
    pe_o = pe[pe.date.isin(set(bd[bd.in_burst == (outcome == "y_burst")].date))]
    m = smf.glm(f"{outcome} ~ product_switch + {base_terms}", data=pe_o, family=sm.families.Poisson()).fit()
    c = m.conf_int().loc["product_switch"]
    log(f"  {outcome}: product_switch RR={np.exp(m.params['product_switch']):.2f} [{np.exp(c[0]):.2f}, "
        f"{np.exp(c[1]):.2f}], p={m.pvalues['product_switch']:.4f}")
for line, sub in pe.groupby("line_id"):
    m = smf.glm("y ~ product_switch + C(date)", data=sub[sub.date.isin(set(sub.loc[sub.y > 0, "date"]))],
                family=sm.families.Poisson()).fit()
    c = m.conf_int().loc["product_switch"]
    log(f"  line {line}: product_switch RR={np.exp(m.params['product_switch']):.2f} [{np.exp(c[0]):.2f}, "
        f"{np.exp(c[1]):.2f}]")
# Train-cluster permutation of the switch indicator within train (keeps each train's switching rate)
obs_k_same = pe.loc[pe.product_switch == 0, "y"].sum()
perm_k = []
for _ in range(2000):
    sh = pe.groupby("train_id").product_switch.transform(lambda s: rng.permutation(s.values))
    perm_k.append(pe.loc[sh == 0, "y"].sum())
perm_k = np.array(perm_k)
p_perm_switch = (1 + np.sum(perm_k <= obs_k_same)) / (len(perm_k) + 1)
log(f"within-train permutation of switch flag: breakdowns on same-product days observed={obs_k_same:.0f}, "
    f"null mean={perm_k.mean():.1f}, one-sided p={p_perm_switch:.4f}")
register("product switch (within-train permutation, one-sided)", p_perm_switch)
flush_log()


# ------------------------------------------------------------------------------------------------
# 8. Time since cleaning at breakdown vs exposure-weighted random times; early window by product/team
# ------------------------------------------------------------------------------------------------
section("8. Hours since the most recent cleaning")
# Null: for each breakdown, a random moment when the same train was in service on day 1 (time of day),
# placed on the breakdown's date, minus that morning's cleaning time. This reproduces the operating-hours
# exposure; under "cleaning has no time-dependent effect" observed and null distributions should match.
train_times = ops.groupby("train_id").ts.apply(lambda s: (s.dt.hour * 3600 + s.dt.minute * 60 + s.dt.second).values)
null_hours = []
for _, rw in bd.iterrows():
    secs = rng.choice(train_times[rw.train_id], 200)
    clean_secs = rw.cleaned_at.hour * 3600 + rw.cleaned_at.minute * 60 + rw.cleaned_at.second
    null_hours.append((secs - clean_secs) / 3600)
null_hours = np.concatenate(null_hours)
log(f"hours since cleaning at breakdown: median={bd.hours_since_cleaning.median():.2f}, "
    f"IQR=({bd.hours_since_cleaning.quantile(.25):.2f}, {bd.hours_since_cleaning.quantile(.75):.2f}); "
    f"null median={np.median(null_hours):.2f}")
ks_all = stats.ks_2samp(bd.hours_since_cleaning, null_hours)
iso = bd[~bd.in_burst]
ks_iso = stats.ks_2samp(iso.hours_since_cleaning, null_hours)
log(f"KS observed vs exposure null: all D={ks_all.statistic:.3f} p={ks_all.pvalue:.3f} (events not independent); "
    f"isolated only (n={len(iso)}) D={ks_iso.statistic:.3f} p={ks_iso.pvalue:.3f}")
register("hours since cleaning vs exposure null (isolated, KS)", ks_iso.pvalue)
for window in [4, 6, 8]:
    k = int((bd.hours_since_cleaning <= window).sum())
    p0 = (null_hours <= window).mean()
    bt = stats.binomtest(k, len(bd), p0)
    log(f"  breakdowns within {window} h of cleaning: {k}/{len(bd)} = {k / len(bd):.3f} vs null {p0:.3f}, p={bt.pvalue:.3f}")

# Does the timing distribution differ by product/team/type (e.g. a product that fails early)?
for f in ["cleaning_product_code", "cleaning_team", "cleaning_type"]:
    grp = [g_.hours_since_cleaning.values for _, g_ in bd.groupby(f)]
    h, p = stats.kruskal(*grp)
    med = bd.groupby(f).hours_since_cleaning.median().round(2).to_dict()
    log(f"hours-since-cleaning by {f}: Kruskal H={h:.2f} p={p:.3f}; medians {med}")
    register(f"hours since cleaning differs by {f} (Kruskal)", p)
    # Early-window (<= 6 h) risk per 1000 train-days by group, vs later in the day
    t = panel.groupby(f).size().rename("train_days").to_frame()
    t["early_events"] = bd[bd.hours_since_cleaning <= 6].groupby(f).size()
    t["late_events"] = bd[bd.hours_since_cleaning > 6].groupby(f).size()
    t = t.fillna(0)
    t["early_rate"] = t.early_events / t.train_days * 1000
    t["late_rate"] = t.late_events / t.train_days * 1000
    chi = stats.chi2_contingency(t[["early_events", "late_events"]].values)
    log(f"  early (<=6h) vs late events by {f}: chi2={chi[0]:.2f} p={chi[1]:.3f}\n" + t.round(2).to_string())
    register(f"early vs late share by {f} (chi-square)", chi[1])
bd_sw = bd.merge(panel[["train_id", "date", "product_switch", "prev_product"]], on=["train_id", "date"], how="left")
h_sw = stats.mannwhitneyu(bd_sw.loc[bd_sw.product_switch == 1, "hours_since_cleaning"],
                          bd_sw.loc[bd_sw.product_switch == 0, "hours_since_cleaning"])
log(f"hours since cleaning, switch vs same product: medians "
    f"{bd_sw.groupby('product_switch').hours_since_cleaning.median().round(2).to_dict()}, MWU p={h_sw.pvalue:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].hist(null_hours, bins=np.arange(0, 21, 1), density=True, alpha=0.4, label="exposure null")
axes[0].hist(bd.hours_since_cleaning, bins=np.arange(0, 21, 1), density=True, histtype="step", lw=2,
             label="breakdowns (all)")
axes[0].hist(iso.hours_since_cleaning, bins=np.arange(0, 21, 2), density=True, histtype="step", lw=2,
             label="isolated only")
axes[0].set_xlabel("hours since that morning's cleaning")
axes[0].legend()
sns.boxplot(data=bd, x="cleaning_product_code", y="hours_since_cleaning", ax=axes[1], order=["CP1", "CP2", "CP3"])
axes[1].set_title("hours since cleaning by product")
savefig(fig, "hours_since_cleaning.png")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
sns.heatmap(trans_rate, annot=trans["sum"].astype(int), fmt="d", cmap="magma", ax=axes[0],
            cbar_kws={"label": "rate /1000 train-days"})
axes[0].set_title("previous-day x today product (annot = events)")
sw_tab = panel.dropna(subset=["product_switch"]).groupby("product_switch").agg(e=("y", "sum"), n=("y", "size"))
lo, hi = poisson_ci(sw_tab.e.values, sw_tab.n.values)
r_ = sw_tab.e / sw_tab.n * 1000
axes[1].errorbar(["same product", "switched"], r_, yerr=[r_ - lo * 1000, hi * 1000 - r_], fmt="o", capsize=4)
axes[1].set_ylabel("breakdowns / 1000 train-days")
axes[1].set_title("product switch vs previous day")
savefig(fig, "cleaning_product_switch.png")
flush_log()


# ------------------------------------------------------------------------------------------------
# 9. Weather x train-attribute (and weather x cleaning) interactions, conditional on date
# ------------------------------------------------------------------------------------------------
section("9. Weather x train attribute interactions (train-day panel, date + route fixed effects)")
# Date FE absorbs weather main effects; interactions ask whether hot/rainy/cold days hit some trains more.
inter_rows = []
for w_var in ["hot", "heavy_rain", "cold", "temp_c"]:
    for attr in ["C(train_type)", "C(batch_number)", "age_years", "C(cleaning_product_code)", "product_switch"]:
        data_ = pe
        m0 = smf.glm(f"y ~ {attr} + {base_terms}", data=data_, family=sm.families.Poisson()).fit()
        m1 = smf.glm(f"y ~ {attr} + {attr}:{w_var} + {base_terms}", data=data_, family=sm.families.Poisson()).fit()
        s, d, p = lr_test(m1, m0)
        inter_rows.append({"weather": w_var, "attribute": attr, "LR": s, "df": d, "p": p})
        register(f"interaction {w_var} x {attr}", p)
inter_tab = pd.DataFrame(inter_rows)
log(inter_tab.round(4).to_string(index=False))
inter_tab.to_csv(TAB / "weather_attribute_interactions.csv", index=False)
log("\nbreakdowns by type on hot / not-hot days (naive rates per 1000 train-days):")
log(panel.groupby(["hot", "train_type"]).agg(events=("y", "sum"), train_days=("y", "size"))
    .assign(rate=lambda d: d.events / d.train_days * 1000).round(2).to_string())
log(panel.groupby(["heavy_rain", "train_type"]).agg(events=("y", "sum"), train_days=("y", "size"))
    .assign(rate=lambda d: d.events / d.train_days * 1000).round(2).to_string())
flush_log()


# ------------------------------------------------------------------------------------------------
# 10. Repeat offenders: per-train overdispersion beyond route/line differences
# ------------------------------------------------------------------------------------------------
section("10. Repeat-offender trains")
m_route = smf.glm("n_bd ~ C(route_id)", data=per_train, family=sm.families.Poisson(), offset=per_train.log_exposure).fit()
per_train["mu_route"] = m_route.fittedvalues
disp = m_route.pearson_chi2 / m_route.df_resid
log(f"Pearson dispersion after route effects: {disp:.3f} (df={m_route.df_resid:.0f}); "
    f"chi2 p={stats.chi2.sf(m_route.pearson_chi2, m_route.df_resid):.3f}")
# Parametric bootstrap of the dispersion statistic and of the max count under the route-Poisson model
boot_disp, boot_max, boot_ge3 = [], [], []
for _ in range(2000):
    ysim = rng.poisson(per_train.mu_route.values)
    boot_disp.append(np.sum((ysim - per_train.mu_route.values) ** 2 / per_train.mu_route.values) / m_route.df_resid)
    boot_max.append(ysim.max())
    boot_ge3.append((ysim >= 3).sum())
boot_disp, boot_max, boot_ge3 = map(np.array, (boot_disp, boot_max, boot_ge3))
log(f"bootstrap: dispersion p={(1 + (boot_disp >= disp).sum()) / 2001:.3f}; max count observed "
    f"{per_train.n_bd.max()} (p={(1 + (boot_max >= per_train.n_bd.max()).sum()) / 2001:.3f}); "
    f"trains with >=3 observed {(per_train.n_bd >= 3).sum()} vs null mean {boot_ge3.mean():.1f} "
    f"(p={(1 + (boot_ge3 >= (per_train.n_bd >= 3).sum()).sum()) / 2001:.3f})")
register("repeat offenders: dispersion after route (bootstrap)", (1 + (boot_disp >= disp).sum()) / 2001)
per_train["p_upper"] = stats.poisson.sf(per_train.n_bd - 1, per_train.mu_route)
per_train["q_bh"] = multipletests(per_train.p_upper, method="fdr_bh")[1]
top = per_train.sort_values("p_upper").head(10)
log(top[["train_id", "route_id", "train_type", "batch_number", "age_years", "n_bd", "n_bd_burst", "mu_route",
         "p_upper", "q_bh"]].round(3).to_string(index=False))
obs_freq = per_train.n_bd.value_counts().sort_index()
exp_freq = pd.Series({k: stats.poisson.pmf(k, per_train.mu_route).sum() for k in range(per_train.n_bd.max() + 1)})
log("count frequency observed vs expected (route-Poisson):\n" + pd.DataFrame({"obs": obs_freq, "exp": exp_freq})
    .fillna(0).round(2).to_string())
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.bar(exp_freq.index - 0.2, exp_freq.values, width=0.4, label="Poisson (route rates)")
ax.bar(obs_freq.index + 0.2, obs_freq.values, width=0.4, label="observed")
ax.set_xlabel("breakdowns per train")
ax.set_ylabel("number of trains")
ax.legend()
savefig(fig, "repeat_offenders.png")
flush_log()


# ------------------------------------------------------------------------------------------------
# 11. Breakdown duration (onset -> recovery)
# ------------------------------------------------------------------------------------------------
section("11. Breakdown duration")
d = bd.duration_min
log(f"duration (min): mean={d.mean():.2f}, sd={d.std():.2f}, min={d.min():.2f}, max={d.max():.2f}; "
    f"KS vs Uniform({d.min():.0f},{np.ceil(d.max()):.0f}) p={stats.kstest(d, 'uniform', args=(5, 5)).pvalue:.3f}")
dur_rows = []
for f in ["train_type", "batch_number", "route_id", "line_id", "age_band", "at_station", "in_burst", "corridor",
          "cleaning_product_code", "cleaning_team", "cleaning_type", "month", "weekday"]:
    grp = [g_.duration_min.values for _, g_ in bd.groupby(f, observed=True) if len(g_) > 1]
    h, p = stats.kruskal(*grp)
    dur_rows.append({"factor": f, "H": h, "p": p,
                     "medians": bd.groupby(f, observed=True).duration_min.median().round(2).to_dict()})
    register(f"duration by {f} (Kruskal)", p)
for v in ["temp_c", "rain_mm", "age_years", "hours_since_cleaning", "hour"]:
    rho_d = stats.spearmanr(bd[v], bd.duration_min)
    dur_rows.append({"factor": v, "H": rho_d.statistic, "p": rho_d.pvalue, "medians": "Spearman rho in H column"})
    register(f"duration vs {v} (Spearman)", rho_d.pvalue)
dur_tab = pd.DataFrame(dur_rows)
log(dur_tab.to_string(index=False))
dur_tab.to_csv(TAB / "duration_tests.csv", index=False)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
axes[0].hist(d, bins=20)
axes[0].set_xlabel("minutes to recovery")
sns.boxplot(data=bd, x="route_id", y="duration_min", ax=axes[1])
sns.boxplot(data=bd, x="train_type", y="duration_min", ax=axes[2])
savefig(fig, "duration.png")
flush_log()


# ------------------------------------------------------------------------------------------------
# 12. Multiple testing summary
# ------------------------------------------------------------------------------------------------
section("12. Multiple-testing summary (Benjamini-Hochberg across all registered screening tests)")
reg = pd.DataFrame(pval_registry, columns=["test", "p"]).drop_duplicates("test")
reg["q_bh"] = multipletests(reg.p, method="fdr_bh")[1]
reg["p_bonferroni"] = np.minimum(reg.p * len(reg), 1)
reg = reg.sort_values("p")
log(f"{len(reg)} tests registered")
log(reg.head(25).to_string(index=False, float_format=lambda x: f"{x:.3g}"))
reg.to_csv(TAB / "multiple_testing_registry.csv", index=False)
flush_log()
log(f"\nDone. Figures and tables in {FIG.relative_to(ROOT)}; log in {LOG_PATH.relative_to(ROOT)}")
flush_log()
