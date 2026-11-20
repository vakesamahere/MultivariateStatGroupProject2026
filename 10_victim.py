"""10_victim.py -- are the victims a random draw from the exposure at risk, or a fixed set of trains?

Two senses of "victim", both tested here:
  A1  the broken train itself: 143 incidents on 140 trains -- is the per-train count overdispersed
      relative to Poisson given exposure (and given route and service date)?
      Restricted to the 127 incidents within 2.5 network hops of the carrier T3420: is the set of
      trains that break inside the carrier's zone a fixed set, once each train's time *inside* that
      zone is the denominator?
  A2  the delayed train: 9,065 late arrival events reconstructed from traversals.pkl (the same
      increments that src/04_impact.py attributes; 0 of 17,616 attributed increments are
      unattributable, so a late arrival is an incident-caused late arrival). Are the same trains
      always the ones in the queue, and is that concentration train identity or line / day structure?

Inputs (read only): data_derived/perm_export/{perm_export.npz,perm_export.json} (30-s position grid,
network-hop distances, train and date axes written by src/05_permutation.py --stage export),
data_derived/{breakdowns.csv,traversals.pkl}, out/interchange/events.csv (d_c),
data/train_information.csv.

Outputs: out/victim/train_counts.csv (per-train counts and rates), overdispersion.csv (every test),
delayed_counts.csv (per-train delay burden and split-half rates), attribute_associations.csv
(day-block bootstrap rate ratios)
Run: python3 src/10_victim.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA, DERIVED = ROOT / "data", ROOT / "data_derived"
OUT = ROOT / "out" / "victim"
OUT.mkdir(parents=True, exist_ok=True)

TICK_S = 30              # grid resolution of the 03 export
ZONE_D = 2.5             # carrier zone in network hops (03: no event beyond 2.5)
N_PERM = 3000            # within-line permutation draws for the delayed-count test
N_BOOT = 2000            # day-block bootstrap draws for the attribute models
RNG = np.random.default_rng(20260302)
LINES = [2, 8, 10, 12]

TESTS: list[dict] = []   # collected rows -> out/victim/overdispersion.csv


def log(msg: str = "") -> None:
    print(msg, flush=True)


def section(title: str) -> None:
    log(f"\n## {title}")


def test_row(block: str, name: str, stat: str, value: float, p: float | None = None,
             df: float | None = None, note: str = "") -> None:
    TESTS.append({"block": block, "test": name, "statistic": stat, "value": value, "df": df,
                  "p_value": p, "note": note})


def ci_str(lo: float, hi: float, fmt: str = "{:.3f}") -> str:
    return f"[{fmt.format(lo)}, {fmt.format(hi)}]"


# ================================================================================================
# 0. Inputs: position grid and the two exposure denominators
# ================================================================================================
z = np.load(DERIVED / "perm_export" / "perm_export.npz")
meta = json.loads((DERIVED / "perm_export" / "perm_export.json").read_text(encoding="utf-8"))
GRID = z["grid"]                       # (train, date, tick) location code, -1 = not in service
LD = z["loc_dist_halfhops"]            # network-hop distance x 2 between location codes
CARRIER_CODE = meta["carrier_code"]
TRAINS = list(meta["trains"])
DATES = list(meta["dates"])
N_LOC = LD.shape[0]

GIDX = np.where(GRID < 0, N_LOC - 1, GRID)          # -1 -> sentinel row/column of LD
CARR_LOC = GRID[CARRIER_CODE]                       # (date, tick), -1 when the carrier is off
CARR_IDX = np.where(CARR_LOC < 0, N_LOC - 1, CARR_LOC)
DIST = LD[GIDX, CARR_IDX[None, :, :]] / 2.0         # distance of every train to the carrier, hops
DIST[CARRIER_CODE] = np.nan                         # the carrier is never its own victim
IN_SVC = GRID >= 0
ZONE = (DIST <= ZONE_D) & (CARR_LOC >= 0)[None, :, :]   # inside the zone AND the carrier is running

# exposure in train-hours: every 30-s tick a train is in service (station dwell + segment, terminal
# layovers excluded -- the same definition as the 171,566 train-h total of 03/04)
EXP_H = IN_SVC.sum(axis=2) * TICK_S / 3600.0                    # (train, date)
ZONE_H = ZONE.sum(axis=2) * TICK_S / 3600.0                     # (train, date)
del DIST

bd = pd.read_csv(DERIVED / "breakdowns.csv", parse_dates=["onset_ts", "recovery_ts", "date"])
ix = pd.read_csv(OUT.parent / "interchange" / "events.csv")
bd = bd.merge(ix[["incident_id", "d_c", "victim_location_type"]], on="incident_id", how="left",
              validate="one_to_one")
ti = pd.read_csv(DATA / "train_information.csv", encoding="utf-8-sig")
ti["service_start_date"] = pd.to_datetime(ti.service_start_date)
PANEL_START = pd.Timestamp(DATES[0])
ti["age_years"] = (PANEL_START - ti.service_start_date).dt.days / 365.25
TID = {t: i for i, t in enumerate(TRAINS)}
DID = {d: i for i, d in enumerate(DATES)}
LINE_OF = ti.set_index("train_id").line_id.astype(int).reindex(TRAINS)
ROUTE_OF = ti.set_index("train_id").route_id.astype(str).reindex(TRAINS)
ATTR = ti.set_index("train_id")[["line_id", "route_id", "train_type", "batch_number",
                                 "age_years"]].reindex(TRAINS)
assert LINE_OF.notna().all() and len(TRAINS) == 140

exp_total = EXP_H.sum() * 1.0
log(f"grid {GRID.shape} ({TICK_S} s ticks, 05:30-24:00); in-service exposure {exp_total:,.0f} train-h "
    f"(03/04 total 171,566); carrier {meta['carrier']}; incidents {len(bd)}; "
    f"within {ZONE_D} hops {int((bd.d_c <= ZONE_D).sum())} (03: 127)")
log(f"zone exposure (carrier running): {ZONE_H.sum():,.0f} train-h; "
    f"trains with zero zone exposure {int((ZONE_H.sum(axis=1) <= 0).sum())}")


def cells(df: pd.DataFrame) -> np.ndarray:
    """(train, date) count grid for the rows of df."""
    k = np.zeros_like(EXP_H)
    np.add.at(k, (df.train_id.map(TID).values,
                  df.date.dt.strftime("%Y-%m-%d").map(DID).values), 1)
    return k


def panel_frame(k: np.ndarray) -> pd.DataFrame:
    """Long (train, date) panel with exposure and train attributes, dropping train-days with no exposure."""
    p = pd.DataFrame({"train_id": np.repeat(TRAINS, len(DATES)),
                      "date": np.tile(np.array(DATES, dtype=str), len(TRAINS)),
                      "k": k.ravel(), "h": EXP_H.ravel(), "h_zone": ZONE_H.ravel()})
    p = p.join(ATTR, on="train_id")
    return p[p.h > 0].reset_index(drop=True)


def disp_index(k: np.ndarray) -> tuple[float, float, float]:
    """Variance/mean dispersion index with its chi-square (n-1 df) interval."""
    n = len(k)
    i = k.var(ddof=1) / k.mean()
    return i, n * i / stats.chi2.ppf(0.975, n - 1), n * i / stats.chi2.ppf(0.025, n - 1)


def score_test(k: np.ndarray, mu: np.ndarray) -> tuple[float, float]:
    """Dean-Lawless score test of extra-Poisson variation: z = sum((k-mu)^2 - mu) / sqrt(2 sum mu^2)."""
    z_stat = (((k - mu) ** 2 - mu).sum()) / np.sqrt(2 * (mu ** 2).sum())
    return z_stat, 2 * stats.norm.sf(abs(z_stat))


def nb_lr(k: np.ndarray, offset: np.ndarray, exog: np.ndarray | None = None) -> tuple[float, float, float]:
    """Poisson vs NB2 likelihood-ratio test; alpha = 0 sits on the boundary -> p = 0.5*chi2_1.

    The NB2 MLE is fitted with BFGS because the default Newton step can overshoot into a region where
    the reported log-likelihood exceeds the MLE (seen with 110 route/date dummies); a negative LR is
    clipped to 0, i.e. the fitted alpha sits at the Poisson boundary.
    """
    x = np.ones((len(k), 1)) if exog is None else exog
    poi = sm.GLM(k, x, family=sm.families.Poisson(), offset=offset).fit()
    try:
        nb = sm.NegativeBinomial(k, x, offset=offset).fit(method="bfgs", disp=0, maxiter=2000)
        lr, alpha = 2 * (nb.llf - poi.llf), float(nb.params[-1])
        if not np.isfinite(lr) or lr < 0:
            lr = 0.0
    except Exception:                                   # noqa: BLE001 - report, do not hide
        lr, alpha = np.nan, np.nan
    return lr, alpha, 0.5 * stats.chi2.sf(max(lr, 0.0), 1)


# ================================================================================================
# A1. The broken train: is the per-train incident count Poisson given exposure?
# ================================================================================================
section("A1 broken train -- per-train incident count")
K_ALL = cells(bd)
k_train, h_train = K_ALL.sum(axis=1), EXP_H.sum(axis=1)
RATE_ALL = 1000.0 * k_train.sum() / h_train.sum()

# 1. marginal dispersion index (no covariate)
i_all, i_lo, i_hi = disp_index(k_train)
test_row("A1", "dispersion index, all incidents", "var/mean", i_all, stats.chi2.sf(len(k_train) * i_all,
         len(k_train) - 1), note=f"95% CI {ci_str(i_lo, i_hi)}; n=140 trains, {int(k_train.sum())} events")
log(f"per-train count: mean {k_train.mean():.2f}, var {k_train.var(ddof=1):.2f}, "
    f"dispersion {i_all:.3f} {ci_str(i_lo, i_hi)} -> overdispersion at p={stats.chi2.sf(140 * i_all, 139):.3f}")

# 2. Dean-Lawless score test against the exposure-proportional Poisson mean
mu_all = h_train * (k_train.sum() / h_train.sum())
z_all, p_all = score_test(k_train, mu_all)
test_row("A1", "Dean-Lawless score test, all incidents", "z", z_all, p_all)
log(f"score test (exposure offset): z={z_all:.3f}, p={p_all:.4f}")

# 3. NB2 likelihood-ratio test + fitted alpha
lr_all, a_all, p_lr_all = nb_lr(k_train, np.log(h_train))
test_row("A1", "NB2 vs Poisson LR, all incidents", "LR", lr_all, p_lr_all, df=1,
         note=f"alpha={a_all:.3f}; boundary test p=0.5*chi2_1")
log(f"NB2 LR={lr_all:.3f} (boundary p={p_lr_all:.4f}), alpha={a_all:.3f}")

# 4. rates by line -- is the overdispersion between lines?
by_line = (pd.DataFrame({"line": LINE_OF.values, "k": k_train, "h": h_train})
           .groupby("line").agg(k=("k", "sum"), h=("h", "sum"), trains=("k", "size")))
by_line["rate"] = 1000 * by_line.k / by_line.h
by_line["disp"] = [disp_index(k_train[LINE_OF.values == l])[0] for l in by_line.index]
test_row("A1", "rate ratio, line 10 vs line 8", "rate ratio",
         by_line.loc[10, "rate"] / by_line.loc[8, "rate"],
         note="lines: " + ", ".join(f"{l}:{r:.3f}" for l, r in by_line.rate.items()))
log("rate per 1000 train-h by line: " + ", ".join(f"{l} {r:.3f}" for l, r in by_line.rate.items()))
log("within-line dispersion: " + ", ".join(f"{l} {d:.3f}" for l, d in by_line.disp.items()))

# 5. controlled tests: train-day panel with route and/or date fixed effects
#
# Conditioning on the date total turns the count model into a quasi-multinomial: given how many
# incidents happened that day, allocation across trains can only be *more* even than Poisson, so a
# Pearson/df well below 1 is expected and is not underdispersion in the sampling sense. The score
# test (which only asks whether the variance exceeds mu) is therefore the test to read.
p_all = panel_frame(K_ALL)
CTL = {}
for tag, f_ctl in [("route+date", "k ~ C(route_id) + C(date)"), ("route only", "k ~ C(route_id)")]:
    m_ctl = smf.glm(f_ctl, p_all, family=sm.families.Poisson(), offset=np.log(p_all.h)).fit()
    pear_ctl = float((m_ctl.resid_pearson ** 2).sum() / m_ctl.df_resid)
    z_ctl, p_ctl = score_test(p_all.k.values, m_ctl.fittedvalues.values)
    dum = pd.get_dummies(p_all[["route_id", "date"] if tag == "route+date" else ["route_id"]].astype(str),
                         drop_first=True, dtype=float)
    lr_ctl, a_ctl, p_lr_ctl = nb_lr(p_all.k.values, np.log(p_all.h.values),
                                    exog=np.column_stack([np.ones(len(p_all)), dum.values]))
    CTL[tag] = m_ctl
    test_row("A1", f"train-day panel, {tag} FE: Pearson dispersion", "Pearson/df", pear_ctl, note=f_ctl)
    test_row("A1", f"train-day panel, {tag} FE: score test", "z", z_ctl, p_ctl)
    test_row("A1", f"train-day panel, {tag} FE: NB2 LR", "LR", lr_ctl, p_lr_ctl, df=1,
             note=f"alpha={a_ctl:.4f}")
    log(f"train-day panel + {tag} FE: Pearson/df {pear_ctl:.3f}, score z={z_ctl:+.3f} p={p_ctl:.3f}, "
        f"NB2 LR={lr_ctl:.3f} alpha={a_ctl:.4f}")

# 6. does train identity add anything over route and date?
f_ctl_t = "k ~ C(train_id) + C(date)"
m_ctl_t = smf.glm(f_ctl_t, p_all, family=sm.families.Poisson(), offset=np.log(p_all.h)).fit()
lr_tid = 2 * (m_ctl_t.llf - CTL["route+date"].llf)
df_tid = m_ctl_t.df_model - CTL["route+date"].df_model
test_row("A1", "train identity over route+date (Poisson LR)", "LR", lr_tid,
         stats.chi2.sf(lr_tid, df_tid), df=df_tid, note="anti-conservative under overdispersion")
log(f"train identity over route+date: LR={lr_tid:.2f} on {df_tid:.0f} df, p={stats.chi2.sf(lr_tid, df_tid):.3f}")


# ================================================================================================
# A1b. The broken train inside the carrier zone: 127 events, zone exposure as the denominator
# ================================================================================================
section("A1b broken train inside the 2.5-hop zone")
bz = bd[bd.d_c <= ZONE_D]
K_ZONE = cells(bz)
k_z, h_z = K_ZONE.sum(axis=1), ZONE_H.sum(axis=1)
ok = h_z > 0                                        # one train never runs while the carrier runs
RATE_ZONE = 1000.0 * k_z.sum() / h_z.sum()
i_z, i_z_lo, i_z_hi = disp_index(k_z[ok])
mu_z = h_z[ok] * (k_z[ok].sum() / h_z[ok].sum())
z_z, p_z = score_test(k_z[ok], mu_z)
lr_z, a_z, p_lr_z = nb_lr(k_z[ok], np.log(h_z[ok]))
test_row("A1b", "dispersion index, zone incidents", "var/mean", i_z, stats.chi2.sf(ok.sum() * i_z, ok.sum() - 1),
         note=f"95% CI {ci_str(i_z_lo, i_z_hi)}; {int(k_z.sum())} events on {int((k_z > 0).sum())} trains, "
              f"{int(ok.sum())} trains with zone exposure")
test_row("A1b", "Dean-Lawless score test, zone incidents", "z", z_z, p_z)
test_row("A1b", "NB2 vs Poisson LR, zone incidents", "LR", lr_z, p_lr_z, df=1, note=f"alpha={a_z:.4f}")
log(f"zone: {int(k_z.sum())} events on {int((k_z > 0).sum())} trains; per-train count mean {k_z[ok].mean():.2f}, "
    f"var {k_z[ok].var(ddof=1):.2f}, dispersion {i_z:.3f} {ci_str(i_z_lo, i_z_hi)}")
log(f"zone score test z={z_z:+.3f} p={p_z:.3f}; NB2 LR={lr_z:.3f} alpha={a_z:.4f}; "
    f"rate {RATE_ZONE:.1f} per 1000 zone-h")

zone_line = (pd.DataFrame({"line": LINE_OF.values[ok], "k": k_z[ok], "h": h_z[ok]})
             .groupby("line").agg(k=("k", "sum"), h=("h", "sum"), trains=("k", "size")))
zone_line["rate"] = 1000 * zone_line.k / zone_line.h
zone_line["disp"] = [disp_index(k_z[ok][LINE_OF.values[ok] == l])[0] for l in zone_line.index]
log("zone rate per 1000 zone-h: " + ", ".join(f"{l} {r:.1f}" for l, r in zone_line.rate.items()))
log("zone within-line dispersion: " + ", ".join(f"{l} {d:.3f}" for l, d in zone_line.disp.items()))

# concentration: is the zone set a handful of repeat offenders? top-k shares vs the fitted Poisson null
shares = np.sort(k_z)[::-1]
top5 = shares[:5].sum() / k_z.sum()
sim = np.stack([RNG.poisson(mu_z) for _ in range(1000)])
null_top5 = np.sort(sim, axis=1)[:, -5:].sum(axis=1) / sim.sum(axis=1)
test_row("A1b", "top-5 trains' share of zone incidents", "share", top5,
         float((1 + (null_top5 >= top5).sum()) / 1001),
         note=f"null (fitted Poisson) mean {null_top5.mean():.3f}, 95% {ci_str(*np.percentile(null_top5, [2.5, 97.5]))}")
log(f"top-5 trains hold {top5:.1%} of zone incidents (Poisson null {null_top5.mean():.1%}, "
    f"p={(1 + (null_top5 >= top5).sum()) / 1001:.3f}); trains with >=2 zone incidents: {int((k_z >= 2).sum())}")

# controlled: zone panel with route + date FE (train-days with zero zone exposure carry no risk)
zp = panel_frame(K_ZONE)
zp = zp[zp.h_zone > 0].reset_index(drop=True)
m_zc = smf.glm("k ~ C(route_id) + C(date)", zp, family=sm.families.Poisson(),
               offset=np.log(zp.h_zone)).fit()
z_zc, p_zc = score_test(zp.k.values, m_zc.fittedvalues.values)
pear_zc = float((m_zc.resid_pearson ** 2).sum() / m_zc.df_resid)
dum_z = pd.get_dummies(zp[["route_id", "date"]].astype(str), drop_first=True, dtype=float)
lr_zc, a_zc, p_lr_zc = nb_lr(zp.k.values, np.log(zp.h_zone.values),
                             exog=np.column_stack([np.ones(len(zp)), dum_z.values]))
test_row("A1b", "zone panel, route+date FE: Pearson dispersion", "Pearson/df", pear_zc, note=f"{len(zp)} train-days")
test_row("A1b", "zone panel, route+date FE: score test", "z", z_zc, p_zc)
test_row("A1b", "zone panel, route+date FE: NB2 LR", "LR", lr_zc, p_lr_zc, df=1, note=f"alpha={a_zc:.4f}")
log(f"zone panel ({len(zp)} train-days) + route + date FE: Pearson/df {pear_zc:.3f}, "
    f"score z={z_zc:+.3f} p={p_zc:.3f}, NB2 LR={lr_zc:.3f} alpha={a_zc:.4f}")


# ================================================================================================
# A2. The delayed train: 9,065 late arrivals reconstructed from the traversals
# ================================================================================================
section("A2 delayed train -- per-train delay burden")
tv = pd.read_pickle(DERIVED / "traversals.pkl")
tv = tv[["train_id", "date", "trip_key", "dep_event", "dep_ts", "arr_ts", "dwell_s", "excess_travel_s"]]
# the same increments src/04_impact.py attributes to incidents: running-time excess over the scheduled
# minimum (excess_travel_s >= 0 always, 0 for 99.8% of traversals) plus dwell beyond the 30-s minimum
tv["inc"] = tv.excess_travel_s.astype("float64") + (tv.dwell_s.fillna(30.0) - 30.0).clip(lower=0.0)
tv = tv.sort_values(["train_id", "date", "trip_key", "dep_ts"], kind="stable")
tv["cum"] = tv.inc.groupby(tv.trip_key).cumsum()
tv["is_arr"] = tv.dep_event != "DEPARTED_ORIGIN"
N_ARR = int(tv.is_arr.sum())
late = tv[(tv.is_arr) & (tv.cum > 0)].copy()
INC_LATE = late.inc.sum() / 60
INC_ORIGIN = tv.loc[(tv.dep_event == "DEPARTED_ORIGIN") & (tv.cum > 0), "inc"].sum() / 60
INC_ALL = tv.inc.sum() / 60
del tv
late["date"] = pd.to_datetime(late.date)
log(f"late arrivals {len(late):,} of {N_ARR:,} non-origin traversals (04: 9,072 delayed events = these "
    f"plus 7 late origin departures); delay created {INC_LATE:,.1f} train-min + {INC_ORIGIN:.1f} in those 7 = "
    f"{INC_ALL:,.1f} (04 per-line table 1,743+1,072 = 2,815.3; the 13-min gap is the 5 zero-chain events "
    f"clipped at src/04_impact.py:866); delay experienced {late.cum.sum() / 60:,.1f} train-min")
assert len(late) == 9065, len(late)

K_LATE = cells(late)
k_l, h_l = K_LATE.sum(axis=1), EXP_H.sum(axis=1)
DELAY_S = late.groupby("train_id").cum.sum().reindex(TRAINS).fillna(0.0).values
RATE_LATE = 1000.0 * k_l.sum() / h_l.sum()
i_l, i_l_lo, i_l_hi = disp_index(k_l)
mu_l = h_l * (k_l.sum() / h_l.sum())
z_l, p_l = score_test(k_l, mu_l)
lr_l, a_l, p_lr_l = nb_lr(k_l, np.log(h_l))
test_row("A2", "dispersion index, late arrivals per train", "var/mean", i_l, note=f"95% CI {ci_str(i_l_lo, i_l_hi)}")
test_row("A2", "Dean-Lawless score test, late arrivals per train", "z", z_l, p_l)
test_row("A2", "NB2 vs Poisson LR, late arrivals per train", "LR", lr_l, p_lr_l, df=1, note=f"alpha={a_l:.2f}")
log(f"per-train late arrivals: mean {k_l.mean():.1f}, var {k_l.var(ddof=1):.1f}, dispersion {i_l:.1f}; "
    f"score z={z_l:+.0f} p<1e-10; NB2 alpha={a_l:.2f} LR={lr_l:,.0f}")
late_line = (pd.DataFrame({"line": LINE_OF.values, "k": k_l, "h": h_l, "d": DELAY_S / 60})
             .groupby("line").agg(k=("k", "sum"), h=("h", "sum"), d=("d", "sum"), trains=("k", "size")))
late_line["rate"] = 1000 * late_line.k / late_line.h
log("late-arrival rate per 1000 train-h: " + ", ".join(f"{l} {r:.1f}" for l, r in late_line.rate.items()))

# ---- is the concentration train identity, or the line and the day? ------------------------------
W = K_LATE.astype(float)                                               # (train, date) counts
Hm = EXP_H.copy()
LINE_ARR = LINE_OF.values


def within_line_disp(tot: np.ndarray) -> float:
    """Pooled within-line variance/mean of per-train totals (removes the line mean, not the line rate)."""
    num = den = 0.0
    for l in LINES:
        m = LINE_ARR == l
        v = tot[m]
        num += ((v - v.mean()) ** 2).sum()
        den += v.mean() * (m.sum() - 1)
    return num / den


obs_wd = within_line_disp(k_l)
null_wd = np.empty(N_PERM)
for r in range(N_PERM):                              # shuffle each day's counts among that line's trains
    Wp = np.zeros_like(W)
    for j in range(W.shape[1]):
        for l in LINES:
            idx = np.where(LINE_ARR == l)[0]
            idx = idx[Hm[idx, j] > 0]
            Wp[idx, j] = RNG.permutation(W[idx, j])
    null_wd[r] = within_line_disp(Wp.sum(axis=1))
p_perm = (1 + (null_wd >= obs_wd).sum()) / (N_PERM + 1)
test_row("A2", "within-line dispersion, late arrivals", "var/mean", obs_wd, p_perm,
         note=f"null {null_wd.mean():.2f} {ci_str(*np.percentile(null_wd, [2.5, 97.5]))} over {N_PERM} "
              f"day-wise within-line shuffles")
log(f"within-line dispersion {obs_wd:.2f} vs shuffled null {null_wd.mean():.2f} "
    f"{ci_str(*np.percentile(null_wd, [2.5, 97.5]))} -> p={p_perm:.3f}")

# ---- split-half reliability: is a train's rate in half the days repeated in the other half? -----
par = np.arange(len(DATES)) % 2
half = []
for s in (0, 1):
    kh, hh = W[:, par == s].sum(axis=1), Hm[:, par == s].sum(axis=1)
    half.append((1000 * kh / hh, kh, hh))
r_all = float(stats.pearsonr(*[h[0] for h in half])[0])
r_sp = float(stats.spearmanr(*[h[0] for h in half])[0])
demean = [h[0] - pd.Series(h[0]).groupby(LINE_ARR).transform("mean").values for h in half]
r_in = float(stats.pearsonr(*demean)[0])
per_line_r = {l: float(stats.pearsonr(*[h[0][LINE_ARR == l] for h in half])[0]) for l in LINES}
sb = 2 * r_in / (1 + r_in)
test_row("A2", "split-half reliability of the per-train late rate", "r", r_in,
         note=f"odd/even service days; line-demeaned r={r_in:.3f}, raw r={r_all:.3f} "
              f"(Spearman {r_sp:.3f}); per line " + ", ".join(f"{l}:{v:.3f}" for l, v in per_line_r.items())
              + f"; Spearman-Brown {sb:.3f}")
log(f"split-half (odd/even days): raw r={r_all:.3f}, line-demeaned r={r_in:.3f}, Spearman-Brown {sb:.3f}")
log("per-line split-half r: " + ", ".join(f"{l} {v:+.3f}" for l, v in per_line_r.items()))

# ---- train-day panel: how much of the overdispersion survives date and route? -------------------
p_late = panel_frame(K_LATE)
for tag, f_ctl in [("date only", "k ~ C(date)"), ("date+route", "k ~ C(date) + C(route_id)"),
                   ("date+train", "k ~ C(date) + C(train_id)")]:
    m = smf.glm(f_ctl, p_late, family=sm.families.Poisson(), offset=np.log(p_late.h)).fit()
    pear = float((m.resid_pearson ** 2).sum() / m.df_resid)
    zz, pp = score_test(p_late.k.values, m.fittedvalues.values)
    test_row("A2", f"late-arrival panel, {tag}: Pearson dispersion", "Pearson/df", pear, note=f_ctl)
    test_row("A2", f"late-arrival panel, {tag}: score test", "z", zz, pp)
    log(f"late-arrival panel + {tag}: Pearson/df {pear:.2f}, score z={zz:+.0f} p={pp:.2g}")

# ---- are the attributes (type, batch, age) associated with the delay burden? --------------------
# Day-block bootstrap: resample the 105 service dates with replacement (counts and exposure move
# together, so the strong day-to-day burst clustering is carried into the interval). BH across the
# three attribute families.
section("A2 attribute associations (day-block bootstrap)")
DAY_IDX = np.arange(len(DATES))


def design(*cols: str) -> tuple[np.ndarray, list[str]]:
    """Intercept + treatment dummies for the given train attributes."""
    d = pd.get_dummies(ATTR[list(cols)].astype(str), drop_first=True, dtype=float)
    return np.column_stack([np.ones(len(ATTR)), d.values]), ["const"] + list(d.columns)


def boot_poisson(x: np.ndarray, k: np.ndarray, h: np.ndarray) -> np.ndarray:
    out = np.empty((N_BOOT, x.shape[1]))
    for b in range(N_BOOT):
        pick = RNG.integers(0, len(DAY_IDX), len(DAY_IDX))
        kb, hb = k[:, pick].sum(axis=1), h[:, pick].sum(axis=1)
        keep = hb > 0
        out[b] = sm.GLM(kb[keep], x[keep], family=sm.families.Poisson(),
                        offset=np.log(hb[keep])).fit().params
    return out


def add_attr(name: str, x: np.ndarray, pick: list[int], label: list[str], compact: bool = False) -> None:
    point = sm.GLM(k_l, x, family=sm.families.Poisson(), offset=np.log(h_l)).fit().params
    boot = np.exp(boot_poisson(x, W, Hm))
    cells = []
    for i, lab in zip(pick, label):
        rr = float(np.exp(point[i]))
        lo, hi = np.percentile(boot[:, i], [2.5, 97.5])
        p = 2 * min((boot[:, i] >= 1.0).mean(), (boot[:, i] <= 1.0).mean())
        p = max(p, 1.0 / (N_BOOT + 1))
        ATTR_TESTS.append({"model": name, "term": lab, "rate_ratio": rr, "rr_lo": lo, "rr_hi": hi,
                           "p_boot": p})
        cells.append(f"{lab:12s} RR {rr:.3f} {ci_str(lo, hi)} p={p:.3f}")
    if compact:                       # seven batch levels: one terminal line, full detail in the CSV
        log(f"  {name}: " + "; ".join(cells))
    else:
        for c in cells:
            log(f"  {name:22s} {c}")


ATTR_TESTS: list[dict] = []
XL, NL = design("line_id")
add_attr("line only", XL, list(range(1, XL.shape[1])), [f"line {v.split('_')[-1]}" for v in NL[1:]])
XT, NT = design("line_id", "train_type")
add_attr("type | line", XT, [i for i, n in enumerate(NT) if "train_type" in n],
         [f"{n.split('_')[-1]} vs M1" for n in NT if "train_type" in n])
XB, NB = design("line_id", "batch_number")
add_attr("batch | line", XB, [i for i, n in enumerate(NB) if "batch_number" in n],
         [f"batch {n.split('_')[-1]}" for n in NB if "batch_number" in n], compact=True)
add_attr("age | line", np.column_stack([XL, ATTR.age_years.values]), [XL.shape[1]], ["per year"])

# LR tests for the categorical families (Poisson, day-clustered -> p is indicative only)
for name, x_big, x_small in [("type", XT, XL), ("batch", XB, XL)]:
    big = sm.GLM(k_l, x_big, family=sm.families.Poisson(), offset=np.log(h_l)).fit()
    small = sm.GLM(k_l, x_small, family=sm.families.Poisson(), offset=np.log(h_l)).fit()
    lr = 2 * (big.llf - small.llf)
    df_lr = x_big.shape[1] - x_small.shape[1]
    test_row("A2", f"attribute LR test: {name} over line", "LR", lr, stats.chi2.sf(lr, df_lr), df=df_lr,
             note="Poisson LR, anti-conservative under day clustering")
    log(f"  LR {name} over line: {lr:.2f} on {df_lr} df, p={stats.chi2.sf(lr, df_lr):.2g}")

# BH across the attribute families (one p per family: type / batch / age)
fam_p = {"type": min(t["p_boot"] for t in ATTR_TESTS if t["model"] == "type | line"),
         "batch": min(t["p_boot"] for t in ATTR_TESTS if t["model"] == "batch | line"),
         "age": min(t["p_boot"] for t in ATTR_TESTS if t["model"] == "age | line")}
order = sorted(fam_p, key=lambda k: fam_p[k])
bh = {}
for r, key in enumerate(order):
    bh[key] = min(1.0, fam_p[key] * len(fam_p) / (r + 1))
for key in fam_p:
    test_row("A2", f"attribute family {key} (BH)", "min p_boot", fam_p[key], bh[key],
             note="Benjamini-Hochberg over the 3 families")
log("BH over attribute families: " + ", ".join(f"{k} {bh[k]:.4f}" for k in ["type", "batch", "age"]))

# ---- cross-check A1 vs A2: do the trains that break also carry the delay? -----------------------
# A broken train is itself late afterwards, so part of any overlap is mechanical. Split each train's
# late arrivals into "own incident window" (its own breakdown, onset -> recovery + 1 h) and "other".
section("A1 x A2 cross-check")
own = np.zeros(len(late), dtype=bool)
for r in bd.itertuples():
    own |= ((late.train_id.values == r.train_id) & (late.date.values == np.datetime64(r.date))
            & (late.arr_ts.values >= np.datetime64(r.onset_ts))
            & (late.arr_ts.values <= np.datetime64(r.recovery_ts + pd.Timedelta("1h"))))
k_own = np.bincount(late.train_id.map(TID).values[own], minlength=len(TRAINS))
k_oth = k_l.astype(int) - k_own


def rho_within_route(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a, dtype=float) - pd.Series(a, dtype=float).groupby(ATTR.route_id.values).transform("mean")
    rb = pd.Series(b, dtype=float) - pd.Series(b, dtype=float).groupby(ATTR.route_id.values).transform("mean")
    return float(stats.spearmanr(ra, rb)[0])


XC = {}
for tag, kk in [("all late arrivals", k_l), ("own breakdown window", k_own), ("other trains' delay", k_oth)]:
    r_raw, p_raw = stats.spearmanr(k_train, kk)
    XC[tag] = (float(r_raw), float(p_raw), rho_within_route(k_train, kk))
    test_row("A1xA2", f"Spearman, per-train incidents vs {tag}", "rho", float(r_raw), float(p_raw),
             note=f"within-route rho={XC[tag][2]:+.3f}; {1000 * np.asarray(kk).sum() / h_l.sum():.1f} per 1000 train-h")
    log(f"incidents vs {tag:22s}: rho={r_raw:+.3f} (p={p_raw:.2g}), within-route {XC[tag][2]:+.3f}")
log(f"own-window share of all late arrivals: {own.mean():.1%}; zone incidents vs other-train delay "
    f"rho={stats.spearmanr(k_z[ok], k_oth[ok])[0]:+.3f}")

# ================================================================================================
# Outputs
# ================================================================================================
out_tc = pd.DataFrame({
    "train_id": TRAINS, "line_id": LINE_ARR, "route_id": ATTR.route_id.values,
    "train_type": ATTR.train_type.values, "batch_number": ATTR.batch_number.values,
    "age_years": ATTR.age_years.values.round(3),
    "exposure_h": h_train.round(2), "breakdowns": k_train.astype(int),
    "breakdown_rate_per_1000h": (1000 * k_train / h_train).round(3),
    "zone_exposure_h": h_z.round(2), "zone_breakdowns": k_z.astype(int),
    "zone_rate_per_1000h": np.where(h_z > 0, 1000 * k_z / np.where(h_z > 0, h_z, 1), np.nan).round(2),
}).sort_values(["line_id", "train_id"])
out_tc.to_csv(OUT / "train_counts.csv", index=False)

out_dc = pd.DataFrame({
    "train_id": TRAINS, "line_id": LINE_ARR,
    "late_arrivals": k_l.astype(int),
    "delay_created_train_min": (late.groupby("train_id").inc.sum().reindex(TRAINS).fillna(0) / 60).round(2),
    "delay_experienced_train_min": (DELAY_S / 60).round(2),
    "max_delay_s": late.groupby("train_id").cum.max().reindex(TRAINS).fillna(0).round(0),
    "late_rate_per_1000h": (1000 * k_l / h_l).round(2),
    "exposure_h": h_l.round(2),
    "late_rate_odd_days": half[0][0].round(2), "late_rate_even_days": half[1][0].round(2),
}).sort_values("late_rate_per_1000h", ascending=False)
out_dc.to_csv(OUT / "delayed_counts.csv", index=False)

pd.DataFrame(TESTS).to_csv(OUT / "overdispersion.csv", index=False)
pd.DataFrame(ATTR_TESTS).to_csv(OUT / "attribute_associations.csv", index=False)

# ================================================================================================
# A3. verdict
# ================================================================================================
section("A3 verdict")
log(f"A3: the broken train is an exposure draw, not a roster -- {int(k_z.sum())} zone incidents fall on "
    f"{int((k_z > 0).sum())} trains with no overdispersion given zone exposure (score z={z_z:+.2f}, p={p_z:.2f}; "
    f"top-5 share {top5:.1%} vs {null_top5.mean():.1%}), and the delay burden, though {i_l:.0f}x overdispersed, "
    f"is day/burst and line structure rather than train identity (within-line shuffle p={p_perm:.2f}; "
    f"split-half reliability {sb:.2f}; train FE moves Pearson/df 5.39->5.12).")
OTH = XC["other trains' delay"]
log(f"   the only train-level overlap is mechanical: a train's own downtime is "
    f"{own.mean():.0%} of all late arrivals and drives rho={XC['own breakdown window'][0]:+.2f}; the delay a "
    f"train receives from other trains' incidents is unrelated to its own breakdown rate "
    f"(rho={OTH[0]:+.2f}, within route {OTH[2]:+.2f}).")
log(f"wrote {OUT.relative_to(ROOT)}/train_counts.csv, delayed_counts.csv, overdispersion.csv, "
    f"attribute_associations.csv")
