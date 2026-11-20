"""13 Carrier state at the moment of each breakdown (Shanghai Metro semi-synthetic data).

Question: when a breakdown strikes, what is the carrier train T3420 doing - moving or standing, and which
movement phase? Is that distribution distinguishable from the carrier's own time budget?

Method
 1. Carrier state on a 30-s grid, 6 classes: depart / cruise / arrive (moving, split by the fraction f of the
    segment already covered), dwell (30-s mid-trip stop), terminal (end-of-trip layover), off (not in service).
 2. Exposure (denominator) computed on the same grid as src/03_mechanism.py: each tick contributes
    "carrier state x number of OTHER trains in service at that tick", i.e. train-hours of possible victims,
    not the carrier's own time.
 3. Event side: the carrier state at each of the 143 onsets, from the exact timestamp (not snapped to a tick).
 4. Chi-square goodness of fit (observed vs exposure share), per-state Poisson RR against cruise with exact CI,
    Benjamini-Hochberg q values; repeated separately for same-line (Line 10) and cross-line (2/8/12) victims.

Outputs: out/carrier/*.csv, out/carrier/carrier_state_results.txt, figures/carrier/*.png.

Run: python3 src/13_carrier_state.py    (~1-2 min, ~1.5 GB peak RAM)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data_derived"
GEO = ROOT / "out" / "geo"
INTER = ROOT / "out" / "interchange"
OUT = ROOT / "out" / "carrier"
FIG = ROOT / "figures" / "carrier"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT / "carrier_state_results.txt"

CARRIER = "T3420"  # the never-breaking Line-10 train found in 03/12
TICK_S = 30  # grid resolution, identical to src/03_mechanism.py (dwell is 30 s, so a stop occupies one tick)
STATES = ["depart", "cruise", "arrive", "dwell", "terminal", "off"]
MOVING = ["depart", "cruise", "arrive"]
STOPPED = ["dwell", "terminal"]
REF = "cruise"  # RR reference state
TERM = STATES.index("terminal")  # 4; needs to be defined before the state grid is built
F_DEPART, F_ARRIVE = 0.25, 0.75  # f < 0.25 -> leaving the station; f >= 0.75 -> braking into the next one
SAME_LINE = "10"  # victim shares the carrier's line
_log_lines: list[str] = []


def log(*parts) -> None:
    text = " ".join(str(p) for p in parts)
    shown = text.splitlines()
    print("\n".join(shown[:15]) + (f"\n   ... (+{len(shown) - 15} lines in results file)" if len(shown) > 15
                                   else ""), flush=True)
    _log_lines.append(text)


def section(title: str) -> None:
    log("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90)


def flush_log() -> None:
    LOG_PATH.write_text("\n".join(_log_lines) + "\n", encoding="utf-8")


def savefig(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def poisson_ci(k, exposure, level: float = 0.95):
    """Exact (Garwood) CI for a Poisson rate k/exposure - same helper as src/03_mechanism.py."""
    k = np.asarray(k, dtype=float)
    a = (1 - level) / 2
    lo = np.where(k > 0, stats.chi2.ppf(a, 2 * k) / 2, 0.0)
    hi = stats.chi2.ppf(1 - a, 2 * k + 2) / 2
    return lo / exposure, hi / exposure


def bh_q(p) -> np.ndarray:
    """Benjamini-Hochberg q values (monotone, capped at 1)."""
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    q = np.empty(n, dtype=float)
    q[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def pois_two_sided(k: int, lam: float) -> float:
    """Two-sided exact Poisson test of k against mean lam (conditional binomial form, capped at 1)."""
    if lam <= 0:
        return 1.0
    lo = stats.poisson.cdf(k, lam) if k > 0 else 1.0
    hi = stats.poisson.sf(k - 1, lam)
    return float(min(1.0, 2 * min(lo, hi)))


# ================================================================================================
section("1. Load traversals and rebuild the 30-s service grid (same convention as src/03_mechanism.py)")
# ================================================================================================
trav = pd.read_pickle(DERIVED / "traversals.pkl")
_keep = ["train_id", "line_id", "date", "trip_key", "segment", "from_station", "to_station",
         "dep_ts", "arr_ts", "dwell_s", "turnaround_s"]
trav = trav[_keep]
log(f"traversals loaded and trimmed to {len(_keep)} columns")
bd = pd.read_csv(DERIVED / "breakdowns.csv", parse_dates=["onset_ts"],
                 usecols=["incident_id", "train_id", "line_id", "date", "onset_ts", "location",
                          "location_type"])
geo = pd.read_csv(GEO / "events_geo.csv", usecols=["incident_id", "carrier_state", "d_c"])
inter = pd.read_csv(INTER / "events.csv", usecols=["incident_id", "lag_min"])
log(f"traversals {len(trav):,} rows; breakdowns {len(bd)}; carrier = {CARRIER}")

# Grid axes. A tick is the instant t_start + i*30 s; an interval [s, e) covers ticks ceil(s) .. ceil(e)-1.
DATES = np.array(sorted(trav.date.unique()))
DATE_IDX = {pd.Timestamp(d): i for i, d in enumerate(DATES)}
TRAINS = sorted(trav.train_id.astype(str).unique())
TRAIN_IDX = {t: i for i, t in enumerate(TRAINS)}
sec_of_day = lambda ts, day: (ts - day).dt.total_seconds().values  # noqa: E731

# Exposure intervals, verbatim from 03: station = [dep_ts - dwell_s, dep_ts) at from_station (no trip origin);
# segment = [dep_ts, arr_ts). Terminal layovers are handled separately in section 2 (the carrier only).
dw = trav[trav.dwell_s.notna()]
st_start = dw.dep_ts - pd.to_timedelta(dw.dwell_s, unit="s")
t_start_s = float(np.floor(np.min([sec_of_day(st_start, dw.date).min(),
                                   sec_of_day(trav.dep_ts, trav.date).min()]) / 3600) * 3600)
t_end_s = float(np.max(sec_of_day(trav.arr_ts, trav.date)))
NT = int(np.ceil((t_end_s - t_start_s) / TICK_S)) + 1
log(f"grid {len(TRAINS)} trains x {len(DATES)} dates x {NT} ticks of {TICK_S} s "
    f"(from {t_start_s / 3600:.1f}h to {t_end_s / 3600:.1f}h)")


def tick_occupancy(start: pd.Series, end: pd.Series, train_id: pd.Series):
    """(train_idx, date_idx, tick, f, interval_idx) for every tick covered by the intervals [start, end).

    f is the fraction of the interval already covered at that tick (only meaningful for segments).
    """
    s_rel = (sec_of_day(start, start.dt.normalize()) - t_start_s) / TICK_S
    e_rel = (sec_of_day(end, end.dt.normalize()) - t_start_s) / TICK_S
    a_t, b_t = np.ceil(s_rel).astype(np.int64), np.ceil(e_rel).astype(np.int64)
    n_t = np.maximum(b_t - a_t, 0)
    rep = np.repeat(np.arange(len(start)), n_t)
    offs = np.arange(n_t.sum()) - np.repeat(np.cumsum(n_t) - n_t, n_t)
    tick = a_t[rep] + offs
    f = (tick - s_rel[rep]) / np.maximum(e_rel[rep] - s_rel[rep], 1e-9)
    return (train_id.astype(str).map(TRAIN_IDX).values[rep].astype(np.int32),
            start.dt.normalize().map(DATE_IDX).values[rep].astype(np.int32), tick, f, rep)


# In-service flag grid, built exactly as in 03: a train-date-tick cell is 1 if any exposure interval covers it.
GRID = np.zeros((len(TRAINS), len(DATES), NT), dtype=np.int8)
for start, end, tid in [(st_start, dw.dep_ts, dw.train_id), (trav.dep_ts, trav.arr_ts, trav.train_id)]:
    ti, di, tick, _, _ = tick_occupancy(start, end, tid)
    GRID[ti, di, tick] = 1
    del ti, di, tick
SVC = GRID.sum(axis=0, dtype=np.int32)  # trains in service at each (date, tick), carrier included
CI = TRAIN_IDX[CARRIER]
log(f"train-ticks in service {int(SVC.sum()):,} = {SVC.sum() * TICK_S / 3600:,.0f} train-h; "
    f"max trains at one tick {int(SVC.max())}; carrier in service {int(GRID[CI].sum()):,} ticks "
    f"({GRID[CI].sum() * TICK_S / 3600:,.0f} train-h)")
log(f"exposure intervals: station {len(dw):,}, segment {len(trav):,}")
flush_log()

# ================================================================================================
section("2. Carrier state on the grid and exposure (denominator = other trains in service)")
# ================================================================================================
car = trav[trav.train_id == CARRIER].sort_values(["date", "dep_ts"]).reset_index(drop=True)
car["next_dep"] = car.dep_ts.shift(-1)
car["next_trip"] = car.trip_key.shift(-1)
car["next_date"] = car.date.shift(-1)
# A station spell is [arr_ts, next dep_ts) at to_station; it is a terminal layover when the next departure starts a
# new trip (trip_key changes) - those gaps are 31-41 min at S003/S089, against 30 s for an ordinary dwell.
stn = car[car.next_dep.notna() & (car.next_date == car.date) & (car.next_dep > car.arr_ts)].copy()
stn["terminal"] = stn.next_trip != stn.trip_key
log(f"carrier: {len(car):,} segment traversals, {len(stn):,} station spells "
    f"({int(stn.terminal.sum()):,} terminal layovers, median terminal "
    f"{(stn.next_dep - stn.arr_ts).dt.total_seconds()[stn.terminal].median() / 60:.0f} min vs dwell "
    f"{(stn.next_dep - stn.arr_ts).dt.total_seconds()[~stn.terminal].median():.0f} s)")

# 6-class state of the carrier at every tick. 0-2 moving (depart / cruise / arrive), 3-4 stopped, 5 off.
CSTATE = np.full((len(DATES), NT), STATES.index("off"), dtype=np.int8)
ti, di, tk, f, _ = tick_occupancy(car.dep_ts, car.arr_ts, car.train_id)
CSTATE[di, tk] = np.digitize(f, [F_DEPART, F_ARRIVE]).astype(np.int8)
ti2, di2, tk2, _, ridx = tick_occupancy(stn.arr_ts, stn.next_dep, stn.train_id)
CSTATE[di2, tk2] = np.where(stn.terminal.values[ridx], TERM, STATES.index("dwell")).astype(np.int8)
del ti, di, tk, f, ti2, di2, tk2, ridx

# Consistency check against the 03 grid. 03 deliberately leaves terminal layovers out of its exposure ("no
# breakdown ever happens at a terminal"), so the two must agree once terminal ticks are set aside.
agree = float(np.mean(np.isin(CSTATE, [STATES.index(s) for s in MOVING + ["dwell"]]) == (GRID[CI] == 1)))
log(f"carrier state grid agrees with the 03 in-service grid (terminal layovers set aside) on {agree:.4%} of "
    f"{len(DATES) * NT:,} ticks; terminal ticks {int((CSTATE == TERM).sum()):,}")
flush_log()

# Exposure = carrier state x number of OTHER trains in service at that tick, in train-hours. This is the 03
# convention (the risk set is every train that could have broken down, not the carrier's own clock).
N_OTHER = (SVC - GRID[CI]).astype(np.int32)  # (date, tick); carrier excluded from its own risk set
assert (N_OTHER >= 0).all()
EXPO_TICKS = np.bincount(CSTATE.ravel(), weights=N_OTHER.ravel().astype(np.float64), minlength=len(STATES))
CARRIER_TICKS = np.bincount(CSTATE.ravel(), minlength=len(STATES))
EXPO = pd.DataFrame({"carrier_state": STATES,
                     "carrier_ticks": CARRIER_TICKS,
                     "carrier_h": CARRIER_TICKS * TICK_S / 3600,
                     "exposure_train_h": EXPO_TICKS * TICK_S / 3600,
                     "mean_other_trains": np.where(CARRIER_TICKS > 0, EXPO_TICKS / np.maximum(CARRIER_TICKS, 1), 0.0)})
EXPO["share"] = EXPO.exposure_train_h / EXPO.exposure_train_h.sum()
EXPO["carrier_share"] = EXPO.carrier_h / EXPO.carrier_h.sum()
EXPO.to_csv(OUT / "carrier_state_exposure.csv", index=False)
log("exposure by carrier state (train-hours of OTHER trains in service, and the carrier's own clock):")
log(EXPO.round(4).to_string(index=False))
log(f"total exposure {EXPO.exposure_train_h.sum():,.0f} train-h over {int(N_OTHER.sum()):,} other-train-ticks; "
    f"carrier own clock {EXPO.carrier_h.sum():,.0f} h over {len(DATES)} days = "
    f"{EXPO.carrier_h.sum() / len(DATES):.1f} h/day, of which terminal layovers "
    f"{EXPO.loc[EXPO.carrier_state == 'terminal', 'carrier_h'].iat[0]:.0f} h")
flush_log()

# placeholder: events
# ================================================================================================
section("3. Carrier state at each breakdown onset (exact timestamp, not snapped to a tick)")
# ================================================================================================
log(f"breakdowns belonging to the carrier itself: {int((bd.train_id == CARRIER).sum())}")
assert (bd.train_id != CARRIER).all(), "carrier has its own breakdowns?"
seg_g = {pd.Timestamp(d): g.sort_values("dep_ts").reset_index(drop=True) for d, g in car.groupby("date")}
stn_g = {pd.Timestamp(d): g.sort_values("arr_ts").reset_index(drop=True) for d, g in stn.groupby("date")}
rows = []
for r in bd.itertuples():
    day = pd.Timestamp(r.onset_ts).normalize()
    t = pd.Timestamp(r.onset_ts)
    state, f_val, where = "off", np.nan, ""
    g = seg_g.get(day)
    if g is not None and len(g):
        i = int(np.searchsorted(g.dep_ts.values, np.datetime64(t), side="right")) - 1
        if i >= 0 and t < g.arr_ts.iat[i]:  # inside a segment traversal
            span = (g.arr_ts.iat[i] - g.dep_ts.iat[i]).total_seconds()
            f_val = float(np.clip((t - g.dep_ts.iat[i]).total_seconds() / span, 0.0, 1.0))
            state = STATES[int(np.digitize([f_val], [F_DEPART, F_ARRIVE])[0])]
            where = g.segment.iat[i]
    if state == "off":
        h = stn_g.get(day)
        if h is not None and len(h):
            j = int(np.searchsorted(h.arr_ts.values, np.datetime64(t), side="right")) - 1
            if j >= 0 and t < h.next_dep.iat[j]:  # standing at a station
                state = "terminal" if h.terminal.iat[j] else "dwell"
                where = str(h.to_station.iat[j])
    rows.append({"incident_id": r.incident_id, "train_id": r.train_id, "onset_ts": t,
                 "victim_line": str(r.line_id), "carrier_state": state, "f": f_val, "carrier_seg": where,
                 "victim_at": r.location, "victim_location_type": r.location_type})
EV = (pd.DataFrame(rows)
      .merge(geo.rename(columns={"carrier_state": "carrier_state_geo"}), on="incident_id", how="left")
      .merge(inter, on="incident_id", how="left"))
EV["same_line"] = EV.victim_line == SAME_LINE
EV["group"] = np.where(EV.same_line, "same line (10)", "cross line (2/8/12)")
# Cross-check against 12_geo.py: its coarse 3-class state must be the collapse of our 6-class one.
coarse = EV.carrier_state.map(lambda s: "segment" if s in MOVING else ("station" if s in STOPPED else "off"))
bad = coarse.values != EV.carrier_state_geo.values
log(f"events {len(EV)}; carrier state counts: {EV.carrier_state.value_counts().reindex(STATES).to_dict()}")
log("coarse 3-class counts: 12_geo.py " + str(geo.carrier_state.value_counts().to_dict()) +
    " vs 13 " + str(pd.Series(coarse).value_counts().to_dict()))
log(f"coarse 3-class check vs out/geo/events_geo.csv: {1 - bad.mean():.1%} agree ({int(bad.sum())} disagreements)")
if bad.any():
    log(EV.loc[bad, ["incident_id", "onset_ts", "carrier_state", "carrier_state_geo", "carrier_seg"]]
        .to_string(index=False))
EV = EV[["incident_id", "train_id", "onset_ts", "victim_line", "carrier_state", "f", "carrier_seg",
         "victim_at", "d_c", "lag_min", "same_line", "group", "victim_location_type"]]
EV.to_csv(OUT / "carrier_state_events.csv", index=False)
flush_log()

# ================================================================================================
section("4. Is the state distribution distinguishable from the carrier's time budget?")
# ================================================================================================


def state_tests(k_obs: dict, h: dict, stratum: str) -> pd.DataFrame:
    """Chi-square GOF + per-state Poisson RR (reference = cruise) + BH q, for one stratum."""
    d = pd.DataFrame({"carrier_state": STATES, "k": [int(k_obs.get(s, 0)) for s in STATES],
                      "exposure_train_h": [float(h[s]) for s in STATES]})
    d["share"] = d.exposure_train_h / d.exposure_train_h.sum()
    d["k_share"] = d.k / d.k.sum()
    d["rate_per_1000h"] = d.k / d.exposure_train_h * 1000
    ref = d.set_index("carrier_state").loc[REF]
    d["expected"] = d.k.sum() * d.share
    d["RR_vs_cruise"] = d.rate_per_1000h / ref.rate_per_1000h
    se = np.where(d.k > 0, np.sqrt(1 / d.k.clip(lower=1) + 1 / ref.k), np.nan)  # Poisson log-RR SE
    with np.errstate(divide="ignore"):
        lrr = np.log(d.RR_vs_cruise)
    d["RR_lo"] = np.where(d.k > 0, np.exp(lrr - 1.96 * se), np.nan)  # undefined with 0 events
    d["RR_hi"] = np.where(d.k > 0, np.exp(lrr + 1.96 * se), np.nan)
    d["p_state"] = [pois_two_sided(k, e) for k, e in zip(d.k, d.expected)]  # vs overall exposure share
    d["q_state"] = bh_q(d.p_state)
    d["p_rr"] = [pois_two_sided(k, hh * ref.rate_per_1000h / 1000) for k, hh in zip(d.k, d.exposure_train_h)]
    d["q_rr"] = bh_q(d.p_rr)
    chi2 = float((((d.k - d.expected) ** 2) / d.expected).sum())
    d["chi2"] = chi2
    d["chi2_df"] = len(STATES) - 1
    d["chi2_p"] = stats.chi2.sf(chi2, len(STATES) - 1)
    d["min_expected"] = d.expected.min()
    d["stratum"] = stratum
    return d


H_ALL = dict(zip(STATES, EXPO.exposure_train_h.values))
T_ALL = state_tests(EV.carrier_state.value_counts().to_dict(), H_ALL, "all")
T_ALL.to_csv(OUT / "carrier_state_tests.csv", index=False)

log("all 143 events, exposure = all other trains in service:")
log(T_ALL[["carrier_state", "k", "share", "k_share", "exposure_train_h", "rate_per_1000h", "expected",
           "RR_vs_cruise", "RR_lo", "RR_hi", "p_state", "q_state"]].round(4).to_string(index=False))
log(f"chi-square GOF: chi2={T_ALL.chi2.iat[0]:.2f}, df={int(T_ALL.chi2_df.iat[0])}, "
    f"p={T_ALL.chi2_p.iat[0]:.3g}; smallest expected count {T_ALL.min_expected.iat[0]:.1f}")

# Coarse contrast: moving vs standing at a station (off excluded), same exposure accounting.
mv = T_ALL[T_ALL.carrier_state.isin(MOVING)]
sp = T_ALL[T_ALL.carrier_state.isin(STOPPED)]
k_mv, k_sp = int(mv.k.sum()), int(sp.k.sum())
h_mv, h_sp = mv.exposure_train_h.sum(), sp.exposure_train_h.sum()
sh_mv = h_mv / (h_mv + h_sp)
p_mv = pois_two_sided(k_mv, (k_mv + k_sp) * sh_mv)
log(f"moving (depart+cruise+arrive): {k_mv} events on {sh_mv:.1%} of in-service exposure vs "
    f"standing at a station (dwell+terminal): {k_sp} events on {1 - sh_mv:.1%}; "
    f"RR={k_mv / h_mv / (k_sp / h_sp):.3f}, exact Poisson p={p_mv:.3g}")
log(f"off (carrier not in service): {int(T_ALL.loc[T_ALL.carrier_state == 'off', 'k'].iat[0])} events on "
    f"{T_ALL.loc[T_ALL.carrier_state == 'off', 'share'].iat[0]:.2%} of exposure")
flush_log()

# ---- stratified by whether the victim shares the carrier's line ------------------------------------------------
train_line = trav.groupby("train_id", observed=True).line_id.first().astype(str)
TL = np.array([train_line.get(t, "") for t in TRAINS])
STRATA = {"same line (10)": TL == SAME_LINE, "cross line (2/8/12)": TL != SAME_LINE}
expo_g, tests_g = {}, {}
for gname, mask in STRATA.items():
    n_other_g = GRID[mask].sum(axis=0, dtype=np.int32) - (GRID[CI] if mask[CI] else 0)
    expo_g[gname] = pd.Series(np.bincount(CSTATE.ravel(), weights=n_other_g.ravel().astype(np.float64),
                                          minlength=len(STATES)) * TICK_S / 3600, index=STATES)
    sub = EV[EV.group == gname]
    tests_g[gname] = state_tests(sub.carrier_state.value_counts().to_dict(), expo_g[gname].to_dict(), gname)
    log(f"\n[{gname}] n={len(sub)} events; exposure {expo_g[gname].sum():,.0f} train-h")
    log(tests_g[gname][["carrier_state", "k", "share", "k_share", "RR_vs_cruise", "RR_lo", "RR_hi",
                        "p_state", "q_state"]].round(4).to_string(index=False))
    log(f"chi2={tests_g[gname].chi2.iat[0]:.2f}, df={int(tests_g[gname].chi2_df.iat[0])}, "
        f"p={tests_g[gname].chi2_p.iat[0]:.3g}; smallest expected count {tests_g[gname].min_expected.iat[0]:.1f}")
pd.concat(tests_g.values()).to_csv(OUT / "carrier_state_tests_by_line.csv", index=False)
pd.DataFrame(expo_g).assign(carrier_state=STATES).to_csv(OUT / "carrier_state_exposure_by_line.csv",
                                                        index=False)
flush_log()

# ---- do the two strata differ? Poisson log-exposure model, interaction LR test ----------------------------------
# Reduced model: rate(cell) = h(cell) * a(group) * b(state), fitted by IPF (the Poisson MLE with a log-exposure
# offset). The saturated model fits each cell exactly, so the LR statistic is just the reduced-model deviance,
# with df = (groups-1)(states-1). IPF is used instead of a GLM fit because the same-line "off" cell has 0 events
# and a GLM hits perfect separation there.
TAB = pd.concat([t.assign(group=g) for g, t in tests_g.items()])[["group", "carrier_state", "k",
                                                                  "exposure_train_h"]]


def poisson_interaction(tab: pd.DataFrame, tag: str) -> tuple[float, int, float]:
    states = [s for s in STATES if s in set(tab.carrier_state)]
    groups = sorted(tab.group.unique())
    k = tab.pivot(index="carrier_state", columns="group", values="k").reindex(states)[groups].values.astype(float)
    h = tab.pivot(index="carrier_state", columns="group", values="exposure_train_h") \
        .reindex(states)[groups].values.astype(float)
    a, b = np.ones(len(groups)), np.ones(len(states))
    for _ in range(10_000):
        a = k.sum(axis=0) / (h * b[:, None]).sum(axis=0)
        b = k.sum(axis=1) / (h * a[None, :]).sum(axis=1)
    mu = h * a[None, :] * b[:, None]
    dev = float(2 * np.sum(np.where(k > 0, k * np.log(np.maximum(k, 1e-300) / mu), 0.0) - (k - mu)))
    df_ = (len(groups) - 1) * (len(states) - 1)
    p = float(stats.chi2.sf(dev, df_))
    log(f"state x group interaction ({tag}): LR chi2={dev:.2f}, df={df_}, p={p:.3g}")
    return dev, df_, p


LR_ALL, DF_ALL, P_INT = poisson_interaction(TAB, "all 6 states")
LR_5, DF_5, P_INT_5 = poisson_interaction(TAB[TAB.carrier_state != "off"].reset_index(drop=True),
                                          "5 in-service states (off dropped: 0 events in one stratum)")
hom = pd.crosstab(EV.carrier_state, EV.group).reindex(STATES).fillna(0)
chi_hom, p_hom, dof_hom = stats.chi2_contingency(hom.values)[:3]
log(f"raw observed 2x6 table (ignores exposure, reference only): chi2={chi_hom:.2f}, df={int(dof_hom)}, "
    f"p={p_hom:.3g}")
log("observed counts by group:\n" + hom.to_string())
pd.DataFrame([{"test": "interaction_poisson_all6", "stat": LR_ALL, "df": DF_ALL, "p": P_INT},
              {"test": "interaction_poisson_5states", "stat": LR_5, "df": DF_5, "p": P_INT_5},
              {"test": "raw_homogeneity_2x6", "stat": chi_hom, "df": int(dof_hom), "p": p_hom}]
             ).to_csv(OUT / "carrier_state_interaction.csv", index=False)
flush_log()

# ================================================================================================
section("5. Figures")
# ================================================================================================
COL = {"depart": "#8ec6e8", "cruise": "#2f7fb5", "arrive": "#14507a",
       "dwell": "#f0a35e", "terminal": "#c96a1b", "off": "#9a9a9a"}


def bar_pair(ax, tab: pd.DataFrame, title: str) -> None:
    x = np.arange(len(STATES))
    ax.bar(x - 0.2, tab.share, 0.4, color=[COL[s] for s in STATES], alpha=0.45, edgecolor="black",
           linewidth=0.5, label="share of exposure (denominator)")
    ax.bar(x + 0.2, tab.k_share, 0.4, color=[COL[s] for s in STATES], edgecolor="black", linewidth=0.5,
           label="share of events (n=%d)" % int(tab.k.sum()))
    for i, r in enumerate(tab.itertuples()):
        ax.text(i + 0.2, r.k_share + 0.012, f"k={int(r.k)}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.carrier_state}\n" + (f"RR {r.RR_vs_cruise:.2f}" if r.k > 0 else "RR n/a")
                        for r in tab.itertuples()], fontsize=8.5)
    ax.set_ylabel("share")
    ax.set_ylim(0, max(tab.k_share.max(), tab.share.max()) * 1.25)
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")


fig, ax = plt.subplots(figsize=(7.2, 4.0))
bar_pair(ax, T_ALL, f"Carrier T3420 state at onset vs exposure  (chi2={T_ALL.chi2.iat[0]:.1f}, "
                    f"df=5, p={T_ALL.chi2_p.iat[0]:.1e})")
ax.text(0.01, 0.97, "RR = rate ratio vs cruise (second line of each tick label)", transform=ax.transAxes,
        fontsize=7.5, va="top", color="#555555")
savefig(fig, "fig_carrier_state.png")

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), sharey=True)
for ax, g in zip(axes, STRATA):
    t = tests_g[g]
    bar_pair(ax, t, f"{g}: n={int(t.k.sum())}, chi2={t.chi2.iat[0]:.1f}, p={t.chi2_p.iat[0]:.1e}")
savefig(fig, "fig_carrier_state_by_line.png")
log(f"figures written: {FIG / 'fig_carrier_state.png'}, {FIG / 'fig_carrier_state_by_line.png'}")
flush_log()

# ================================================================================================
section("6. Answers (every number quoted in notes/08_carrier_state.md)")
# ================================================================================================
A = T_ALL.set_index("carrier_state")
log(f"Q1 moving vs stopped: {k_mv} events while moving ({k_mv / len(EV):.1%} of events) on {sh_mv:.1%} of "
    f"in-service exposure; {k_sp} while standing at a station ({k_sp / len(EV):.1%}) on {1 - sh_mv:.1%}; "
    f"off {int(A.loc['off', 'k'])} on {A.loc['off', 'share']:.2%}")
log(f"Q1b moving-only split: " + ", ".join(f"{s} {int(A.loc[s, 'k'])} ({A.loc[s, 'k_share']:.1%} of events, "
                                          f"{A.loc[s, 'share']:.1%} of exposure)" for s in MOVING))
log(f"Q1c stopped split: " + ", ".join(f"{s} {int(A.loc[s, 'k'])} ({A.loc[s, 'k_share']:.1%} of events, "
                                       f"{A.loc[s, 'share']:.1%} of exposure)" for s in STOPPED))
log("Q2 RR ranking (vs cruise): " + " > ".join(
    f"{s} {A.loc[s, 'RR_vs_cruise']:.2f} [{A.loc[s, 'RR_lo']:.2f}, {A.loc[s, 'RR_hi']:.2f}]"
    for s in A.RR_vs_cruise.sort_values(ascending=False).index))
log(f"Q3 significance: chi2={T_ALL.chi2.iat[0]:.2f}, df={int(T_ALL.chi2_df.iat[0])}, p={T_ALL.chi2_p.iat[0]:.3g}; "
    "BH q: " + ", ".join(f"{s} {A.loc[s, 'q_state']:.3g}" for s in STATES))
log(f"Q3b smallest expected count {T_ALL.min_expected.iat[0]:.1f} (state {A.expected.idxmin()})")
log(f"Q4 same vs cross: per-stratum chi2 p = " + ", ".join(
    f"{g} {tests_g[g].chi2_p.iat[0]:.3g}" for g in STRATA) +
    f"; interaction LR={LR_ALL:.2f} df={DF_ALL} p={P_INT:.3g} (5 states: LR={LR_5:.2f} df={DF_5} p={P_INT_5:.3g})")
for g in STRATA:
    t = tests_g[g].set_index("carrier_state")
    log(f"Q4b {g}: " + ", ".join(f"{s} {int(t.loc[s, 'k'])}/{t.loc[s, 'k_share']:.1%}ev/"
                                 f"{t.loc[s, 'share']:.1%}ex" for s in STATES))
log(f"Q5 rare-but-risky states (low exposure share, high event share): " + ", ".join(
    f"{s} RR={A.loc[s, 'RR_vs_cruise']:.2f}, exposure {A.loc[s, 'share']:.2%} vs events "
    f"{A.loc[s, 'k_share']:.1%}" for s in ["arrive", "dwell"]))
log(f"Q5b opposite (common but safe): " + ", ".join(
    f"{s} RR={A.loc[s, 'RR_vs_cruise']:.2f}, exposure {A.loc[s, 'share']:.1%} vs events {A.loc[s, 'k_share']:.1%}"
    for s in ["terminal", "cruise"]))
log(f"Q5c terminal layovers: {int((CSTATE == TERM).sum()):,} ticks = "
    f"{(CSTATE == TERM).sum() * TICK_S / 3600:,.0f} carrier-h ({EXPO.loc[EXPO.carrier_state == 'terminal', 'carrier_share'].iat[0]:.1%} of the carrier's service day), "
    f"{int(A.loc['terminal', 'k'])} events")
same = tests_g["same line (10)"].set_index("carrier_state")
log(f"Q4c same-line concentration: cruise+arrive = {int(same.loc['cruise', 'k'] + same.loc['arrive', 'k'])} "
    f"of {int(same.k.sum())} = {(same.loc['cruise', 'k'] + same.loc['arrive', 'k']) / same.k.sum():.1%}; "
    f"cross-line dwell = {int(tests_g['cross line (2/8/12)'].set_index('carrier_state').loc['dwell', 'k'])} "
    f"of {int(tests_g['cross line (2/8/12)'].k.sum())}")
log(f"Q5e event/exposure ratio: " + ", ".join(f"{s} {A.loc[s, 'k_share'] / A.loc[s, 'share']:.2f}x" for s in STATES))
log(f"Q5f terminal events by group: {hom.loc['terminal'].to_dict()}; "
    f"off events by group: {hom.loc['off'].to_dict()}")
log(f"Q5g the single genuinely-off incident: "
    f"{EV.loc[EV.carrier_state == 'off', ['incident_id', 'onset_ts', 'victim_line', 'victim_at']].to_dict('records')}")
log(f"Q5d median f for moving events: {EV.loc[EV.carrier_state.isin(MOVING), 'f'].median():.3f}; "
    f"median d_c {EV.d_c.median():.2f}; median lag_min {EV.lag_min.median():.2f}")
flush_log()

flush_log()
