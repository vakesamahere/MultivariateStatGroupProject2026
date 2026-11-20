"""Is the empty 250-600 m band a risk fact or a timetable fact?

Purpose
-------
`17_distance_groups.py` reads the 142 breakdowns as two channels with almost disjoint
distance supports: on the carrier's own line 62 of 68 events sit inside 250 m and none
between 250 m and 600 m (against 383.7 train-h of exposure there); off the line 73 of 74
events sit beyond 600 m and none inside 250 m (against 123.7 train-h).  The paper turns
that into "the pooled smooth kernel is a mixture of two configurations".

That reading has a rival that has nothing to do with risk: the timetable.  The data are
semi-synthetic with a deterministic schedule, 30 s dwells and a moving block, so two
trains of the same line may simply never be 250-600 m apart -- either almost co-located
(one passing the other, a window of seconds) or at least one station apart (a segment is
588.7 m at the shortest).  If so, "disjoint supports" is a property of the schedule, and
the on-line spike is a statement about where same-line trains can be, not about danger.

This script is the falsification test: it maps the FULL 30 s grid distribution of
same-line train-pair distances to the carrier, checks whether the 250-600 m band is ever
occupied, and then recomputes the on-line breakdown rate across only the distance bands
the timetable actually populates.  That last number decides the question.

Method
------
1. Every tick of the 30 s grid of every service date (not the 5-minute sample of
   `14_rewind_dist.py`), the straight-line distance from T3420 to every other in-service
   train, labelled same line (Line 10, 10B branch included) or off line (2, 8, 12), is
   histogrammed in 25 m bins to 3 km and 100 m bins to 5 km.  Chunked by date.
2. The same pairs in along-track (network) distance, with the same binning.
3. The 68 on-line and 74 off-line events are overlaid on those histograms.
4. Why a band would be empty: the smallest same-line separation observed, the scheduled
   headway between consecutive same-line departures from the same station and the metres
   it implies, the shortest segment length, and a 1 s refinement of every pair-tick that
   comes within 1 km, which separates "structurally unoccupiable" from "crossed in 20 s
   and missed by the 30 s grid".
5. The decisive comparison: breakdowns per 1000 train-hours in each distance band the
   timetable actually populates, with exact Poisson CIs, on the full-grid exposure.
6. Robustness: along-track distance, isolated events only (the 23 network-wide bursts of
   `02_screening.py` dropped), and the six terminal-layover events dropped.

Exposure
--------
One row of the grid = one train x 30 s = 1/120 train-hour of that train being at that
distance from the carrier.  The risk set is the one `14_rewind_dist.py` and
`17_distance_groups.py` use: every train inside a segment or a 30 s dwell, plus the
carrier's own terminal layovers (its 6 terminal-station breakdowns need them).

Outputs
-------
out/support_gap/hist_25m.csv, hist_100m.csv      straight-line histograms, same / off line
out/support_gap/hist_25m_network.csv             along-track histograms
out/support_gap/per_date.csv                     the per-date chunk log of the day loop
out/support_gap/events_bands.csv                 every event with its band in both metrics
out/support_gap/rate_populated.csv               the decisive table + its robustness variants
out/support_gap/refine_1s.csv                    sub-tick occupancy of the near field
out/support_gap/headway.csv                      scheduled same-line headway per station
out/support_gap/support_gap_results.txt          the full log (console shows <= 15 lines)
"""
from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
GEO = ROOT / "out" / "geo"
OUT = ROOT / "out" / "support_gap"
OUT.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT / "support_gap_results.txt"
CARRIER = "T3420"
HOURS_PER_ROW = 1.0 / 120.0  # one 30 s tick of one train
# 25 m bins to 3 km (the brief's fine grid) and 100 m bins to 5 km (the coarse one); both start at 0.
FINE_EDGES = np.arange(0.0, 3000.0 + 25.0, 25.0)
COARSE_EDGES = np.arange(0.0, 5000.0 + 100.0, 100.0)
# The off-line channel lives at kilometres, so the rate table needs exposure all the way out; 500 m bins to 50 km.
TAIL_EDGES = np.arange(0.0, 50000.0 + 500.0, 500.0)
GAP_LO, GAP_HI = 250.0, 600.0  # the band whose occupancy is the whole question
CHUNK_DAYS = 15  # save the accumulators every this many dates
NEAR_M = 1000.0  # pairs closer than this get the 1 s refinement of section 4
REFINE_HALF_S = 15  # half-width of a refinement window: half a tick, so the windows tile
THIN_H = 20.0  # a rate built on less exposure than this is flagged wherever it appears
SHORTEST_SEG_HINT = 588.717401  # Line 10 S079-S080, from out/geo/segment_lengths.csv (asserted below)

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
    """Exact Poisson (Garwood) 95% interval for a rate per 1000 train-hours, same estimator as 12/17."""
    k = np.asarray(k, dtype=float)
    hours = np.maximum(np.asarray(hours, dtype=float), 1e-12)
    lo = np.where(k > 0, stats.chi2.ppf(0.025, 2 * np.maximum(k, 1)) / 2.0, 0.0)
    hi = stats.chi2.ppf(0.975, 2 * k + 2) / 2.0
    return lo / hours * 1000.0, hi / hours * 1000.0


def load_locator():
    """Exec the part of 12_geo.py above section 5 in a scratch root, so out/geo/ is never rewritten."""
    src_path = SRC / "12_geo.py"
    text = src_path.read_text(encoding="utf-8")
    cut = text.index('section("5. Exposure on the 30-s grid')
    scratch = Path(tempfile.mkdtemp(prefix="supportgap_geo_"))
    (scratch / "src").mkdir()
    (scratch / "out" / "geo").mkdir(parents=True)
    (scratch / "out" / "interchange").mkdir(parents=True)
    for link, target in {
        scratch / "data": ROOT / "data",
        scratch / "data_derived": ROOT / "data_derived",
        scratch / "out" / "geo" / "osm_raw_rel2.json": GEO / "osm_raw_rel2.json",
        scratch / "out" / "interchange" / "events.csv": ROOT / "out" / "interchange" / "events.csv",
    }.items():
        link.symlink_to(target)
    ns = {"__file__": str(scratch / "src" / "12_geo.py"), "__name__": "geo12_for_supportgap"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(text[:cut], str(src_path), "exec"), ns)
    shutil.rmtree(scratch)
    return ns


NS = load_locator()
locate, haversine = NS["locate"], NS["haversine"]
trav = NS["trav"]
DEP, ARR, DWELL = NS["DEP"], NS["ARR"], NS["DWELL"]
FROM, TO = NS["FROM"], NS["TO"]
NODE, LAT, LON = NS["NODE"], NS["LAT"], NS["LON"]
NET, spd = NS["NET"], NS["spd"]
U, V = np.array([NODE[s] for s in FROM], dtype=np.int16), np.array([NODE[s] for s in TO], dtype=np.int16)
TC = trav.train_id.cat.codes.values.astype(np.int16)
N_TRAIN = len(trav.train_id.cat.categories)
DEP_S, ARR_S = NS["sec_of_day"](trav.dep_ts, trav.date), NS["sec_of_day"](trav.arr_ts, trav.date)
T0 = float(np.floor(np.nanmin(DEP_S) / 3600) * 3600)
TICK_S = NS["TICK_S"]
NT = int(np.ceil((np.nanmax(ARR_S) - T0) / TICK_S)) + 1
TCODES, DCODES = NS["TCODES"], NS["DCODES"]
T_ARR, T_DEP, T_ORDER, T_COMBO_S = NS["T_ARR"], NS["T_DEP"], NS["T_ORDER"], NS["T_COMBO_S"]
CAR_T_START_S, CAR_T_END_S = NS["CAR_T_START_S"], NS["CAR_T_END_S"]
CAR_T_NODE, CAR_T_DCODE = NS["CAR_T_NODE"], NS["CAR_T_DCODE"]
CI = TCODES[CARRIER]
TRAV_DIR = trav.direction.cat.codes.values.astype(np.int8)
TRAV_LINE10 = (trav.line_id.astype(str) == "10").values.astype(bool)
DATES = NS["DATES"]
DCODE_ARR = pd.Categorical(trav.date, categories=DATES).codes
TRAIN_NAMES = list(trav.train_id.cat.categories)

# The 143 breakdowns, with the burst label from 02_screening so the isolated-event robustness check can be run.
EV = pd.read_csv(GEO / "events_geo.csv", parse_dates=["onset_ts"])
_scr = pd.read_csv(ROOT / "figures" / "screening" / "tables" / "breakdown_events.csv", parse_dates=["onset"])
EV = EV.merge(_scr[["train_id", "onset", "in_burst", "burst_id"]],
              left_on=["train_id", "onset_ts"], right_on=["train_id", "onset"], how="left")
assert EV.in_burst.notna().all() and len(EV) == 143
EV["is_online"] = EV.line_id == 10
EV["ev_date"] = pd.to_datetime(EV.onset_ts).dt.date.astype(str)
EV_DCODE = pd.Categorical(EV.ev_date, categories=DATES).codes
assert (EV_DCODE >= 0).all(), "an event falls on a date outside the traversal panel"
EV_TS = NS["sec_of_day"](EV.onset_ts, pd.to_datetime(EV.ev_date))
EV_TRAIN = np.array([TRAIN_NAMES.index(t) for t in EV.train_id], dtype=np.int16)
EV_DS = np.full(len(EV), np.nan)  # straight distance at the nearest tick, recomputed on our own grid
EV_DN = np.full(len(EV), np.nan)  # network distance, with the same-segment case done by direction


def interp_pos(i, k0, k1, w, u_g, v_g, plat, plon):
    """Position of train(s) i at fractional tick w between ticks k0 and k1 (arrays broadcast).

    Linear interpolation is exact only while a train stays on one segment; where it changes segment between the
    two ticks the interpolated point would cut the corner through the station, so fall back to the nearer tick.
    """
    keep = (u_g[i, k0] == u_g[i, k1]) & (v_g[i, k0] == v_g[i, k1])
    near = np.where(w < 0.5, k0, k1)
    w_k = np.where(keep, w, np.where(near == k0, 0.0, 1.0))
    return (plat[i, k0] * (1 - w_k) + plat[i, k1] * w_k,
            plon[i, k0] * (1 - w_k) + plon[i, k1] * w_k)


def frac_state(i, k, w, u_g, v_g, f_g, len_g, dir_g):
    """(u, v, f, length, direction) of train(s) i at fractional tick k+w, held on one segment.

    Only used for the event geometry, so that an onset is placed at its own time instead of at the nearest tick: two
    same-line trains can be exactly co-located at one tick and 80 m apart 15 s later, which would otherwise show up
    as a spurious zero distance for every near-field event.
    """
    k1 = np.minimum(k + 1, NT - 1)
    keep = (u_g[i, k] == u_g[i, k1]) & (v_g[i, k] == v_g[i, k1]) & (u_g[i, k] != v_g[i, k])
    kk = np.where(keep, k, np.where(w < 0.5, k, k1))
    ff = np.where(keep, f_g[i, k] + w * (f_g[i, k1] - f_g[i, k]), f_g[i, kk])
    return u_g[i, kk], v_g[i, kk], ff, len_g[i, kk], dir_g[i, kk]

# Scheduled running time and segment length per traversal, keyed like 14_rewind_dist.py: the denominator of the
# position fraction is min(recorded spell, scheduled running time), so a breakdown spell does not drag the train
# backwards, and the length is what the along-track metric needs.
_plan = (pd.DataFrame({"route_id": trav.route_id.astype(str), "key": trav.segment.astype(str).map(NS["undirected"])})
         .merge(NS["spd"][["route_id", "key", "travel_s_med", "length_m"]].assign(
             route_id=lambda d: d.route_id.astype(str)), on=["route_id", "key"], how="left"))
PLAN_S = _plan.travel_s_med.values.astype(float)
LEN_S = _plan.length_m.values.astype(float)
DUR_S = (ARR - DEP) / 1e9
DEN_S = np.where(np.isfinite(PLAN_S), np.minimum(DUR_S, PLAN_S), DUR_S)
DEN_S = np.where(DEN_S > 0, DEN_S, 1.0)
assert np.isclose(np.nanmin(LEN_S[trav.line_id.astype(str) == "10"]), SHORTEST_SEG_HINT, atol=1e-4), \
    "the shortest Line 10 segment moved; the section-4 comparison is stated against 588.7 m"


def tick_ts(day, tick: int):
    """Timestamp of grid tick `tick` on `day` (the grid is T0 + k * TICK_S seconds after midnight)."""
    return pd.Timestamp(day) + pd.Timedelta(seconds=float(T0 + tick * TICK_S))


# ================================================================================================
section("1. The full 30 s grid: distance from T3420 to every in-service train, same line vs off line")
# ================================================================================================
# 14_rewind_dist.py keeps every 10th tick (a 5-minute background sample).  This is the same day loop with stride 1,
# so the exposure is the whole grid, plus two things 14 does not need: the segment length of every train-tick (for the
# along-track metric) and a 1 s refinement of the near field (section 4).  The 5-minute sample is the reason 17 saw
# 0 rows in the 50-150 m same-line bins; the full grid is what tells us whether that band is empty or merely sparse.
log(f"grid: {N_TRAIN} trains x {len(DATES)} dates x {NT} ticks of {TICK_S:.0f} s from {T0 / 3600:.1f}:00; "
    f"line-10 trains {int((trav.groupby('train_id', observed=True).line_id.first().astype(str) == '10').sum())}, "
    f"off-line trains "
    f"{int((trav.groupby('train_id', observed=True).line_id.first().astype(str) != '10').sum())}; "
    f"full grid = {N_TRAIN * NT * len(DATES) / 1e6:.1f}M train-ticks at most (minus out-of-service ticks)")

H_SAME_25 = np.zeros(FINE_EDGES.size - 1, dtype=np.int64)
H_CROSS_25 = np.zeros(FINE_EDGES.size - 1, dtype=np.int64)
H_SAME_100 = np.zeros(COARSE_EDGES.size - 1, dtype=np.int64)
H_CROSS_100 = np.zeros(COARSE_EDGES.size - 1, dtype=np.int64)
H_SAME_TAIL = np.zeros(TAIL_EDGES.size - 1, dtype=np.int64)
H_CROSS_TAIL = np.zeros(TAIL_EDGES.size - 1, dtype=np.int64)
H_SAME_25_NET = np.zeros(FINE_EDGES.size - 1, dtype=np.int64)
H_CROSS_25_NET = np.zeros(FINE_EDGES.size - 1, dtype=np.int64)
REF_EDGES = np.arange(0.0, NEAR_M + 25.0, 25.0)  # the 1 s refinement covers the near field only
H_SAME_1S = np.zeros(REF_EDGES.size - 1, dtype=np.int64)
H_CROSS_1S = np.zeros(REF_EDGES.size - 1, dtype=np.int64)
stat = {"min_same_m": np.inf, "min_same_tick": None, "min_net_same_m": np.inf, "min_same_1s_m": np.inf,
        "n_corr_net": 0, "n_corr_net_big": 0, "n_ref_pairs": 0, "n_ref_samples": 0, "n_ref_invalid": 0,
        "n50": 0, "n50_persist": 0, "n50_sameseg": 0, "n50_opp": 0}
# The 25 m / 100 m histograms stop at 3 km / 5 km; these two counters keep the whole distribution (the far tail is
# where the off-line channel lives, and 17's exposure totals are checked against them).
N_SAME_ALL = 0
N_CROSS_ALL = 0
per_date: list[dict] = []


def net_dist(uc, vc, fc, lc, uu, vv, ff, ll, dirs):
    """Along-track distance from the carrier to every train, metres.

    12_geo.network_m takes the minimum of four end-to-end paths.  For two positions on the same piece of track that
    overestimates an opposite-direction pair by up to a segment length (the two trains are inside one segment, not at
    its ends), so the same-segment case is computed directly -- with the direction, which 12_geo's version does not
    have -- and the end-to-end paths are kept as a fallback.  For same-direction co-segment pairs the two agree.
    """
    best = np.full(uu.shape, np.inf)
    for an, pa in ((uc, fc * lc), (vc, (1.0 - fc) * lc)):
        for bn, pb in ((uu, ff * ll), (vv, (1.0 - ff) * ll)):
            best = np.minimum(best, pa[None, :] + NET[an[None, :], bn] + pb)
    su, sv = np.minimum(uu, vv), np.maximum(uu, vv)
    cu, cv = np.minimum(uc, vc), np.maximum(uc, vc)
    same_seg = (su == cu[None, :]) & (sv == cv[None, :]) & (su != sv)  # uu == -1 fails both equalities
    d_same_dir = np.abs(fc[None, :] - ff) * lc[None, :]
    d_opp = np.abs(fc[None, :] + ff - 1.0) * lc[None, :]
    d_direct = np.where(dirs, d_same_dir, d_opp)
    stat["n_corr_net"] += int((same_seg & (d_direct < best - 1e-9)).sum())
    stat["n_corr_net_big"] += int((same_seg & (d_direct < best - 100.0)).sum())
    return np.where(same_seg, np.minimum(best, d_direct), best)


for di, day in enumerate(DATES):
    sel = DCODE_ARR == di
    u_g = np.full((N_TRAIN, NT), -1, dtype=np.int16)
    v_g = np.full((N_TRAIN, NT), -1, dtype=np.int16)
    f_g = np.zeros((N_TRAIN, NT), dtype=np.float64)
    len_g = np.zeros((N_TRAIN, NT), dtype=np.float32)
    dir_g = np.full((N_TRAIN, NT), -1, dtype=np.int8)
    lin_g = np.zeros((N_TRAIN, NT), dtype=bool)
    rows = np.flatnonzero(sel)
    a = np.ceil((DEP_S[rows] - T0) / TICK_S).astype(np.int64)
    b = np.ceil((ARR_S[rows] - T0) / TICK_S).astype(np.int64)
    n_t = np.maximum(b - a, 0)
    rep_local = np.repeat(np.arange(rows.size), n_t)
    rep = rows[rep_local]
    tick = a[rep_local] + (np.arange(n_t.sum()) - np.repeat(np.cumsum(n_t) - n_t, n_t))
    assert tick.size == 0 or (tick.min() >= 0 and tick.max() < NT)
    f = np.clip((tick * TICK_S + T0 - DEP_S[rep]) / DEN_S[rep], 0.0, 1.0)
    u_g[TC[rep], tick], v_g[TC[rep], tick], f_g[TC[rep], tick] = U[rep], V[rep], f
    len_g[TC[rep], tick] = LEN_S[rep]
    dir_g[TC[rep], tick] = TRAV_DIR[rep]
    lin_g[TC[rep], tick] = TRAV_LINE10[rep]
    del rep, rep_local, tick, f, n_t, a, b
    dw = np.flatnonzero(np.isfinite(DWELL) & sel)  # station dwell intervals [dep - dwell, dep)
    a = np.ceil((DEP_S[dw] - DWELL[dw] - T0) / TICK_S).astype(np.int64)
    b = np.ceil((DEP_S[dw] - T0) / TICK_S).astype(np.int64)
    n_t = np.maximum(b - a, 0)
    rep_local = np.repeat(np.arange(dw.size), n_t)
    rep = dw[rep_local]
    tick = a[rep_local] + (np.arange(n_t.sum()) - np.repeat(np.cumsum(n_t) - n_t, n_t))
    u_g[TC[rep], tick], v_g[TC[rep], tick], f_g[TC[rep], tick] = U[rep], U[rep], 0.0
    dir_g[TC[rep], tick] = TRAV_DIR[rep]
    lin_g[TC[rep], tick] = TRAV_LINE10[rep]
    del rep, rep_local, tick, n_t, a, b, dw
    # The carrier's terminal layovers carry a position (12_geo section 4/5 rule); other trains' layovers do not.
    tw = np.flatnonzero(CAR_T_DCODE == di)
    a = np.ceil((CAR_T_START_S[tw] - T0) / TICK_S).astype(np.int64)
    b = np.ceil((CAR_T_END_S[tw] - T0) / TICK_S).astype(np.int64)
    n_t = np.maximum(b - a, 0)
    rep_local = np.repeat(np.arange(tw.size), n_t)
    tick = a[rep_local] + (np.arange(n_t.sum()) - np.repeat(np.cumsum(n_t) - n_t, n_t))
    u_g[CI, tick], v_g[CI, tick], f_g[CI, tick] = CAR_T_NODE[tw][rep_local], CAR_T_NODE[tw][rep_local], 0.0
    del tw, rep_local, tick, n_t, a, b

    valid = u_g >= 0
    uu, vv, ff, ll = np.maximum(u_g, 0), np.maximum(v_g, 0), f_g, np.where(valid, len_g, 0.0)
    plat = np.where(valid, LAT[uu] + ff * (LAT[vv] - LAT[uu]), np.nan)
    plon = np.where(valid, LON[uu] + ff * (LON[vv] - LON[uu]), np.nan)
    con = valid[CI]
    d = haversine(plat[CI][None, :], plon[CI][None, :], plat, plon)
    d[CI] = np.nan
    d[:, ~con] = np.nan
    # Event geometry on our own grid, at the tick nearest each onset and with both metrics, so the event distances
    # and the exposure denominators come from the same code.  (events_geo.network_m uses 12_geo's same-segment
    # formula |f1-f2|*L, which is wrong for opposing co-segment pairs -- see the robustness section.)
    ev_d = np.flatnonzero(EV_DCODE == di)
    if ev_d.size:
        raw = (EV_TS[ev_d] - T0) / TICK_S
        k = np.clip(np.floor(raw).astype(np.int64), 0, NT - 2)
        w = np.clip(raw - k, 0.0, 1.0)
        k1 = k + 1
        iv = EV_TRAIN[ev_d]
        okv = valid[iv, k] & valid[iv, k1] & con[k] & con[k1]
        la, lo_ = interp_pos(iv, k, k1, w, u_g, v_g, plat, plon)
        lc_, oc_ = interp_pos(np.full(iv.size, CI), k, k1, w, u_g, v_g, plat, plon)
        EV_DS[ev_d] = np.where(okv, haversine(lc_, oc_, la, lo_), np.nan)
        _cc = np.full(iv.size, CI)
        cu, cv_, cf, cl, cd = frac_state(_cc, k, w, u_g, v_g, f_g, len_g, dir_g)
        vu, vv_, vf, vl, vd = frac_state(iv, k, w, u_g, v_g, f_g, len_g, dir_g)
        EV_DN[ev_d] = np.where(okv, net_dist(cu, cv_, cf, cl, vu, vv_, vf, vl, vd == cd), np.nan)
    ok_same = valid & lin_g & con[None, :]
    ok_same[CI] = False
    ok_cross = valid & ~lin_g & con[None, :]
    ok_cross[CI] = False  # the carrier is not a pair with itself: its own row is NaN at layover ticks too
    ds, dx = d[ok_same], d[ok_cross]
    N_SAME_ALL += int(ds.size)
    N_CROSS_ALL += int(dx.size)
    # How much of the <50 m mass is a sustained co-location and how much is a single-tick coincidence?  This is the
    # question that decides whether the 30 s grid's 0 m spike is real occupancy or sampling phase.
    near50 = ok_same & (d < 50.0)
    near50_p = near50 & (np.roll(near50, 1, axis=1) | np.roll(near50, -1, axis=1))
    same_seg = (uu == uu[CI][None, :]) & (vv == vv[CI][None, :])
    stat["n50"] += int(near50.sum())
    stat["n50_persist"] += int(near50_p.sum())
    stat["n50_sameseg"] += int((near50 & same_seg).sum())
    stat["n50_opp"] += int((near50 & ~(dir_g == dir_g[CI][None, :])).sum())
    H_SAME_25 += np.histogram(ds, FINE_EDGES)[0]
    H_CROSS_25 += np.histogram(dx, FINE_EDGES)[0]
    H_SAME_100 += np.histogram(ds, COARSE_EDGES)[0]
    H_CROSS_100 += np.histogram(dx, COARSE_EDGES)[0]
    H_SAME_TAIL += np.histogram(ds, TAIL_EDGES)[0]
    H_CROSS_TAIL += np.histogram(dx, TAIL_EDGES)[0]
    dn = net_dist(uu[CI], vv[CI], ff[CI], ll[CI], uu, vv, ff, ll, dir_g == dir_g[CI][None, :])
    dn[~valid] = np.nan  # an out-of-service train sits at node 0 in the arrays above, not on the track
    dn[CI] = np.nan
    dn[:, ~con] = np.nan
    dns, dnx = dn[ok_same], dn[ok_cross]
    H_SAME_25_NET += np.histogram(dns, FINE_EDGES)[0]
    H_CROSS_25_NET += np.histogram(dnx, FINE_EDGES)[0]
    if ds.size:
        j = int(np.argmin(ds))
        if ds[j] < stat["min_same_m"]:
            stat["min_same_m"] = float(ds[j])
            stat["min_same_tick"] = (str(pd.Timestamp(day).date()), int(np.nonzero(ok_same)[1][j]),
                                     TRAIN_NAMES[int(np.nonzero(ok_same)[0][j])])
    if dns.size:
        stat["min_net_same_m"] = min(stat["min_net_same_m"], float(np.nanmin(dns)))
    # 1 s refinement: every same-line pair-tick closer than NEAR_M is re-sampled at 1 s over the half-tick windows
    # that bracket it, so the occupancy of a band crossed at speed is measured rather than inferred.
    near = ok_same & (d < NEAR_M)
    ti, tr = np.nonzero(near)
    if ti.size:
        stat["n_ref_pairs"] += int(ti.size)
        for s in range(-REFINE_HALF_S, REFINE_HALF_S):
            if s < 0:
                k0, k1, w = tr - 1, tr, np.full(tr.size, (TICK_S + s) / TICK_S)
            else:
                k0, k1, w = tr, tr + 1, np.full(tr.size, s / TICK_S)
            good = (k0 >= 0) & (k1 < NT)
            good[good] &= valid[ti[good], k0[good]] & valid[ti[good], k1[good]] & con[np.clip(k0, 0, NT - 1)] \
                & con[np.clip(k1, 0, NT - 1)]
            stat["n_ref_invalid"] += int((~good).sum())
            if not good.any():
                continue
            i_g, a0, a1, ww = ti[good], k0[good], k1[good], w[good]
            la, lo_ = interp_pos(i_g, a0, a1, ww, u_g, v_g, plat, plon)
            lc_, oc_ = interp_pos(np.full(i_g.size, CI), a0, a1, ww, u_g, v_g, plat, plon)
            dd = haversine(lc_, oc_, la, lo_)
            H_SAME_1S += np.histogram(dd, REF_EDGES)[0]
            stat["n_ref_samples"] += int(dd.size)
            stat["min_same_1s_m"] = min(stat["min_same_1s_m"], float(dd.min()))
    per_date.append({"date": str(pd.Timestamp(day).date()), "carrier_ticks": int(con.sum()),
                     "same_pairs": int(ds.size), "cross_pairs": int(dx.size),
                     "same_in_250_600": int(((ds >= GAP_LO) & (ds < GAP_HI)).sum()),
                     "same_lt250": int((ds < GAP_LO).sum()), "min_same_m": float(ds.min()) if ds.size else np.nan,
                     "ref_pairs": int(ti.size)})
    del u_g, v_g, f_g, len_g, dir_g, lin_g, valid, uu, vv, ff, ll, plat, plon, d, dn, ok_same, ok_cross
    if (di + 1) % CHUNK_DAYS == 0 or di == len(DATES) - 1:
        np.savez_compressed(OUT / "hist_partial.npz", same25=H_SAME_25, cross25=H_CROSS_25,
                            same100=H_SAME_100, cross100=H_CROSS_100, same25net=H_SAME_25_NET,
                            cross25net=H_CROSS_25_NET, same1s=H_SAME_1S, cross1s=H_CROSS_1S)
        pd.DataFrame(per_date).to_csv(OUT / "per_date.csv", index=False)
        log(f"  [{di + 1}/{len(DATES)} dates] same-line pair-ticks so far {N_SAME_ALL:,} "
            f"({N_SAME_ALL * HOURS_PER_ROW:,.1f} train-h), "
            f"in {GAP_LO:.0f}-{GAP_HI:.0f} m "
            f"{int(H_SAME_25[(FINE_EDGES[:-1] >= GAP_LO) & (FINE_EDGES[:-1] < GAP_HI)].sum()):,}; "
            f"1 s samples {stat['n_ref_samples']:,}")

PD = pd.DataFrame(per_date)
log(f"same-line pair-ticks {N_SAME_ALL:,} ({N_SAME_ALL * HOURS_PER_ROW:,.1f} train-h), "
    f"off-line {N_CROSS_ALL:,} ({N_CROSS_ALL * HOURS_PER_ROW:,.1f} train-h) over {len(DATES)} dates; "
    f"inside 3 km: same {H_SAME_25.sum():,} ({H_SAME_25.sum() * HOURS_PER_ROW:,.1f} h), "
    f"off {H_CROSS_25.sum():,} ({H_CROSS_25.sum() * HOURS_PER_ROW:,.1f} h)")
log(f"smallest same-line separation on the 30 s grid: {stat['min_same_m']:.1f} m "
    f"(date {stat['min_same_tick'][0]}, tick {stat['min_same_tick'][1]}, train {stat['min_same_tick'][2]}); "
    f"1 s refinement minimum {stat['min_same_1s_m']:.1f} m; along-track minimum {stat['min_net_same_m']:.1f} m")
log(f"along-track metric: the same-segment direct case was smaller than the four end-to-end paths for "
    f"{stat['n_corr_net']:,} pair-ticks ({stat['n_corr_net_big']:,} of them by more than 100 m) -- these are exactly "
    f"the co-located pairs, so the correction matters for the near field and nowhere else")
log(f"1 s refinement: {stat['n_ref_pairs']:,} near pair-ticks x {2 * REFINE_HALF_S} samples = "
    f"{stat['n_ref_samples']:,} evaluated positions; {stat['n_ref_invalid']:,} dropped at a service boundary")
log(f"the <50 m mass: {stat['n50']:,} same-line pair-ticks, {stat['n50_persist']:,} of them "
    f"({100.0 * stat['n50_persist'] / max(stat['n50'], 1):.1f}%) with a neighbouring tick also under 50 m; "
    f"{stat['n50_sameseg']:,} on the carrier's own segment, {stat['n50_opp']:,} opposing direction")
flush_log()


# ================================================================================================
section("2. The histograms: is the 250-600 m band occupied, and does the off-line one have the same gap?")
# ================================================================================================
# The bands used from here on: 17's seven shells, split at 50/100/150/250/400/600 m in the near field so the
# 250-600 m band that 17 rests on is one cell rather than the tail of a 0-500 m cell.
AN_EDGES = np.array([0.0, 50.0, 100.0, 150.0, 250.0, 400.0, 600.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0, np.inf])
AN_LABELS = [f"{lo:.0f}-{hi:.0f}" if np.isfinite(hi) else f"{lo:.0f}+" for lo, hi in zip(AN_EDGES[:-1], AN_EDGES[1:])]

hist25 = pd.DataFrame({"band_lo_m": FINE_EDGES[:-1], "band_hi_m": FINE_EDGES[1:],
                       "same_ticks": H_SAME_25, "same_h": H_SAME_25 * HOURS_PER_ROW,
                       "off_ticks": H_CROSS_25, "off_h": H_CROSS_25 * HOURS_PER_ROW})
hist25.to_csv(OUT / "hist_25m.csv", index=False)
hist100 = pd.DataFrame({"band_lo_m": COARSE_EDGES[:-1], "band_hi_m": COARSE_EDGES[1:],
                        "same_ticks": H_SAME_100, "same_h": H_SAME_100 * HOURS_PER_ROW,
                        "off_ticks": H_CROSS_100, "off_h": H_CROSS_100 * HOURS_PER_ROW})
hist100.to_csv(OUT / "hist_100m.csv", index=False)
hist25n = pd.DataFrame({"band_lo_m": FINE_EDGES[:-1], "band_hi_m": FINE_EDGES[1:],
                        "same_ticks": H_SAME_25_NET, "same_h": H_SAME_25_NET * HOURS_PER_ROW,
                        "off_ticks": H_CROSS_25_NET, "off_h": H_CROSS_25_NET * HOURS_PER_ROW})
hist25n.to_csv(OUT / "hist_25m_network.csv", index=False)


def _cum(edges, counts, x) -> int:
    """Pair-ticks in a histogram with edge value < x (x is always on an edge, so this is exact)."""
    return int(counts[:int(np.searchsorted(edges, float(x), side="left"))].sum())


def cum_ticks(x, h25, h100, htail) -> int:
    """Pair-ticks with distance < x, assembled from the 25 m, 100 m and 500 m grids on their own ranges."""
    x = min(float(x), float(TAIL_EDGES[-1]))
    return (_cum(FINE_EDGES, h25, min(x, FINE_EDGES[-1]))
            + _cum(COARSE_EDGES, h100, min(x, COARSE_EDGES[-1])) - _cum(COARSE_EDGES, h100, min(x, FINE_EDGES[-1]))
            + _cum(TAIL_EDGES, htail, x) - _cum(TAIL_EDGES, htail, min(x, COARSE_EDGES[-1])))


def band_ticks(edges, h25, h100, htail) -> np.ndarray:
    c = np.array([cum_ticks(e, h25, h100, htail) for e in edges])
    return np.diff(c)


BAND = pd.DataFrame({"band": AN_LABELS,
                     "same_ticks": band_ticks(AN_EDGES, H_SAME_25, H_SAME_100, H_SAME_TAIL),
                     "off_ticks": band_ticks(AN_EDGES, H_CROSS_25, H_CROSS_100, H_CROSS_TAIL),
                     "same_ticks_net": band_ticks(AN_EDGES, H_SAME_25_NET, H_SAME_100, H_SAME_TAIL),
                     "off_ticks_net": band_ticks(AN_EDGES, H_CROSS_25_NET, H_CROSS_100, H_CROSS_TAIL)})
BAND["same_h"] = BAND.same_ticks * HOURS_PER_ROW
BAND["off_h"] = BAND.off_ticks * HOURS_PER_ROW
BAND["same_h_net"] = BAND.same_ticks_net * HOURS_PER_ROW
BAND["off_h_net"] = BAND.off_ticks_net * HOURS_PER_ROW
BAND.to_csv(OUT / "bands.csv", index=False)
assert BAND.same_ticks.sum() == N_SAME_ALL and BAND.off_ticks.sum() == N_CROSS_ALL, "band table loses ticks"

gap_ticks = int(hist25.loc[(hist25.band_lo_m >= GAP_LO) & (hist25.band_hi_m <= GAP_HI), "same_ticks"].sum())
gap_h = gap_ticks * HOURS_PER_ROW
near_ticks = int(hist25.loc[hist25.band_hi_m <= GAP_LO, "same_ticks"].sum())
zero_bins = int((hist25.loc[hist25.band_lo_m < 3000.0, "same_ticks"] == 0).sum())
log(f"ANSWER TO THE FALSIFICATION TEST: the 250-600 m band IS occupied -- {gap_ticks:,} same-line pair-ticks = "
    f"{gap_h:,.1f} train-h (0-250 m: {near_ticks:,} ticks = {near_ticks * HOURS_PER_ROW:,.1f} h).  "
    f"{zero_bins} of the {FINE_EDGES.size - 1} same-line 25 m bins below 3 km are empty.")
log(f"off-line pairs in 250-600 m: "
    f"{int(hist25.loc[(hist25.band_lo_m >= GAP_LO) & (hist25.band_hi_m <= GAP_HI), 'off_ticks'].sum()):,} ticks; "
    f"0-250 m: {int(hist25.loc[hist25.band_hi_m <= GAP_LO, 'off_ticks'].sum()):,} ticks")
log("\n100 m bins 0-1 km (pair-ticks = one train x 30 s):\n" + hist100[hist100.band_lo_m < 1000].assign(
    same_h=lambda d: d.same_h.round(2), off_h=lambda d: d.off_h.round(1)).to_string(index=False))
log("\n25 m bins across the band 17 called empty (250-600 m):\n" + hist25[
    (hist25.band_lo_m >= 250) & (hist25.band_hi_m <= 600)].assign(
    same_h=lambda d: d.same_h.round(2)).to_string(index=False))
log("\nthe whole distribution, 100 m bins aggregated to 500 m:\n" + pd.DataFrame({
    "band": [f"{lo:.0f}-{hi:.0f}" for lo, hi in zip(np.arange(0, 5000, 500), np.arange(500, 5500, 500))],
    "same_h": [hist100.same_h[(hist100.band_lo_m >= lo) & (hist100.band_lo_m < lo + 500)].sum()
               for lo in range(0, 5000, 500)],
    "off_h": [hist100.off_h[(hist100.band_lo_m >= lo) & (hist100.band_lo_m < lo + 500)].sum()
              for lo in range(0, 5000, 500)]}).round(1).to_string(index=False))

# The 30 s grid can only resolve occupancy that persists for half a minute.  The same-line near field is where
# two trains pass, so re-measure it at 1 s: if the 50-200 m hole is a sampling artifact it fills in here.
H1S = pd.DataFrame({"band_lo_m": REF_EDGES[:-1], "band_hi_m": REF_EDGES[1:],
                    "same_1s": H_SAME_1S, "same_1s_h": (H_SAME_1S / 3600.0).round(3),
                    "same_30s": [H_SAME_25[FINE_EDGES[:-1] == lo][0] if (FINE_EDGES[:-1] == lo).any() else 0
                                 for lo in REF_EDGES[:-1]]})
H1S.to_csv(OUT / "hist_1s_near.csv", index=False)
_near_tab = H1S.assign(same_30s_h=lambda d: (d.same_30s / 120.0).round(3))[
    ["band_lo_m", "band_hi_m", "same_1s_h", "same_30s_h"]]
log("\n1 s refinement of the same-line near field, 25 m bins (train-h):\n" + _near_tab.to_string(index=False))
_band_tab = BAND.assign(same_h=lambda d: d.same_h.round(1), off_h=lambda d: d.off_h.round(1))[
    ["band", "same_ticks", "same_h", "off_ticks", "off_h"]]
log("\nband exposure used as the denominator from here on (train-h, full 30 s grid):\n" + _band_tab.to_string(index=False))

# The 5-minute sample of 14/17 is every 10th tick, i.e. it always lands on the same phase of the 30 s grid.  A
# deterministic timetable makes that a systematic, not a random, subsample: check the two exposure tables against
# each other before using either as a denominator.
bg = np.load(ROOT / "out" / "rewind_dist" / "bg_positions.npz", allow_pickle=True)
BG_D, BG_CLS = bg["bg_dist"].astype(np.float64), bg["bg_cls"]
ALIAS = pd.DataFrame({"band": AN_LABELS,
                      "same_h_full": BAND.same_h.round(1),
                      "same_h_5min": (np.histogram(BG_D[BG_CLS <= 1], bins=AN_EDGES)[0] / 12.0).round(1),
                      "off_h_full": BAND.off_h.round(1),
                      "off_h_5min": (np.histogram(BG_D[BG_CLS >= 2], bins=AN_EDGES)[0] / 12.0).round(1)})
ALIAS["same_ratio"] = (ALIAS.same_h_5min / ALIAS.same_h_full).round(2)
ALIAS["off_ratio"] = (ALIAS.off_h_5min / ALIAS.off_h_full).round(2)
ALIAS.to_csv(OUT / "exposure_aliasing.csv", index=False)
log(f"\nfull grid vs the 5-minute sample of 14/17 (ratio 1.0 = the subsample is representative; the grid has "
    f"{N_SAME_ALL:,} same-line / {N_CROSS_ALL:,} off-line pair-ticks, the sample "
    f"{BG_D.size:,} rows = {BG_D.size / 12:,.1f} train-h)\n" + ALIAS.to_string(index=False))
flush_log()


# ================================================================================================
section("3. The 143 breakdowns on that distribution")
# ================================================================================================
EV["d_grid_m"] = EV_DS
EV["net_grid_m"] = np.where(np.isfinite(EV_DN), EV_DN, np.nan)
EV["band"] = pd.cut(EV.straight_m, AN_EDGES, labels=AN_LABELS, right=False)
EV["band_grid"] = pd.cut(EV.d_grid_m, AN_EDGES, labels=AN_LABELS, right=False)
EV["band_net"] = pd.cut(EV.network_m, AN_EDGES, labels=AN_LABELS, right=False)
EV["band_net_grid"] = pd.cut(EV.net_grid_m, AN_EDGES, labels=AN_LABELS, right=False)
EV.to_csv(OUT / "events_bands.csv", index=False)
ON, OFF = EV[EV.is_online], EV[~EV.is_online]
diff = (ON.d_grid_m - ON.straight_m).abs()
log(f"cross-check of the event geometry: our grid distance vs events_geo.straight_m (on-line), median |diff| "
    f"{np.nanmedian(diff):.1f} m, 90th pct {np.nanpercentile(diff, 90):.1f} m, "
    f"{int((diff < 5).sum())}/{diff.notna().sum()} within 5 m")
log(f"on-line ({len(ON)} events): smallest {ON.straight_m.min():.0f} m, largest below 250 m "
    f"{ON.straight_m[ON.straight_m < GAP_LO].max():.0f} m, smallest above 600 m "
    f"{ON.straight_m[ON.straight_m > GAP_HI].min():.0f} m -> the 250-600 m window holds "
    f"{int(((ON.straight_m >= GAP_LO) & (ON.straight_m <= GAP_HI)).sum())} on-line events")
log(f"off-line ({len(OFF)} events, {int(OFF.straight_m.notna().sum())} with a finite distance): smallest "
    f"{OFF.straight_m.min():.0f} m, {int((OFF.straight_m < GAP_LO).sum())} below 250 m, "
    f"{int((OFF.straight_m > GAP_HI).sum())} above 600 m, the rest in between")
_ev_tab = pd.DataFrame({"band": AN_LABELS,
                        "on": [int((ON.band == b).sum()) for b in AN_LABELS],
                        "off": [int((OFF.band == b).sum()) for b in AN_LABELS],
                        "on_net_ours": [int((ON.band_net_grid == b).sum()) for b in AN_LABELS],
                        "on_net_12geo": [int((ON.band_net == b).sum()) for b in AN_LABELS]})
log("\nbreakdowns by straight-line distance, with two network-metric counts beside it:\n"
    + _ev_tab.to_string(index=False))
_near_ev, _far_ev = ON[ON.straight_m < GAP_LO], ON[ON.straight_m >= GAP_LO]
log("carrier state of the on-line events -- below 250 m: " +
    ", ".join(f"{k} {v}" for k, v in _near_ev.carrier_state.value_counts().items()) +
    f" (n={len(_near_ev)}); 250 m and beyond: " +
    ", ".join(f"{k} {v}" for k, v in _far_ev.carrier_state.value_counts().items()) + f" (n={len(_far_ev)})")
log("\non-line events above 250 m: " + ", ".join(
    f"{r.incident_id} {r.straight_m:.0f} m" for r in ON[ON.straight_m >= GAP_LO].itertuples()))
flush_log()


# ================================================================================================
section("4. Mechanism: the timetable's own spacing floor, and what the near field really looks like")
# ================================================================================================
seg10 = LEN_S[TRAV_LINE10 & np.isfinite(LEN_S)]
shortest = float(np.min(seg10))
assert abs(shortest - SHORTEST_SEG_HINT) < 1e-6, f"shortest Line-10 segment moved: {shortest}"
log(f"shortest Line-10 segment {shortest:.1f} m (median {np.median(seg10):.1f} m); smallest same-line separation "
    f"anywhere in the panel {stat['min_same_m']:.1f} m -> the segment length is not a spacing floor: "
    f"{int((ON.straight_m < shortest).sum())} of the {len(ON)} on-line events sit inside a single segment")
log(f"the sub-50 m mass is {stat['n50_opp']}/{stat['n50']} opposing-direction, and {stat['n50'] - stat['n50_sameseg']}"
    f" of them are on different segments: these are meetings at a shared station node (one train arriving, one "
    f"departing) that a 30 s grid lands on exactly, which is why the 30 s histogram has a spike at 0 m and a hole "
    f"at 50-200 m while the 1 s refinement is flat")
_t = trav.loc[TRAV_LINE10, ["date", "from_station", "direction", "dep_ts"]].sort_values(
    ["date", "from_station", "direction", "dep_ts"])
GAP_S = _t.groupby(["date", "from_station", "direction"]).dep_ts.diff().dt.total_seconds().values
GAP_S = GAP_S[np.isfinite(GAP_S) & (GAP_S > 0)]
v10 = float(np.median(seg10 / PLAN_S[TRAV_LINE10 & np.isfinite(PLAN_S)]))
log(f"same-line departure headway at a station (n={GAP_S.size:,}): min {GAP_S.min():.0f} s, 1st pct "
    f"{np.percentile(GAP_S, 1):.0f} s, 5th {np.percentile(GAP_S, 5):.0f} s, median {np.median(GAP_S):.0f} s; at "
    f"the median Line-10 speed {v10:.1f} m/s the smallest headway implies {GAP_S.min() * v10:.0f} m of track and "
    f"the median one {np.median(GAP_S) * v10:.0f} m, against a {shortest:.0f} m shortest segment")


# ================================================================================================
section("5. The decisive number: on-line rate per 1000 train-h by distance, over the bands the timetable populates")
# ================================================================================================


def same_hours(lo, hi, fine=True) -> float:
    """Same-line exposure in a band: the 1 s refinement below 1 km (where the 30 s grid aliases), the grid above."""
    if hi <= NEAR_M and fine:
        return (_cum(REF_EDGES, H_SAME_1S, hi) - _cum(REF_EDGES, H_SAME_1S, lo)) / 3600.0
    return (cum_ticks(hi, H_SAME_25, H_SAME_100, H_SAME_TAIL)
            - cum_ticks(lo, H_SAME_25, H_SAME_100, H_SAME_TAIL)) * HOURS_PER_ROW


RATE = pd.DataFrame({"band": AN_LABELS, "lo_m": AN_EDGES[:-1], "hi_m": AN_EDGES[1:]})
RATE["events"] = [int((ON.band == b).sum()) for b in AN_LABELS]
RATE["h_best"] = [same_hours(lo, hi) for lo, hi in zip(AN_EDGES[:-1], AN_EDGES[1:])]
RATE["h_30s"] = [same_hours(lo, hi, fine=False) for lo, hi in zip(AN_EDGES[:-1], AN_EDGES[1:])]
RATE["h_5min"] = np.histogram(BG_D[BG_CLS <= 1], bins=AN_EDGES)[0] / 12.0
for tag in ("best", "30s", "5min"):
    # A band with almost no exposure gives a meaningless rate (and 17's sample has empty cells), so blank those.
    RATE[f"rate_{tag}"] = np.where(RATE[f"h_{tag}"] >= THIN_H,
                                   1000.0 * RATE.events / RATE[f"h_{tag}"].clip(lower=1e-9), np.nan)
RATE["ci_lo"], RATE["ci_hi"] = garwood(RATE.events.values, RATE.h_best.values)
# The off-line side of the same table, so the two channels are read against one another rather than in isolation.
RATE["off_events"] = [int((OFF.band == b).sum()) for b in AN_LABELS]
RATE["off_h"] = BAND.off_h.values
RATE["off_rate"] = np.where(RATE.off_h >= THIN_H, 1000.0 * RATE.off_events / RATE.off_h.clip(lower=1e-9), np.nan)
RATE["off_ci_lo"], RATE["off_ci_hi"] = garwood(RATE.off_events.values, RATE.off_h.values)
RATE.to_csv(OUT / "rate_by_band.csv", index=False)
_cols = ["band", "events", "h_best", "rate_best", "ci_lo", "ci_hi", "off_events", "off_h", "off_rate",
         "off_ci_hi", "rate_30s", "rate_5min"]
_tab = RATE[_cols].copy()
for c in ("h_best", "off_h"):
    _tab[c] = _tab[c].round(1)
for c in ("rate_best", "ci_lo", "ci_hi", "off_rate", "off_ci_hi", "rate_30s", "rate_5min"):
    _tab[c] = _tab[c].round(1)
log(f"breakdowns per 1000 train-hours by distance band, exact Poisson 95% CI. h_best is the 1 s-refined exposure "
    f"below 1 km and the 30 s grid above; the off-line columns are the same quantities for the 74 off-line events "
    f"against off-line exposure (blank where exposure < {THIN_H:.0f} train-h):\n"
    + _tab.to_string(index=False, na_rep="--"))
_h1 = float(RATE.h_best[RATE.hi_m <= 600].sum())
_h3 = float(RATE.h_30s[RATE.hi_m <= 600].sum())
log(f"\nconsistency of the refinement: below 600 m the 1 s samples total {_h1:.1f} train-h against {_h3:.1f} h from "
    f"the 30 s grid ({100 * _h1 / _h3:.0f}%), so the refinement redistributes exposure between bands without "
    f"creating or destroying it")

def band_agg(lo, hi):
    """Events and exposure summed over every band inside [lo, hi], with the exact Poisson CI."""
    sel = RATE[(RATE.lo_m >= lo) & (RATE.hi_m <= hi)]
    n, h = int(sel.events.sum()), float(sel.h_best.sum())
    ci = garwood(np.array([n], dtype=float), np.array([h]))
    return n, h, float(ci[0][0]), float(ci[1][0])


def trend(sel, name):
    """Poisson score test for a rate that varies with log distance: U = sum(x_i (z_i - zbar)) with the exposure
    weights, Var = lam * sum(h_i (z_i - zbar)^2) under a constant rate lam."""
    w = sel.h_best.values
    z = np.log(np.where(np.isfinite(sel.hi_m.values), np.sqrt(np.maximum(sel.lo_m.values, 1.0) * sel.hi_m.values),
                        20000.0))
    lam = float(sel.events.sum() / w.sum())
    zbar = float((w * z).sum() / w.sum())
    U = float((sel.events.values * (z - zbar)).sum())
    V = lam * float((w * (z - zbar) ** 2).sum())
    zsc = U / np.sqrt(V)
    return zsc, float(2 * stats.norm.sf(abs(zsc))), lam, len(sel)


N_LO, H_LO, CLO_LO, CLO_HI = band_agg(0.0, GAP_LO)
N_GAP, H_GAP, CGAP_LO, CGAP_HI = band_agg(GAP_LO, GAP_HI)
R_LO, R_GAP = 1000.0 * N_LO / H_LO, 1000.0 * N_GAP / H_GAP
log(f"\nTHE NUMBER: below 250 m {N_LO} on-line events in {H_LO:.1f} same-line train-h = {R_LO:.1f} per 1000 h "
    f"[{CLO_LO:.1f}, {CLO_HI:.1f}];  250-600 m {N_GAP} events in {H_GAP:.1f} train-h = {R_GAP:.1f} per 1000 h "
    f"[{CGAP_LO:.1f}, {CGAP_HI:.1f}].  The band the claim rests on is populated by {H_GAP:.0f} train-h, "
    f"{H_GAP / H_LO:.2f}x the exposure below 250 m, and holds none of the {N_LO + N_GAP} near events.")
log(f"rate ratio: at least {R_LO / max(CGAP_HI, 1e-9):.1f}x (the numerator's lower CI bound over the 250-600 m upper "
    f"bound gives {CLO_LO / max(CGAP_HI, 1e-9):.1f}x), i.e. the timetable cannot be manufacturing this gap unless it "
    f"also concentrates the breakdowns into the near field")
for _sel, _name in ((RATE, "all 12 bands"), (RATE[RATE.hi_m <= 600.0], "the six bands inside 600 m")):
    _z, _p, _lam, _k = trend(_sel, _name)
    log(f"trend in the rate against log distance, {_name}: score z = {_z:+.2f}, p = {_p:.1e} "
        f"(constant rate {1000 * _lam:.1f} per 1000 h would give z = 0)")
def chi2_hom(sel):
    """Pearson chi-square of the band counts against one pooled rate over those bands."""
    h, x = sel.h_best.values, sel.events.values
    exp = x.sum() / h.sum() * h
    return float(((x - exp) ** 2 / np.maximum(exp, 1e-12)).sum()), len(x) - 1


_c_all, _df_all = chi2_hom(RATE)
_c_near, _df_near = chi2_hom(RATE[RATE.hi_m <= 600.0])
log(f"homogeneity against one pooled rate: all 12 bands chi2 = {_c_all:,.0f} on {_df_all} df (p = "
    f"{stats.chi2.sf(_c_all, _df_all):.1e}), the six near bands alone chi2 = {_c_near:,.0f} on {_df_near} df "
    f"(p = {stats.chi2.sf(_c_near, _df_near):.1e})")
flush_log()


# ================================================================================================
section("6. Robustness: another metric, only isolated events, and no terminal layovers")
# ================================================================================================


def cum_hours(x, net=False) -> float:
    """Same-line exposure below x metres: the 1 s refinement (straight metric) or the 30 s grid (either metric)."""
    if net:
        assert x <= 3000.0, "the network-metric grid only runs to 3 km"
        return cum_ticks(x, H_SAME_25_NET, np.zeros_like(H_SAME_100), np.zeros_like(H_SAME_TAIL)) * HOURS_PER_ROW
    if x <= NEAR_M:
        return _cum(REF_EDGES, H_SAME_1S, x) / 3600.0
    return cum_ticks(x, H_SAME_25, H_SAME_100, H_SAME_TAIL) * HOURS_PER_ROW


def pair_row(name, d_on, net=False, n_used=None):
    n_lo = int((d_on < GAP_LO).sum())
    n_gap = int(((d_on >= GAP_LO) & (d_on <= GAP_HI)).sum())
    h_lo = cum_hours(GAP_LO, net)
    h_gap = cum_hours(GAP_HI, net) - h_lo
    lo_lo, hi_lo = garwood(np.array([n_lo], dtype=float), np.array([h_lo]))
    lo_g, hi_g = garwood(np.array([n_gap], dtype=float), np.array([h_gap]))
    return {"variant": name, "n_used": len(d_on) if n_used is None else n_used,
            "n_lo": n_lo, "h_lo": h_lo, "rate_lo": 1000.0 * n_lo / h_lo,
            "ci_lo_lo": float(lo_lo[0]), "ci_lo_hi": float(hi_lo[0]),
            "n_gap": n_gap, "h_gap": h_gap, "rate_gap": 1000.0 * n_gap / h_gap,
            "ci_gap_hi": float(hi_g[0]),
            "ratio_min": (1000.0 * n_lo / h_lo) / max(float(hi_g[0]), 1e-9)}


_isolated = ON[~ON.in_burst]
_no_term = ON[ON.carrier_state != "terminal"]
ROB = [pair_row("baseline: straight, 1 s-refined exposure", ON.straight_m.values)]
ROB.append(pair_row("(a) direction-corrected network distance, 30 s grid", ON.net_grid_m.values, net=True))
ROB.append(pair_row("(b) isolated events only (no 23 bursts)", _isolated.straight_m.values))
ROB.append(pair_row("(c) no terminal-layover events", _no_term.straight_m.values))
ROB = pd.DataFrame(ROB)
ROB.to_csv(OUT / "robustness.csv", index=False)
_tab = ROB.copy()
for c in ("h_lo", "h_gap", "rate_lo", "ci_lo_lo", "ci_lo_hi", "rate_gap", "ci_gap_hi", "ratio_min"):
    _tab[c] = _tab[c].round(1)
log(_tab.to_string(index=False))
_gap_net = ON[(ON.net_grid_m >= GAP_LO) & (ON.net_grid_m <= GAP_HI)]
log(f"(a) events left in the band under the network metric: {len(_gap_net)}"
    + ("".join(f" ({r.incident_id}: {r.straight_m:.0f} m straight, {r.network_m:.0f} m as 12_geo has it, "
               f"{r.net_grid_m:.0f} m corrected)" for r in _gap_net.itertuples()) if len(_gap_net) else ""))
_term = ON[ON.carrier_state == "terminal"]
log(f"(c) the {len(_term)} on-line terminal-layover events sit at "
    + ", ".join(f"{r.straight_m:.0f}" for r in _term.sort_values("straight_m").itertuples())
    + f" m, so dropping them changes neither band; (b) is the one check that costs power: {len(_isolated)} of the "
    f"{len(ON)} on-line events are isolated, {len(_isolated[_isolated.straight_m < GAP_LO])} of them below 250 m, "
    f"so the {GAP_LO:.0f}-{GAP_HI:.0f} m zero is then worth only a {float(ROB.ratio_min.iloc[2]):.1f}x ratio")
_n_geo = int(((ON.network_m >= GAP_LO) & (ON.network_m <= GAP_HI)).sum())
_n_net = int(((ON.net_grid_m >= GAP_LO) & (ON.net_grid_m <= GAP_HI)).sum())
log(f"\n(a) doubles as the artefact check: against events_geo.network_m {_n_geo} on-line events appear to sit in "
    f"250-600 m, because 12_geo's same-segment formula |f1-f2|*L is applied to opposing co-segment pairs; taking the "
    f"distance by direction, and at each onset's own time rather than at the nearest 30 s tick, leaves {_n_net} in "
    f"the band and {int((ON.net_grid_m < GAP_LO).sum())} below 250 m")
flush_log()
