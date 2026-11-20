"""14_rewind_dist.py -- the carrier-victim distance at 9 rewind instants, against the in-service background.

Four classes (victim line 10? x same travel direction as the carrier?) x nine rewind instants t = 0 .. 600 s.
Each cell overlays two frequency polygons over 0 .. 5000 m:
  grey   -- every in-service train's straight-line distance to the carrier, at the same class, sampled on the 30-s
            grid every 5 min (no onset, so there is nothing to rewind: the grey curve is the same in all 9 columns
            of a row and is repeated there so each panel can be read on its own),
  colour -- the real breakdown trains' distance to the carrier at onset - t (blue: same line, orange: cross line).
Both curves are normalised to a density inside the 0 .. 5000 m window (area = 1), so the shapes are comparable; the
share of each distribution beyond 5 km is reported separately, never hidden by the truncation.

Positions come from src/12_geo.py's own locate() -- the top level of that script is cut before section 5 and executed
in a scratch root, so out/geo/ is never rewritten and there is exactly one interpolator in the project. The
background walks the same 30-s grid as src/03_mechanism.py with the same vectorised construction as 12_geo section 5;
section 2 asserts it reproduces locate() station-for-station on a random sample of train-ticks.

Direction: a train's direction is the `direction` field (asc/desc) of the traversal that covers the instant. At an
end-of-trip terminal the carrier has no traversal -- it is reversing -- and it keeps the direction of the trip it just
arrived on; that is the convention that reproduces the event counts of the brief (same line: 1 same-direction vs 67
opposite), and section 4 reports what the other convention (the trip about to depart) would give instead.

Sections 3b/5 turn the same cells into rate curves: events per band over background exposure in train-hours, with exact
Poisson intervals, which is the main figure -- the distance distribution is only a density and cannot separate "the victims
are close" from "the line is close". Section 6 profiles the latency t by maximum likelihood under
lambda(d) = lambda0 * exp(-d/L) fitted on the background exposure, with lambda0 eliminated analytically and L profiled out;
section 8 repeats the whole profile on placebo events (a random in-service train-tick of the same line and day) to show
that the sharpness of the real peak is not a geometric artefact.

Outputs: out/rewind_dist/{bg_positions.npz, event_dist.csv, curve_density.csv, rate_by_t.csv,
                           profile_likelihood.csv, rewind_dist_results.txt},
         figures/rewind/{fig_rewind_4x9.png (rate curves, main), fig_rewind_4x9_dist.png (densities),
                         fig_profile_t.png (profile likelihood + placebo)}
Run: python3 src/14_rewind_dist.py   (~12 s; reads data_derived/traversals.pkl and the OSM cache; no network access)
"""
from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402
from scipy.special import gammaln  # noqa: E402
from scipy.stats import chi2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
GEO = ROOT / "out" / "geo"
OUT = ROOT / "out" / "rewind_dist"
FIG = ROOT / "figures" / "rewind"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT / "rewind_dist_results.txt"
CARRIER = "T3420"
T_SHOW = [0, 5, 10, 15, 30, 60, 120, 300, 600]  # rewind instants of the figures, seconds before onset
T_GRID = np.arange(0, 601, 5)  # every 5 s: the candidate latencies of the profile likelihood (section 6)
BG_TICK_STRIDE = 10  # background ticks: every 10th 30-s tick = one sample every 5 minutes
X_MAX, N_BINS = 5000.0, 60  # the plotted window and the bin count of every curve
BIN_EDGES = np.linspace(0.0, X_MAX, N_BINS + 1)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
BIN_W = BIN_EDGES[1] - BIN_EDGES[0]
# (key, victim on line 10, same direction as the carrier, row label, line colour)
CLASSES = [
    ("same_line_same_dir", True, True, "同线同向 · same line, same direction", "#2b6cb0"),
    ("same_line_opposite", True, False, "同线反向 · same line, opposite direction", "#2b6cb0"),
    ("cross_line_same_dir", False, True, "跨线同向 · cross line, same direction", "#dd6b20"),
    ("cross_line_opposite", False, False, "跨线反向 · cross line, opposite direction", "#dd6b20"),
]
CKEY = [c[0] for c in CLASSES]
ROW_LABEL = {c[0]: c[3] for c in CLASSES}
ROW_COLOR = {c[0]: c[4] for c in CLASSES}

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


def load_locator():
    """Exec the part of 12_geo.py above section 5 in a scratch root, so out/geo/ is never rewritten."""
    src_path = SRC / "12_geo.py"
    text = src_path.read_text(encoding="utf-8")
    cut = text.index('section("5. Exposure on the 30-s grid')
    scratch = Path(tempfile.mkdtemp(prefix="rewinddist_geo_"))
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
    ns = {"__file__": str(scratch / "src" / "12_geo.py"), "__name__": "geo12_for_rewinddist"}
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
U, V = np.array([NODE[s] for s in FROM], dtype=np.int16), np.array([NODE[s] for s in TO], dtype=np.int16)
TC = trav.train_id.cat.codes.values.astype(np.int16)
N_TRAIN = len(trav.train_id.cat.categories)
DEP_S, ARR_S = NS["sec_of_day"](trav.dep_ts, trav.date), NS["sec_of_day"](trav.arr_ts, trav.date)
T0 = float(np.floor(np.nanmin(DEP_S) / 3600) * 3600)  # 05:30, the first departure of the day
TICK_S = NS["TICK_S"]
NT = int(np.ceil((np.nanmax(ARR_S) - T0) / TICK_S)) + 1
TCODES, DCODES = NS["TCODES"], NS["DCODES"]
combo, order, combo_s = NS["combo"], NS["order"], NS["combo_s"]
T_ARR, T_DEP, T_ORDER, T_COMBO_S = NS["T_ARR"], NS["T_DEP"], NS["T_ORDER"], NS["T_COMBO_S"]
CAR_T_START_S, CAR_T_END_S = NS["CAR_T_START_S"], NS["CAR_T_END_S"]
CAR_T_NODE, CAR_T_DCODE = NS["CAR_T_NODE"], NS["CAR_T_DCODE"]
CI = TCODES[CARRIER]
DIR_NAME = list(trav.direction.cat.categories)
TRAV_DIR = trav.direction.cat.codes.values.astype(np.int8)
TRAV_LINE10 = (trav.line_id.astype(str) == "10").values.astype(bool)  # a train-day never changes line
CAR_LINE10 = True  # the carrier is a Line 10 train, so "same line" means "the other train is on Line 10"


def tick_ts(day, tick: int):
    """Timestamp of grid tick `tick` on `day` (the grid is T0 + k * TICK_S seconds after midnight)."""
    return pd.Timestamp(day) + pd.Timedelta(seconds=float(T0 + tick * TICK_S))


def direction_at(train: str, day, ts, at_terminal: str = "arrive") -> int | None:
    """Direction code of the traversal covering `ts` (0 = asc, 1 = desc), or None when the train is out of service.

    A train reversing at an end-of-trip station has no traversal; `at_terminal` decides which neighbour to copy:
    'arrive' (the trip that just ended, the default) or 'depart' (the trip about to start).
    """
    if train not in TCODES or pd.Timestamp(day) not in DCODES:
        return None
    c = TCODES[train] * 1000 + DCODES[pd.Timestamp(day)]
    lo, hi = np.searchsorted(combo_s, c, "left"), np.searchsorted(combo_s, c, "right")
    rows = order[lo:hi]
    if rows.size == 0:
        return None
    t = np.datetime64(pd.Timestamp(ts), "ns").astype(np.int64)
    m = (DEP[rows] <= t) & (ARR[rows] > t)
    if m.any():  # inside a segment
        return int(TRAV_DIR[rows[m][0]])
    m2 = np.isfinite(DWELL[rows]) & (DEP[rows] - DWELL[rows] * 1e9 <= t) & (DEP[rows] > t)
    if m2.any():  # dwelling at a station
        return int(TRAV_DIR[rows[m2][0]])
    lo, hi = np.searchsorted(T_COMBO_S, c, "left"), np.searchsorted(T_COMBO_S, c, "right")
    tr = T_ORDER[lo:hi]
    if ((T_ARR[tr] <= t) & (T_DEP[tr] > t)).any():  # reversing at a terminal
        if at_terminal == "depart":
            cand = rows[DEP[rows] > t]
            return int(TRAV_DIR[cand[np.argmin(DEP[cand])]]) if cand.size else None
        cand = rows[ARR[rows] <= t]
        return int(TRAV_DIR[cand[np.argmax(ARR[cand])]]) if cand.size else None
    return None


# ================================================================================================
section("1. The 143 breakdowns: class, and the distance to the carrier at onset - t")
# ================================================================================================
ev = pd.read_csv(ROOT / "out" / "interchange" / "events.csv", usecols=["incident_id", "train_id", "line",
                                                                       "onset_ts", "location"])
ev = ev.merge(pd.read_csv(GEO / "events_geo.csv", usecols=["incident_id", "carrier_state"]), on="incident_id")
assert len(ev) == 143, f"expected 143 breakdowns, got {len(ev)}"
n_off = int((ev.carrier_state == "off").sum())
ev = ev[ev.carrier_state != "off"].reset_index(drop=True)
EVENTS = ev  # kept: later sections reuse the name `ev` for counts
KEY_OF = {(c[1], c[2]): c[0] for c in CLASSES}


def classify(line, dir_victim, dir_carrier) -> str:
    return KEY_OF[(str(line) == "10", dir_victim == dir_carrier)]


rows = []
for r in ev.itertuples():
    onset, day = pd.Timestamp(r.onset_ts), pd.Timestamp(r.onset_ts).normalize()
    dv, dc = direction_at(r.train_id, day, onset), direction_at(CARRIER, day, onset)
    cls = classify(r.line, dv, dc)
    for t in T_GRID:  # the whole 5-s grid, so section 6 can score any candidate latency
        at = onset - pd.Timedelta(seconds=int(t))
        v, c = locate(r.train_id, day, at), locate(CARRIER, day, at)
        rows.append({"incident_id": r.incident_id, "train_id": r.train_id, "line": str(r.line), "class": cls,
                     "t": t, "victim_kind": None if v is None else v["kind"],
                     "carrier_kind": None if c is None else c["kind"],
                     "dist_m": np.nan if (v is None or c is None)
                     else float(haversine(v["lat"], v["lon"], c["lat"], c["lon"]))})
edist_all = pd.DataFrame(rows)
edist = edist_all[edist_all.t.isin(T_SHOW)].reset_index(drop=True)  # the 9 instants of the figures
edist.to_csv(OUT / "event_dist.csv", index=False)
ev_cls = edist.drop_duplicates("incident_id")[["incident_id", "class"]]
log(f"breakdowns: {len(ev) + n_off}; carrier off (dropped): {n_off}; classified: {len(ev)}; "
    f"event x t rows {len(edist_all):,} on the 5-s grid ({len(edist):,} at the 9 plotted instants), of which "
    f"locatable {int(edist_all.dist_m.notna().sum()):,}")
log("events per class (direction = direction of the trip that just arrived, at terminals): "
    + ", ".join(f"{k} {int((ev_cls['class'] == k).sum())}" for k in CKEY))
alt = {}
for r in ev.itertuples():
    onset, day = pd.Timestamp(r.onset_ts), pd.Timestamp(r.onset_ts).normalize()
    alt[classify(r.line, direction_at(r.train_id, day, onset, "depart"),
                 direction_at(CARRIER, day, onset, "depart"))] = \
        alt.get(classify(r.line, direction_at(r.train_id, day, onset, "depart"),
                         direction_at(CARRIER, day, onset, "depart")), 0) + 1
log("events per class under the other terminal convention (direction of the trip about to depart): "
    + ", ".join(f"{k} {alt.get(k, 0)}" for k in CKEY))
log("victim state at onset-t: " + str(edist.victim_kind.value_counts(dropna=False).to_dict()))
log("carrier state at onset-t: " + str(edist.carrier_kind.value_counts(dropna=False).to_dict()))
flush_log()


# ================================================================================================
section("2. Background: every in-service train's distance to the carrier, sampled every 5 min")
# ================================================================================================
# The day loop below is 12_geo section 5's vectorised twin with two changes: only every 10th tick is kept, and each
# train-tick also carries its line and its direction. The segment fraction uses locate()'s corrected denominator
# (scheduled running time when the recorded spell is longer) -- without that the two would disagree inside breakdown
# spells -- and the validation at the end of this section proves the two agree everywhere.
DATES = NS["DATES"]
DCODE_ARR = pd.Categorical(trav.date, categories=DATES).codes
TRAIN_NAMES = list(trav.train_id.cat.categories)
plan_s = (pd.DataFrame({"route_id": trav.route_id.astype(str), "key": trav.segment.astype(str).map(NS["undirected"])})
          .merge(NS["spd"][["route_id", "key", "travel_s_med"]].assign(route_id=lambda d: d.route_id.astype(str)),
                 on=["route_id", "key"], how="left").travel_s_med.values.astype(float))
DUR_S = (ARR - DEP) / 1e9
DEN_S = np.where(np.isfinite(plan_s), np.minimum(DUR_S, plan_s), DUR_S)
DEN_S = np.where(DEN_S > 0, DEN_S, 1.0)  # locate() divides by this; a zero-length spell has no in-service tick
SAMP = np.arange(0, NT, BG_TICK_STRIDE)  # the sampled ticks, every 5 minutes
log(f"grid: {N_TRAIN} trains x {len(DATES)} dates x {NT} ticks of {TICK_S:.0f} s from {T0 / 3600:.1f}:00; "
    f"sampled ticks {SAMP.size}/day ({SAMP.size * len(DATES):,} tick-days), stride {BG_TICK_STRIDE}")

acc = {k: [] for k in ("day", "tick", "train", "cls", "dist", "lat", "lon")}
car_acc = {k: [] for k in ("day", "tick", "lat", "lon", "dir")}
n_term_ticks = 0
val = {"n": 0, "max_m": 0.0, "on_ok": 0, "off_none": 0, "off_term": 0, "off_bad": 0}
rng = np.random.default_rng(0)
for di, day in enumerate(DATES):
    sel = DCODE_ARR == di
    u_g = np.full((N_TRAIN, NT), -1, dtype=np.int16)  # segment start node, -1 = not in service
    v_g = np.full((N_TRAIN, NT), -1, dtype=np.int16)  # segment end node (== start when dwelling)
    f_g = np.zeros((N_TRAIN, NT), dtype=np.float64)  # position along the segment, 0..1
    dir_g = np.full((N_TRAIN, NT), -1, dtype=np.int8)  # direction code, -1 = not set
    lin_g = np.zeros((N_TRAIN, NT), dtype=bool)  # the train is on Line 10 at that tick
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
    # The carrier's terminal layovers carry a position (12_geo section 4/5 rule); other trains' layovers do not, so a
    # train is in the background risk set exactly while it is inside a segment or a 30-s dwell.
    tw = np.flatnonzero(CAR_T_DCODE == di)
    a = np.ceil((CAR_T_START_S[tw] - T0) / TICK_S).astype(np.int64)
    b = np.ceil((CAR_T_END_S[tw] - T0) / TICK_S).astype(np.int64)
    n_t = np.maximum(b - a, 0)
    rep_local = np.repeat(np.arange(tw.size), n_t)
    tick = a[rep_local] + (np.arange(n_t.sum()) - np.repeat(np.cumsum(n_t) - n_t, n_t))
    u_g[CI, tick], v_g[CI, tick], f_g[CI, tick] = CAR_T_NODE[tw][rep_local], CAR_T_NODE[tw][rep_local], 0.0
    del tw, rep_local, tick, n_t, a, b

    idx = SAMP
    valid = u_g[:, idx] >= 0
    uu, vv, ff = np.maximum(u_g[:, idx], 0), np.maximum(v_g[:, idx], 0), f_g[:, idx]
    plat = np.where(valid, LAT[uu] + ff * (LAT[vv] - LAT[uu]), np.nan)  # position of every in-service train
    plon = np.where(valid, LON[uu] + ff * (LON[vv] - LON[uu]), np.nan)
    con = valid[CI]
    d = haversine(plat[CI][None, :], plon[CI][None, :], plat, plon)  # (train, tick) metres to the carrier
    d[CI] = np.nan  # the carrier is not its own neighbour
    d[:, ~con] = np.nan  # no distance when the carrier is out of service
    cd = dir_g[CI, idx].astype(np.int16)
    for j in np.flatnonzero(con & (cd < 0)):  # carrier reversing at a terminal: direction of the trip it arrived on
        cd[j] = direction_at(CARRIER, day, tick_ts(day, idx[j]))
        n_term_ticks += 1
    assert (cd[con] >= 0).all(), f"carrier direction unknown on an in-service tick of {day.date()}"
    assert (dir_g[:, idx][valid & (np.arange(N_TRAIN)[:, None] != CI)] >= 0).all(), "direction missing in service"
    same_dir = dir_g[:, idx] == cd[None, :]
    same_line = lin_g[:, idx]  # the carrier is on Line 10, so this is exactly the pair's "same line" label
    cls = np.where(same_line, np.where(same_dir, 0, 1), np.where(same_dir, 2, 3)).astype(np.int8)
    ok = np.isfinite(d)
    ti, tr = np.nonzero(ok)
    acc["day"].append(np.full(ti.size, di, np.int16))
    acc["tick"].append(idx[tr].astype(np.int16))
    acc["train"].append(tr.astype(np.int16))
    acc["cls"].append(cls[ti, tr])
    acc["dist"].append(d[ok].astype(np.float32))
    acc["lat"].append(plat[ti, tr])
    acc["lon"].append(plon[ti, tr])
    car_acc["day"].append(np.full(int(con.sum()), di, np.int16))
    car_acc["tick"].append(idx[con].astype(np.int16))
    car_acc["lat"].append(plat[CI][con])
    car_acc["lon"].append(plon[CI][con])
    car_acc["dir"].append(cd[con].astype(np.int8))

    if di in (0, len(DATES) // 2):  # the grid must be locate(), not a lookalike: check two whole days' ticks
        ti_, tr_ = np.nonzero(valid)
        for p in rng.choice(ti_.size, min(800, ti_.size), replace=False):
            i, k = int(ti_[p]), int(tr_[p])
            pos = locate(TRAIN_NAMES[i], day, tick_ts(day, idx[k]))
            val["on_ok"] += int(pos is not None)
            val["off_bad"] += int(pos is None)  # grid says in service, locate() disagrees: a real defect
            if pos is not None:
                val["max_m"] = max(val["max_m"], float(haversine(pos["lat"], pos["lon"], plat[i, k], plon[i, k])))
                val["n"] += 1
        ni_, nk_ = np.nonzero(~valid)  # and the empties: locate() must return None there too
        for p in rng.choice(ni_.size, min(400, ni_.size), replace=False):
            pos = locate(TRAIN_NAMES[int(ni_[p])], day, tick_ts(day, idx[int(nk_[p])]))
            if pos is None:
                val["off_none"] += 1
            elif pos["kind"] == "terminal":  # a layover of a train other than the carrier: out of the risk set
                val["off_term"] += 1
            else:
                val["off_bad"] += 1
    del u_g, v_g, f_g, dir_g, lin_g, valid, uu, vv, ff, plat, plon, d, ok, cls, same_dir, same_line

bg = {k: np.concatenate(v) for k, v in acc.items()}
bg_car = {k: np.concatenate(v) for k, v in car_acc.items()}
np.savez_compressed(OUT / "bg_positions.npz", **{f"bg_{k}": v for k, v in bg.items()},
                    **{f"car_{k}": v for k, v in bg_car.items()},
                    classes=np.array(CKEY), train_names=np.array(TRAIN_NAMES), tick_stride=BG_TICK_STRIDE,
                    t0_s=T0, tick_s=TICK_S, day_values=DATES.astype("datetime64[D]").astype(str))
log(f"background pairs: {bg['dist'].size:,} train-ticks on {np.unique(bg['day']).size} days x {SAMP.size} ticks; "
    f"per class " + ", ".join(f"{k} {int((bg['cls'] == i).sum()):,}" for i, k in enumerate(CKEY)))
log(f"carrier in service on {bg_car['lat'].size:,} sampled ticks, of "
    f"which {n_term_ticks:,} are terminal layovers (direction copied from the arriving trip)")
assert val["off_bad"] == 0, "grid marks an in-service tick as empty (or the reverse) -- see the day loop"
log(f"grid vs locate() on {val['n']:,} random in-service train-ticks of 2 days: max position difference "
    f"{val['max_m']:.2e} m, and locate() agreed on the service state in all {val['on_ok']:,} of them")
log(f"the same check on {val['off_none'] + val['off_term']:,} out-of-service train-ticks: locate() returned None for "
    f"{val['off_none']:,} and a terminal layover for {val['off_term']:,} (other trains' layovers are deliberately "
    f"outside the risk set); 0 disagreements of any other kind")
flush_log()


# ================================================================================================
section("3. Density curves over 0-5000 m, medians, modes, and the share beyond the window")
# ================================================================================================
def curve(v) -> dict:
    """60-bin density of `v` inside the 0..5000 m window (area 1), plus the numbers that qualify it.

    The density is normalised on the in-window count, so both curves of a panel have area 1 and only their shapes are
    compared; `share_gt` keeps the part of the distribution the window cuts off, and `median_all` is the untruncated
    median (it can sit outside the window, which is why `median_in` is reported separately).
    """
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    h, _ = np.histogram(v, bins=BIN_EDGES)
    n_all, n_in = int(v.size), int(h.sum())
    dens = h / (n_in * BIN_W) if n_in else np.zeros(N_BINS)
    return {"n_all": n_all, "n_in": n_in, "density": dens, "count": h,
            "median_all": float(np.median(v)) if n_all else np.nan,
            "median_in": float(np.median(v[v <= X_MAX])) if n_in else np.nan,
            "mode_in": float(BIN_CENTERS[int(np.argmax(dens))]) if n_in else np.nan,
            "share_gt": float(np.mean(v > X_MAX)) if n_all else np.nan}


bg_dist, bg_cls = bg["dist"].astype(np.float64), bg["cls"]
BG = {k: curve(bg_dist[bg_cls == i]) for i, k in enumerate(CKEY)}
EV = {(k, t): curve(edist.loc[(edist["class"] == k) & (edist.t == t), "dist_m"]) for k in CKEY for t in T_SHOW}

crow = []
for k in CKEY:
    for tag, c, t in [("background", BG[k], np.nan)] + [("breakdown", EV[(k, t)], t) for t in T_SHOW]:
        for j in range(N_BINS):
            crow.append({"class": k, "curve": tag, "t": t, "bin_lo": BIN_EDGES[j], "bin_hi": BIN_EDGES[j + 1],
                         "bin_center": BIN_CENTERS[j], "count": int(c["count"][j]), "density": float(c["density"][j]),
                         "n_all": c["n_all"], "n_in": c["n_in"], "median_all_m": c["median_all"],
                         "median_in_m": c["median_in"], "mode_in_m": c["mode_in"], "share_gt_5km": c["share_gt"]})
pd.DataFrame(crow).to_csv(OUT / "curve_density.csv", index=False)

for k in CKEY:
    b = BG[k]
    log(f"\n[{k}]  background (all in-service trains, 5-min ticks): n={b['n_all']:,} pairs, "
        f"median {b['median_all']:,.0f} m, mode {b['mode_in']:,.0f} m, beyond 5 km {b['share_gt']:.1%}")
    tbl = pd.DataFrame([{"t": t, "n": EV[(k, t)]["n_all"], "median_all": EV[(k, t)]["median_all"],
                         "median_in": EV[(k, t)]["median_in"], "mode_in": EV[(k, t)]["mode_in"],
                         ">5km": EV[(k, t)]["share_gt"],
                         "bg_median_all": b["median_all"], "bg_mode_in": b["mode_in"]} for t in T_SHOW])
    log(tbl.round({"median_all": 0, "median_in": 0, "mode_in": 0, ">5km": 3, "bg_median_all": 0,
                   "bg_mode_in": 0}).to_string(index=False))
flush_log()


# ================================================================================================
section("3b. Exposure and rates in 11 distance bands (the rate curves of section 5)")
# ================================================================================================
# The grey curve is not there to look like the coloured one: it is the denominator. One background pair is one
# train-tick, i.e. 30 s = 1/120 train-hour of exposure, so a band's rate is events / (pairs / 120) per train-hour.
RATE_EDGES = np.array([0, 50, 100, 150, 250, 400, 600, 900, 1300, 2000, 3000, 5000], float)
RATE_MID = 0.5 * (RATE_EDGES[:-1] + RATE_EDGES[1:])
TICK_H = TICK_S / 3600.0
MIN_EXP_H = 5.0  # a band with less exposure than this is drawn hollow and left out of the slope
EXPO = {k: np.histogram(bg_dist[bg_cls == i], bins=RATE_EDGES)[0] * TICK_H for i, k in enumerate(CKEY)}


def pois_ci(k, exp_h):
    """Garwood exact 95% interval for a Poisson count k over exp_h train-hours."""
    return (np.where(k > 0, chi2.ppf(0.025, np.maximum(2 * k, 2)) / 2, 0.0) / exp_h,
            chi2.ppf(0.975, 2 * k + 2) / 2 / exp_h)


def lograte_slope(ev, exp_h) -> float:
    """Exposure-weighted slope of ln(rate) against distance in km (0.5 continuity correction; empty bands drop out)."""
    m = (exp_h >= MIN_EXP_H) & (ev > 0)
    if m.sum() < 3:
        return np.nan
    x, y, w = RATE_MID[m] / 1000.0, np.log((ev[m] + 0.5) / exp_h[m]), ev[m] + 0.5
    xc, yc = x - np.average(x, weights=w), y - np.average(y, weights=w)
    return float(np.sum(w * xc * yc) / np.sum(w * xc * xc))


RATE, SLOPE = {}, {}
for k in CKEY:
    for t in T_SHOW:
        RATE[(k, t)] = np.histogram(edist.loc[(edist["class"] == k) & (edist.t == t), "dist_m"], bins=RATE_EDGES)[0]
        SLOPE[(k, t)] = lograte_slope(RATE[(k, t)], EXPO[k])
rate_rows = []
for k in CKEY:
    lo, hi = pois_ci(np.stack([RATE[(k, t)] for t in T_SHOW]), EXPO[k])
    for i, t in enumerate(T_SHOW):
        for j in range(RATE_EDGES.size - 1):
            rate_rows.append({"class": k, "t_s": t, "band_lo_m": RATE_EDGES[j], "band_hi_m": RATE_EDGES[j + 1],
                              "band_mid_m": RATE_MID[j], "n_events": int(RATE[(k, t)][j]),
                              "exposure_h": float(EXPO[k][j]), "rate_per_h": float(RATE[(k, t)][j] / EXPO[k][j]),
                              "rate_lo": float(lo[i, j]), "rate_hi": float(hi[i, j]),
                              "thin_band": bool(EXPO[k][j] < MIN_EXP_H)})
rate_tab = pd.DataFrame(rate_rows)
rate_tab.to_csv(OUT / "rate_by_t.csv", index=False, float_format="%.6g")
log(f"wrote {OUT / 'rate_by_t.csv'} ({len(rate_tab)} rows = {len(CKEY)} classes x {len(T_SHOW)} instants x "
    f"{RATE_EDGES.size - 1} bands; exposure per class in train-hours: "
    + ", ".join(f"{k} {EXPO[k].sum():.0f}" for k in CKEY) + f"; thin bands (< {MIN_EXP_H:.0f} h): "
    + ", ".join(f"{k} {int((EXPO[k] < MIN_EXP_H).sum())}" for k in CKEY))
flush_log()


# ================================================================================================
section("3c. Which instant gives the steepest rate curve?")
# ================================================================================================
STEEP = {}
for k in CKEY:
    row = [SLOPE[(k, t)] for t in T_SHOW]
    ok = [s for s in row if np.isfinite(s)]
    t_best = T_SHOW[int(np.nanargmin(row))] if ok else None
    STEEP[k] = t_best
    log(f"[{k}] slope of ln(rate) per km: " + ", ".join(f"t={t}:{s:+.2f}" for t, s in zip(T_SHOW, row))
        + f"  -> steepest at t = {t_best} s" if ok else f"[{k}] no usable slope")
flush_log()


# ================================================================================================
section("4. The 4 x 9 figure")
# ================================================================================================
plt.rcParams.update({"font.sans-serif": ["Arial Unicode MS", "Heiti TC", "DejaVu Sans"], "axes.unicode_minus": False,
                     "font.size": 8.5})
fig, axes = plt.subplots(len(CKEY), len(T_SHOW), figsize=(23.0, 10.8), sharex=True)
for i, k in enumerate(CKEY):
    b = BG[k]
    for j, t in enumerate(T_SHOW):
        ax = axes[i, j]
        e = EV[(k, t)]
        ax.fill_between(BIN_CENTERS, b["density"], color="#b0b4b8", alpha=0.55, linewidth=0,
                        label="background: every in-service train")
        ax.plot(BIN_CENTERS, b["density"], color="#5f6368", lw=1.1)
        if e["n_in"]:
            ax.plot(BIN_CENTERS, e["density"], color=ROW_COLOR[k], lw=1.9, label="breakdown trains at onset - $t$")
        top = b["density"].max() if not e["n_in"] else max(b["density"].max(), e["density"].max())
        ax.set_ylim(0, top * 1.25)
        ax.set_xlim(0, X_MAX)
        ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
        ax.set_xticklabels(["0", "1", "2", "3", "4", "5"])
        ax.grid(axis="y", alpha=0.22, lw=0.5)
        ax.tick_params(labelsize=7.5, length=2)
        ax.annotate(f"n = {e['n_all']}\nmed {e['median_all']:,.0f} m", xy=(0.975, 0.965), xycoords="axes fraction",
                    ha="right", va="top", fontsize=7.4, linespacing=1.3,
                    color=ROW_COLOR[k] if e["n_in"] else "#8a8a8a")
        if not e["n_in"]:
            ax.annotate("no breakdown\ninside the window", xy=(0.5, 0.55), xycoords="axes fraction", ha="center",
                        fontsize=7.4, color="#8a8a8a", style="italic", linespacing=1.3)
        if i == 0:
            ax.set_title(f"t = {t} s" if t else "onset (t = 0)", fontsize=10.5, pad=6)
        if i == len(CKEY) - 1:
            ax.set_xlabel("distance to the carrier (km)", fontsize=8.5)
    n_ev = int((edist.drop_duplicates("incident_id")["class"] == k).sum())
    fig.text(0.012, 1 - (i + 0.5) / len(CKEY) * 0.86 - 0.075, f"{ROW_LABEL[k]}\n{n_ev} breakdowns · "
             f"background N = {BG[k]['n_all']:,}", rotation=90, va="center", ha="center", fontsize=10,
             color=ROW_COLOR[k], linespacing=1.5)
h, lb = axes[0, 0].get_legend_handles_labels()
fig.legend(h, lb, loc="lower center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Carrier T3420: distance to every in-service train (grey) vs to the trains that broke down, rewound by "
             "$t$\nboth curves are densities inside 0-5 km (area 1 per panel); classes split by the victim's line and "
             "its travel direction relative to the carrier", fontsize=12.5, y=0.995)
fig.subplots_adjust(left=0.085, right=0.995, top=0.905, bottom=0.085, wspace=0.16, hspace=0.14)
fig.savefig(FIG / "fig_rewind_4x9_dist.png", dpi=300)
log(f"wrote {FIG / 'fig_rewind_4x9_dist.png'} (4 x 9, {len(CKEY) * len(T_SHOW)} panels, 300 dpi)")
flush_log()


# ================================================================================================
section("5. The 4 x 9 main figure: rate curves, log-log")
# ================================================================================================
# A rate curve is the honest form of the question: the grey counts are the exposure, so a band's rate is what the
# event curve has to be compared against. If distance matters, the rate falls with distance; the instant t at which it
# falls fastest is the instant at which the victim was still in the carrier's neighbourhood.
from matplotlib.lines import Line2D  # noqa: E402

XT = [0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5]
RM_KM = RATE_MID / 1000.0  # the bands' midpoints, km
fig, axes = plt.subplots(len(CKEY), len(T_SHOW), figsize=(23.0, 10.8), sharex=True)
for i, k in enumerate(CKEY):
    for j, t in enumerate(T_SHOW):
        ax = axes[i, j]
        ev, exp_h = RATE[(k, t)], EXPO[k]
        has = exp_h > 0  # a band the background never visits has no rate at all
        rate = np.where(has, ev / np.where(has, exp_h, 1.0), np.nan)
        lo, hi = pois_ci(ev, np.where(has, exp_h, 1.0))
        thick = has & (exp_h >= MIN_EXP_H) & (ev > 0)
        thin = has & (exp_h < MIN_EXP_H) & (ev > 0)
        pos = rate[thick | thin]
        floor = 0.45 * pos.min() if pos.size else 1e-3
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.errorbar(RM_KM[thick], rate[thick], yerr=[rate[thick] - lo[thick], hi[thick] - rate[thick]],
                    color=ROW_COLOR[k], lw=1.8, marker="o", ms=3.6, capsize=2, elinewidth=0.8, ecolor=ROW_COLOR[k],
                    alpha=0.95)
        ax.plot(RM_KM[thin], rate[thin], ls="none", marker="o", ms=4.0, mfc="none", mec=ROW_COLOR[k], mew=1.2)
        zero = (ev == 0) & has & (exp_h >= MIN_EXP_H)
        if zero.any():
            ax.plot(RM_KM[zero], np.full(zero.sum(), floor), ls="none", marker="v", ms=4.0, mfc="none",
                    mec=ROW_COLOR[k], mew=1.0)
        ax.set_xlim(0.016, 6.0)
        ax.set_xticks(XT)
        ax.set_xticklabels(["0.05", "0.1", "0.25", "0.5", "1", "2", "3", "5"])
        ax.tick_params(labelsize=7.5, length=2, which="both")
        ax.grid(alpha=0.22, lw=0.5, which="both")
        if pos.size:
            ax.set_ylim(max(floor * 0.55, 1e-5), hi[thick | thin].max() * 2.6)
        if ev.sum():
            ax.axhline(ev.sum() / exp_h.sum(), color="#8a8a8a", ls="--", lw=1.0, zorder=0)  # the flat null
        else:
            ax.annotate("the only breakdown in this class is at\n"
                        f"{float(edist.loc[edist['class'] == k, 'dist_m'].iloc[0]):,.0f} m — outside every band",
                        xy=(0.5, 0.62), xycoords="axes fraction", ha="center", fontsize=7.4, color="#8a8a8a",
                        style="italic", linespacing=1.35)
        s = SLOPE[(k, t)]
        ax.annotate(f"ev {int(ev.sum())} · exp {exp_h.sum():,.0f} h\nslope "
                    + (f"{s:+.2f}/km" if np.isfinite(s) else "n/a (too few bands)"), xy=(0.03, 0.04),
                    xycoords="axes fraction", ha="left", va="bottom", fontsize=7.2, linespacing=1.35,
                    color=ROW_COLOR[k])
        if i == 0:
            steep = STEEP[k] == t
            ax.set_title(f"t = {t} s" if t else "onset (t = 0)", fontsize=10.5, pad=6,
                         color="#b3261e" if steep else "black", fontweight="bold" if steep else "normal")
        if i == len(CKEY) - 1:
            ax.set_xlabel("distance to the carrier (km)", fontsize=8.5)
    n_ev = int((edist.drop_duplicates("incident_id")["class"] == k).sum())
    fig.text(0.012, 1 - (i + 0.5) / len(CKEY) * 0.86 - 0.075, f"{ROW_LABEL[k]}\n{n_ev} breakdowns · "
             f"exposure {EXPO[k].sum():,.0f} train-h", rotation=90, va="center", ha="center", fontsize=10,
             color=ROW_COLOR[k], linespacing=1.5)
fig.legend(handles=[Line2D([], [], color="#444444", lw=1.8, marker="o", ms=3.6, label="rate, bands with ≥ 5 train-h"),
                    Line2D([], [], color="#444444", lw=0, marker="o", ms=4.0, mfc="none",
                           label="rate, thin band (< 5 train-h)"),
                    Line2D([], [], color="#444444", lw=0, marker="v", ms=4.0, mfc="none",
                           label="no event in the band (drawn at the axis floor)")],
           loc="lower center", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Carrier T3420: breakdown rate by distance, rewound by $t$ — bands 0/50/100/150/250/400/600/900/1300/2000/"
             "3000/5000 m, 95% Poisson bars\nthe red column is the instant with the steepest decay; a flat column means "
             "the victim was not near the carrier at that instant", fontsize=12.5, y=0.995)
fig.subplots_adjust(left=0.085, right=0.995, top=0.905, bottom=0.085, wspace=0.16, hspace=0.14)
fig.savefig(FIG / "fig_rewind_4x9.png", dpi=300)
log(f"wrote {FIG / 'fig_rewind_4x9.png'} (main rate figure, 4 x 9, 300 dpi)")
flush_log()


# ================================================================================================
section("6. Profile likelihood for the latency t")
# ================================================================================================
# The question is not whether distance matters but *when* the carrier was close: the carrier acts at onset - t and the
# victim breaks down t seconds later. Model: a train's onset rate at distance d is lambda(d) = lambda0 * exp(-d / L),
# with the exposure taken from the background pairs. For each candidate t on a 5-s grid, lambda0 is profiled out
# analytically (lambda0 = sum n / sum exposure * exp(-d/L)) and logL is maximised over L; t_hat = argmax logL and the
# 95% CI is {t : 2 (max logL - logL(t)) <= 3.84}. If the curve spans less than 3.84 the latency is not identifiable
# and no t_hat is quoted. Events beyond 5 km have no exposure band and drop out of the fit (counted in the output).
# The bands must cover every distance the events can take, otherwise dropping the far events would move the total
# exposure and the likelihood would just follow the event count. 200 m near the carrier, then a coarse tail to 50 km.
FIT_EDGES = np.concatenate([np.arange(0.0, 5000.0, 200.0),
                            [5000.0, 7500.0, 10000.0, 15000.0, 20000.0, 30000.0, 50000.0]])
FIT_MID = 0.5 * (FIT_EDGES[:-1] + FIT_EDGES[1:])
EXPO_F = {k: np.histogram(bg_dist[bg_cls == i], bins=FIT_EDGES)[0] * TICK_H for i, k in enumerate(CKEY)}
GROUPS = {**{k: [k] for k in CKEY}, "same_line": ["same_line_same_dir", "same_line_opposite"],
          "cross_line": ["cross_line_same_dir", "cross_line_opposite"]}
EXPO_G = {g: sum(EXPO_F[k] for k in ks) for g, ks in GROUPS.items()}


def prof_ll(n, expo, L):
    """log-likelihood of the binned Poisson model at scale L, with lambda0 profiled out."""
    e = expo * np.exp(-FIT_MID / L)
    s = e.sum()
    if s <= 0:
        return -np.inf, np.nan
    lam0 = n.sum() / s
    mu = lam0 * e
    keep = mu > 0
    return float(np.sum(n[keep] * np.log(mu[keep]) - mu[keep]) - gammaln(n + 1).sum()), float(lam0)


def fit_profile(n, expo):
    r = minimize_scalar(lambda z: -prof_ll(n, expo, float(np.exp(z)))[0], bounds=(np.log(20.0), np.log(1e5)),
                        method="bounded", options={"xatol": 1e-4})
    L = float(np.exp(r.x))
    ll, lam0 = prof_ll(n, expo, L)
    return ll, lam0, L


PL, PL_N = {}, {}
for gname, ks in GROUPS.items():
    ev_g = edist_all[edist_all["class"].isin(ks) & np.isfinite(edist_all["dist_m"])]
    for t in T_GRID:
        n = np.histogram(ev_g.loc[ev_g.t == t, "dist_m"], bins=FIT_EDGES)[0]
        PL[(gname, int(t))] = fit_profile(n, EXPO_G[gname])
        PL_N[(gname, int(t))] = int(n.sum())
def flat_ll(n, expo):
    """log-likelihood of the same model with no distance effect (L -> infinity): the null the distance term must beat."""
    s = n.sum()
    return float(s * np.log(s / expo.sum()) - s - gammaln(n + 1).sum()) if s > 0 else np.nan


PL_FLAT = {}
for gname, ks in GROUPS.items():
    ev_g = edist_all[edist_all["class"].isin(ks) & np.isfinite(edist_all["dist_m"])]
    for t in T_GRID:
        PL_FLAT[(gname, int(t))] = flat_ll(np.histogram(ev_g.loc[ev_g.t == t, "dist_m"], bins=FIT_EDGES)[0],
                                           EXPO_G[gname])
pl_rows = [{"group": g, "t_s": t, "logL": PL[(g, t)][0], "logL_flat": PL_FLAT[(g, t)],
            "LR_distance": 2.0 * (PL[(g, t)][0] - PL_FLAT[(g, t)]),
            "lambda0_per_h": PL[(g, t)][1], "L_m": PL[(g, t)][2],
            "n_events": PL_N[(g, t)], "n_events_in_class": int(len(edist[(edist["class"].isin(GROUPS[g]))
                                                                    & (edist.t == 0)
                                                                    & np.isfinite(edist.dist_m)]))}
           for g in GROUPS for t in T_GRID]
pd.DataFrame(pl_rows).to_csv(OUT / "profile_likelihood.csv", index=False, float_format="%.6g")
log(f"wrote {OUT / 'profile_likelihood.csv'} ({len(pl_rows)} rows = {len(GROUPS)} groups x {T_GRID.size} "
    f"candidates)")
flush_log()


# ================================================================================================
section("6b. t_hat, the 95% CI, and identifiability")
# ================================================================================================
def runs_of(t) -> list:
    """Contiguous runs of a sorted integer grid, as (first, last) pairs."""
    t = np.asarray(t)
    if t.size == 0:
        return []
    cut = np.flatnonzero(np.diff(t) > 1)
    return [(int(t[a]), int(t[b])) for a, b in zip(np.r_[0, cut + 1], np.r_[cut, t.size - 1])]


PL_SUM = {}
for g in GROUPS:
    ll = np.array([PL[(g, int(t))][0] for t in T_GRID])
    dt = 2.0 * (ll.max() - ll)
    ident = float(ll.max() - ll.min()) >= 3.84  # does the curve move at all?
    ci = runs_of(T_GRID[dt <= 3.84]) if ident else []
    i_best = int(np.argmax(ll))
    t_hat = int(T_GRID[i_best]) if ident else None
    lr = 2.0 * (ll[i_best] - PL_FLAT[(g, t_hat)]) if ident else np.nan
    loc = [i for i in range(1, T_GRID.size - 1) if ll[i] >= ll[i - 1] and ll[i] >= ll[i + 1]]
    loc = sorted(loc, key=lambda i: -ll[i])
    second = next((int(T_GRID[i]) for i in loc if abs(int(T_GRID[i]) - t_hat) > 10), None) if ident else None
    PL_SUM[g] = {"t_hat": t_hat, "ci": ci, "identifiable": ident, "logL_range": float(ll.max() - ll.min()),
                 "n": PL_N[(g, t_hat if ident else int(T_GRID[0]))], "L": PL[(g, t_hat)][2] if ident else np.nan,
                 "lambda0": PL[(g, t_hat)][1] if ident else np.nan, "LR": lr,
                 "second": second, "second_dt": float(2.0 * (ll.max() - ll[np.flatnonzero(T_GRID == second)[0]]))
                 if second is not None else np.nan}
    s_ = PL_SUM[g]
    if not s_["identifiable"]:
        log(f"[{g}] n = {s_['n']}, logL range {s_['logL_range']:.2f} < 3.84 -> t NOT identifiable, no t_hat quoted")
        continue
    span = f"{ci[0][0]}-{ci[-1][1]} s" if len(ci) == 1 else f"{ci[0][0]}-{ci[-1][1]} s (fragmented)"
    n_in = int(sum((b - a) // 5 + 1 for a, b in ci))
    s_["ci_span"], s_["ci_frac"] = (ci[0][0], ci[-1][1]), n_in / T_GRID.size
    s_["sharp"] = s_["ci_frac"] <= 0.1 and len(ci) == 1
    log(f"[{g}] n = {s_['n']}, t_hat = {t_hat} s, 95% CI {span} ({n_in}/{T_GRID.size} candidates"
        + (", a sharp peak" if s_["sharp"] else ", a flat/bumpy plateau") + f"); L = {s_['L']:,.0f} m, "
        f"lambda0 = {s_['lambda0']:.3f}/h, LR(distance) = {lr:.1f}"
        + (f"; 2nd local max at {second} s (2dlogL = {s_['second_dt']:.2f})" if second is not None else ""))
flush_log()


# ================================================================================================
section("8. Placebo: the same profile on random in-service train-instants of the same line")
# ================================================================================================
# If a sharp t_hat is pure geometry -- the carrier sweeps past a station and the victim sits there -- then random
# in-service train-instants of the same line must show a sharp t_hat too. The placebo keeps the calendar, the line and
# the number of events, and drops only the fact that these trains actually broke down.
TRAINS_BY_LINE = {ln: np.unique(TC[trav.line_id.astype(str).values == ln]) for ln in ("10", "2", "8", "12")}
rows_by_cd = {}
for c in np.unique(list(TCODES.values())):
    lo, hi = np.searchsorted(combo_s, c * 1000, "left"), np.searchsorted(combo_s, c * 1000 + 1000, "left")
    rows_by_cd[int(c)] = order[lo:hi]
ev_line = edist_all.drop_duplicates("incident_id").set_index("incident_id").line.to_dict()
ev_day = {r.incident_id: pd.Timestamp(r.onset_ts).normalize() for r in EVENTS.itertuples()}


def placebo_events(rng):
    """One placebo replicate: the same 142 breakdowns replaced by random in-service train-instants of the same line."""
    out = []
    for inc, line in ev_line.items():
        day = ev_day[inc]
        di = DCODES[pd.Timestamp(day)]
        for _ in range(60):  # a train of that line which actually runs that day
            c = int(rng.choice(TRAINS_BY_LINE[line]))
            rr = rows_by_cd[c]
            rr = rr[DCODE_ARR[rr] == di]
            if rr.size:
                break
        dur = (ARR[rr] - DEP[rr]) / 1e9
        r = int(rng.choice(rr, p=dur / dur.sum()))
        ts = pd.Timestamp(DEP[r] / 1e9, unit="s")
        dc = direction_at(CARRIER, day, ts)
        if dc is None:
            continue
        cls = classify(line, int(TRAV_DIR[r]), dc)
        for t in T_GRID:
            at = ts - pd.Timedelta(seconds=int(t))
            v, c_ = locate(TRAIN_NAMES[c], day, at), locate(CARRIER, day, at)
            out.append({"class": cls, "t": int(t), "dist_m": np.nan if (v is None or c_ is None)
                        else float(haversine(v["lat"], v["lon"], c_["lat"], c_["lon"]))})
    return pd.DataFrame(out)


PLACEBO, PLACEBO_CURVE = {}, {}
for rep in range(2):
    rng_p = np.random.default_rng(100 + rep)
    pe = placebo_events(rng_p)
    for gname, ks in GROUPS.items():
        ev_g = pe[pe["class"].isin(ks) & np.isfinite(pe.dist_m)]
        ll = np.array([fit_profile(np.histogram(ev_g.loc[ev_g.t == int(t), "dist_m"], bins=FIT_EDGES)[0],
                                   EXPO_G[gname])[0] for t in T_GRID])
        dt = 2.0 * (ll.max() - ll)
        ci = runs_of(T_GRID[dt <= 3.84])
        PLACEBO_CURVE[(rep, gname)] = dt
        PLACEBO[(rep, gname)] = {"t_hat": int(T_GRID[int(np.argmax(ll))]), "range": float(ll.max() - ll.min()),
                                 "ci_span": (ci[0][0], ci[-1][1]) if ci else None, "n": int(ev_g.t.nunique() and
                                 len(ev_g[ev_g.t == 0]))}
    log(f"placebo replicate {rep + 1}: " + "; ".join(
        f"{g} n={PLACEBO[(rep, g)]['n']} t_hat={PLACEBO[(rep, g)]['t_hat']} s "
        f"range={PLACEBO[(rep, g)]['range']:.1f} CI={PLACEBO[(rep, g)]['ci_span']}" for g in GROUPS))
flush_log()


# ================================================================================================
section("9. The profile-likelihood figure, with the placebo overlaid")
# ================================================================================================
GLAB = {"same_line_same_dir": "same line, same direction", "same_line_opposite": "same line, opposite direction",
        "cross_line_same_dir": "cross line, same direction", "cross_line_opposite": "cross line, opposite direction",
        "same_line": "same line, both directions pooled", "cross_line": "cross line, both directions pooled"}
GCOL = {"same_line_same_dir": "#1f4e9c", "same_line_opposite": "#1f4e9c", "same_line": "#0b2e63",
        "cross_line_same_dir": "#c2570a", "cross_line_opposite": "#c2570a", "cross_line": "#7a3603"}
fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.4))
for ax, g in zip(axes.ravel(), GROUPS):
    ll = np.array([PL[(g, int(t))][0] for t in T_GRID])
    dt = 2.0 * (ll.max() - ll)
    s_ = PL_SUM[g]
    for rep in range(2):  # the placebo: same geometry, same line, no breakdown
        ax.plot(T_GRID, PLACEBO_CURVE[(rep, g)], color="#9a9a9a", lw=1.0, ls="--", zorder=0)
    ax.plot(T_GRID, dt, color=GCOL[g], lw=1.4, zorder=2)
    ax.axhline(3.84, color="#8a8a8a", ls="--", lw=1.0)
    ax.set_ylim(0, 25)
    ax.set_xlim(0, 600)
    ax.set_xticks(range(0, 601, 100))
    ax.grid(alpha=0.22, lw=0.5)
    if s_["identifiable"]:
        ax.axvspan(s_["ci_span"][0], s_["ci_span"][1], color=GCOL[g], alpha=0.16, lw=0)
        ax.axvline(s_["t_hat"], color=GCOL[g], lw=1.6)
        ax.annotate(f"t_hat = {s_['t_hat']} s\n95% CI {s_['ci_span'][0]}-{s_['ci_span'][1]} s\n"
                    f"L = {s_['L']:,.0f} m\n" + ("sharp peak" if s_["sharp"] else "flat plateau"),
                    xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=9, color=GCOL[g],
                    linespacing=1.45, bbox={"fc": "white", "ec": "none", "alpha": 0.78, "pad": 2.0})
    else:
        ax.annotate(f"no t_hat: logL range {s_['logL_range']:.2f} < 3.84", xy=(0.5, 0.55), xycoords="axes fraction",
                    ha="center", fontsize=9.5, color=GCOL[g], style="italic")
    ax.set_title(f"{GLAB[g]} — {s_['n']} breakdowns", fontsize=11, color=GCOL[g])
    ax.set_xlabel("candidate latency $t$ (s)", fontsize=9.5)
    ax.set_ylabel("2 (max logL - logL($t$))", fontsize=9.5)
    ax.tick_params(labelsize=8.5)
axes[0, 0].legend(handles=[Line2D([], [], color="#444444", lw=1.4, label="real breakdowns"),
                           Line2D([], [], color="#9a9a9a", lw=1.0, ls="--", label="placebo (2 replicates)")],
                  loc="upper left", fontsize=8.5, frameon=False)
fig.suptitle("Profile likelihood of the latency: onset rate $\\lambda(d)=\\lambda_0\\exp(-d/L)$ on the background exposure, "
             "121 candidates on a 5-s grid\nthe dashed line is the 95% cut (3.84); the curve is clipped at 25; "
             "the shaded band is the CI span", fontsize=12.5, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FIG / "fig_profile_t.png", dpi=300)
log(f"wrote {FIG / 'fig_profile_t.png'} (6 profile curves)")
flush_log()


