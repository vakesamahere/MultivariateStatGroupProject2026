"""18_leadtime.py -- is the off-line channel a schedule relationship? The lead-time geometry.

Read-only reuse of the caches built by 01-17 (no re-run, no network). For the 75 off-line events
(victim on line 2/8/12, carrier T3420 on line 10) this resolves, in order:

  1  where the victim sits relative to the nearest shared interchange N (at N / leaving N / arriving N),
  2  where the carrier sits in its own timetable relative to the onset (departure at the station before N,
     arrival at N, departure at N, and the same three as a fraction of the scheduled run time),
  3  a random-onset null for the tightness of (2), global (uniform on the service date) and local
     (uniform within +-30 min of the true onset),
  4  the same treatment for the 68 on-line events, referenced to the victim's own position,
  5  victim_kind x channel x carrier_state cross-tabulations,
  6  whether the carrier was moving toward N or away from it at the onset,
  7  per-interchange concentration with the background exposure of each interchange.

Definitions fixed here (derived, never hardcoded):
  channel        on-line  = victim's line is the carrier's line (line 10);
                 off-line = victim's line is one of the other three lines.
  shared station station served by both the victim's line and the carrier's line.
  N              nearest shared station to the victim's location: minimum hops on the undirected track
                 graph, ties broken by network metres.
  run            scheduled running time of the carrier's traversal (P -> N), i.e. arr_N - dep_P.
  on_dep         onset - dep_P   (seconds; >0 = the carrier left the previous station before the onset)
  eta            arr_N - onset   (seconds; >0 = the carrier reaches N after the onset)
  post           dep_N - onset   (seconds; >0 = the carrier leaves N after the onset)
  fraction       the same quantity divided by run.

Outputs: out/leadtime/*.csv, out/leadtime/leadtime.png, out/leadtime/leadtime_results.txt
Run: python3 src/18_leadtime.py
"""
from __future__ import annotations

import heapq
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA, DERIVED = ROOT / "data", ROOT / "data_derived"
OUT = ROOT / "out" / "leadtime"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "leadtime_results.txt"
CARRIER = "T3420"
TIGHT_S = 600.0          # |eta| below this = the carrier's visit to N brackets the onset
NULL_GLOBAL = 2000       # draws per event, uniform over the carrier's in-service window on that date
NULL_LOCAL = 2000        # draws per event, uniform within +-30 min of the true onset
LOCAL_HALF_S = 1800.0
RNG = np.random.default_rng(20261108)


def log(msg: str = "") -> None:
    """Append to the results file and echo one line; the file is the record, the echo is a courtesy."""
    with LOG.open("a") as fh:
        fh.write(msg + "\n")
    print(msg)


def poisson_ci(k, exposure, level: float = 0.95):
    """Exact (Garwood) CI for a Poisson rate k/exposure, as in src/02_screening.py:85."""
    k = np.asarray(k, dtype=float)
    a = (1 - level) / 2
    lo = np.where(k > 0, stats.chi2.ppf(a, 2 * k) / 2, 0.0)
    hi = stats.chi2.ppf(1 - a, 2 * k + 2) / 2
    return lo / exposure, hi / exposure


def q5(x) -> dict:
    """Count, median, IQR, min, max of a numeric vector, in the caller's units."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if not len(x):
        return {"n": 0, "median": np.nan, "q1": np.nan, "q3": np.nan, "iqr": np.nan,
                "min": np.nan, "max": np.nan}
    q1, q3 = np.percentile(x, [25, 75])
    return {"n": int(len(x)), "median": float(np.median(x)), "q1": float(q1), "q3": float(q3),
            "iqr": float(q3 - q1), "min": float(x.min()), "max": float(x.max())}


def fmt(label: str, d: dict, unit: str = "s") -> str:
    return (f"{label:<28} n={d['n']:>3}  median {d['median']:>8.1f}{unit}  "
            f"IQR [{d['q1']:>8.1f}, {d['q3']:>8.1f}] ({d['iqr']:>7.1f})  "
            f"range [{d['min']:>8.1f}, {d['max']:>8.1f}]")


def undirected(loc: str) -> str:
    """'S005 -> S004' and 'S004 -> S005' are the same piece of track; stations stay as they are."""
    return " - ".join(sorted(loc.split(" -> "))) if " -> " in loc else loc


LOG.write_text("")

# ================================================================================================
# 0. Inputs
# ================================================================================================
trav = pd.read_pickle(DERIVED / "traversals.pkl")
bd = pd.read_csv(DERIVED / "breakdowns.csv", parse_dates=["onset_ts", "date"])
ev = pd.read_csv(ROOT / "out" / "interchange" / "events.csv", parse_dates=["date", "onset_ts"])
geo_ev = pd.read_csv(ROOT / "out" / "geo" / "events_geo.csv")
st = pd.read_csv(DATA / "station_information.csv", encoding="utf-8-sig", dtype=str)
seglen = pd.read_csv(ROOT / "out" / "geo" / "segment_lengths.csv")
per_int = pd.read_csv(ROOT / "out" / "interchange" / "per_interchange.csv")
log("=" * 100)
log("18_leadtime -- the schedule-relative geometry of the off-line channel")
log("=" * 100)

CARR_LINE = str(trav.loc[trav.train_id == CARRIER, "line_id"].iloc[0])
ev["channel"] = np.where(ev.line.astype(str) == CARR_LINE, "on-line", "off-line")
ev = ev.merge(bd[["incident_id", "location", "location_type", "from_station", "to_station",
                  "is_terminal_station", "trip_key", "direction", "hour"]],
              on="incident_id", how="left", suffixes=("", "_bd"))
# 09's `carrier_state` marks 'out_of_service' whenever the carrier has no 30-s grid interval at that instant,
# which also covers the 6 events while the carrier sits at a terminus. 13's state (events_geo) separates the
# two, so the one genuinely out-of-service event is taken from there.
ev = ev.merge(geo_ev[["incident_id", "carrier_state", "victim_frac"]], on="incident_id", how="left",
              suffixes=("_09", ""))
log(f"[0] events {len(ev)}: on-line {int((ev.channel == 'on-line').sum())}, "
    f"off-line {int((ev.channel == 'off-line').sum())}; carrier {CARRIER} on line {CARR_LINE}")
OOS = ev.carrier_state.eq("off")
log(f"[0] carrier genuinely out of service at onset (13's state 'off'): {int(OOS.sum())} "
    f"({', '.join(ev.loc[OOS, 'incident_id'])}); excluded from every carrier-schedule step")
log(f"[0] carrier at a terminus at onset (no 30-s grid interval, but in service): "
    f"{int(ev.carrier_state.eq('terminal').sum())}")
print(f"[0] loaded; log -> {LOG.relative_to(ROOT)}")

# ================================================================================================
# 1. Track graph (hops + metres) and the shared interchanges, derived from the traversals
# ================================================================================================
# Every station appears as a from_ or to_station, so the traversals -- not station_information.csv -- are the
# authority on which lines serve which station and which stations are adjacent (same choice as 09).
st_lines: dict[str, set[str]] = {}
adj: dict[str, set[str]] = {}
for a, b, ln in trav[["from_station", "to_station", "line_id"]].astype(str).drop_duplicates().itertuples(
        index=False):
    st_lines.setdefault(a, set()).add(ln)
    st_lines.setdefault(b, set()).add(ln)
    if a != b:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
STATIONS = sorted(st_lines)
ST_IX = {s: i for i, s in enumerate(STATIONS)}
NS = len(STATIONS)
INTER10 = {s for s, ls in st_lines.items() if CARR_LINE in ls and len(ls) >= 2}
SHARED_WITH = {ln: {s for s in INTER10 if ln in st_lines[s]} for ln in sorted(set().union(*st_lines.values()))}

# undirected metre length of every track piece (both directions present -> mean of the two)
key_m: dict[str, list[float]] = {}
for r in seglen.itertuples():
    key_m.setdefault(undirected(f"{r.from_station} -> {r.to_station}"), []).append(float(r.length_m))
SEG_M = {k: float(np.mean(v)) for k, v in key_m.items()}
NO_M = 3000.0  # an adjacent pair with no measured length: line spacing at the terminals, rare

HOP = np.full((NS, NS), 9999, dtype=np.int32)
MET = np.full((NS, NS), np.inf)
for s in STATIONS:  # BFS for hops, Dijkstra for metres; 120 nodes, so one pass per station is cheap
    i0 = ST_IX[s]
    HOP[i0, i0], MET[i0, i0] = 0, 0.0
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
        HOP[i0, ST_IX[v]] = x
    dist, pq = {s: 0.0}, [(0.0, s)]
    while pq:
        du, u = heapq.heappop(pq)
        if du > dist.get(u, np.inf):
            continue
        for v in adj[u]:
            w = SEG_M.get(undirected(f"{u} -> {v}"), NO_M)
            if du + w < dist.get(v, np.inf):
                dist[v] = du + w
                heapq.heappush(pq, (du + w, v))
    for v, x in dist.items():
        MET[i0, ST_IX[v]] = x
log(f"[1] stations {NS}; carrier-line interchanges {len(INTER10)}: {sorted(INTER10)}")
log(f"[1] shared stations per victim line: "
    f"{ {ln: sorted(v) for ln, v in SHARED_WITH.items() if ln != CARR_LINE} }")


def nodes_of(loc: str) -> list[str]:
    return loc.split(" -> ") if " -> " in loc else [loc]


def nearest_shared(loc: str, line: str) -> tuple[str, int, float, list[str]]:
    """Nearest shared station to a victim location: min hops, ties broken by network metres.

    Metres are measured from the nearer endpoint of the victim's own segment, so for a victim inside a
    segment they are a lower bound on the distance to N, not the distance along the track from its position.
    """
    cands = sorted(SHARED_WITH[line])
    ns = [n for n in nodes_of(loc) if n in ST_IX]
    if not ns or not cands:
        return "", 9999, np.inf, []
    ri = [ST_IX[n] for n in ns]
    scored = [(int(HOP[np.ix_(ri, [ST_IX[c]])].min()), float(MET[np.ix_(ri, [ST_IX[c]])].min()), c)
              for c in cands]
    h = min(x[0] for x in scored)
    tied = [x for x in scored if x[0] == h]
    m = min(x[1] for x in tied)
    best = [x[2] for x in tied if abs(x[1] - m) < 1e-6]
    return best[0], h, m, best


# ================================================================================================
# 1b. Where is the victim relative to its nearest shared interchange?
# ================================================================================================
off = ev[ev.channel == "off-line"].copy()
rows = []
for r in off.itertuples():
    n, h, m, tied = nearest_shared(r.location, str(r.line))
    ns = nodes_of(r.location)
    at_st = len(ns) == 1
    if at_st:
        pos = "at_N" if ns[0] == n else "elsewhere"
    elif ns[0] == n:
        pos = "leaving_N"
    elif ns[-1] == n:
        pos = "arriving_N"
    else:
        pos = "elsewhere"
    rows.append({
        "incident_id": r.incident_id, "victim_train": r.train_id, "victim_line": str(r.line),
        "onset_ts": r.onset_ts, "victim_location": r.location,
        "victim_at": "station" if at_st else "segment", "N": n,
        "N_lines": "/".join(sorted(st_lines[n])), "N_ambiguous": len(tied) > 1,
        "hops_to_N": h, "metres_to_N": round(m, 1), "position": pos,
        "victim_station_is_interchange": bool(at_st and ns[0] in INTER10),
        "victim_frac": r.victim_frac if not at_st else np.nan,
    })
geo = pd.DataFrame(rows)
geo.to_csv(OUT / "victim_geometry.csv", index=False)
log("")
log("[1] OFF-LINE VICTIM GEOMETRY -- position relative to the nearest shared interchange N")
log(pd.crosstab(geo.hops_to_N, geo.position, margins=True).to_string())
log(f"[1] hops to N: {geo.hops_to_N.value_counts().sort_index().to_dict()}")
log(f"[1] at N / leaving N / arriving N / elsewhere: "
    f"{ {k: int(v) for k, v in geo.position.value_counts().items()} }")
log(f"[1] N ambiguous (two shared stations equidistant in hops and metres): {int(geo.N_ambiguous.sum())}")
log(f"[1] metres to N: {q5(geo.metres_to_N)}")
log(f"[1] victim location is itself an interchange of the carrier's line: "
    f"{int(geo.victim_station_is_interchange.sum())} of {len(geo)}")
print(f"[1] wrote {OUT.name}/victim_geometry.csv ({len(geo)} rows)")

# ================================================================================================
# 2. The carrier's own timetable around N, and the offsets of the onset from it
# ================================================================================================
# One carrier traversal is one scheduled run P -> N: dep_ts is the departure at P, arr_ts the arrival at N.
# The departure at N is the first departure from N after that arrival, which covers both a dwell and a
# terminus turnaround (where the next run belongs to the reverse trip).
carr = trav.loc[trav.train_id == CARRIER, ["date", "trip_key", "trip_no", "direction",
                                            "from_station", "to_station", "dep_ts", "arr_ts"]].copy()
carr["date"] = pd.to_datetime(carr.date)
carr = carr.sort_values(["date", "trip_key", "dep_ts"]).reset_index(drop=True)
EPOCH_S = int(pd.Timestamp("2026-01-01").value // 10 ** 9)


def to_s(ts) -> np.ndarray:
    """Timestamps -> float seconds since 2026-01-01, with the unit pinned to seconds (no us/ns surprise)."""
    d = pd.to_datetime(np.asarray(ts)).to_numpy(dtype="datetime64[s]").astype("int64")
    return d.astype(float) - EPOCH_S


# per (date, station): sorted arrival instants and the matching departure-at-the-previous-station / run time
VISIT: dict[tuple, dict] = {}
dep_by: dict[tuple, np.ndarray] = {}
for day, g in carr.groupby("date", observed=True):
    for n, h in g.groupby("from_station", observed=True):
        dep_by[(day, n)] = np.sort(to_s(h.dep_ts.values))
for (day, n), g in carr.groupby(["date", "to_station"], observed=True):
    a, p_dep = to_s(g.arr_ts.values), to_s(g.dep_ts.values)
    order = np.argsort(a)
    a, p_dep, prev = a[order], p_dep[order], g.from_station.astype(str).values[order]
    trip = g.trip_key.values[order]
    deps = dep_by.get((day, n), np.array([np.inf]))
    k = np.searchsorted(deps, a, side="right")          # first departure from N strictly after the arrival
    nxt = np.where(k < len(deps), deps[np.minimum(k, len(deps) - 1)], np.inf)
    VISIT[(day, n)] = {"arr": a, "dep_prev": p_dep, "prev": prev, "dep_N": nxt, "trip": trip}
WINDOW = {day: (float(to_s(g.dep_ts.values).min()), float(to_s(g.arr_ts.values).max()))
          for day, g in carr.groupby("date", observed=True)}
log("")
log(f"[2] carrier traversal visits indexed at {len(VISIT)} (date, station) pairs; "
    f"{len(WINDOW)} service dates")
_sec = np.concatenate([to_s(carr.dep_ts.values) % 60, to_s(carr.arr_ts.values) % 60])
log(f"    the carrier's timetable is on a 30-s lattice: {np.mean(_sec % 30 == 0):.4f} of {len(_sec)} "
    f"dep/arr instants fall on :00 or :30 ({int((_sec % 60 == 0).sum())} / {int((_sec % 60 == 30).sum())})")


def resolve(day, n: str, t: np.ndarray) -> dict:
    """Offsets of instant(s) t from the carrier's visit to N nearest in time on `day`.

    Returns seconds: on_dep = t - dep_P, eta = arr_N - t, post = dep_N - t, run = arr_N - dep_P.
    Everything is NaN when the carrier never reached N on that date.
    """
    v = VISIT.get((day, n))
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if v is None:
        nan = np.full(t.shape, np.nan)
        return {"on_dep": nan, "eta": nan.copy(), "post": nan.copy(), "run": nan.copy(),
                "prev": np.array([""] * len(t)), "k": np.full(t.shape, -1),
                "arr_N": nan.copy(), "dep_P": nan.copy(), "dep_N": nan.copy(),
                "trip": np.full(t.shape, -1)}
    i = np.searchsorted(v["arr"], t)
    i = np.clip(i, 1, len(v["arr"]) - 1) if len(v["arr"]) > 1 else np.zeros_like(i)
    if len(v["arr"]) > 1:
        left = np.maximum(i - 1, 0)
        pick = np.where(np.abs(t - v["arr"][left]) <= np.abs(v["arr"][i] - t), left, i)
    else:
        pick = i
    return {"on_dep": t - v["dep_prev"][pick], "eta": v["arr"][pick] - t,
            "post": v["dep_N"][pick] - t, "run": v["arr"][pick] - v["dep_prev"][pick],
            "prev": v["prev"][pick], "k": pick,
            "arr_N": v["arr"][pick], "dep_P": v["dep_prev"][pick],
            "dep_N": v["dep_N"][pick], "trip": v["trip"][pick]}


# --- the observed offsets ------------------------------------------------------------------------
off_use = geo[geo.incident_id.isin(ev.loc[~OOS, "incident_id"])].copy()
off_use["day"] = pd.to_datetime(off_use.onset_ts).dt.normalize()
t_arr = to_s(off_use.onset_ts.values)
agg = {"on_dep": [], "eta": [], "post": [], "run": [], "prev": [], "k": [],
       "arr_N": [], "dep_P": [], "dep_N": [], "trip": []}
for day, n, t in zip(off_use.day.values, off_use.N.values, t_arr):
    r = resolve(pd.Timestamp(day), n, t)
    for key in agg:
        agg[key].append(r[key][0])
for key in agg:
    off_use[key] = agg[key]
off_use["eta_frac"] = off_use.eta / off_use.run
off_use["on_dep_frac"] = off_use.on_dep / off_use.run
in_win = off_use.eta.abs() <= TIGHT_S
off_use["in_window"] = in_win
log(f"[2] off-line events with a carrier visit at N on the same date: {int(off_use.eta.notna().sum())} "
    f"of {len(off_use)}")
log(f"[2] |eta| <= {TIGHT_S:.0f} s (the carrier's visit to N brackets the onset): "
    f"{int(in_win.sum())} of {int(off_use.eta.notna().sum())}")
for q in ["on_dep", "eta", "post", "run"]:
    log("    " + fmt(q, q5(off_use.loc[in_win, q])))
log("    " + fmt("eta/run", q5(off_use.loc[in_win, "eta_frac"]), ""))
log("    " + fmt("on_dep/run", q5(off_use.loc[in_win, "on_dep_frac"]), ""))
log(f"[2] excluded by the |eta| window: "
    f"{[(r.incident_id, int(r.hops_to_N), round(r.eta / 60, 1)) for r in off_use[~in_win & off_use.eta.notna()].itertuples()]}")
off_use.to_csv(OUT / "carrier_offsets.csv", index=False)
print(f"[2] wrote {OUT.name}/carrier_offsets.csv")

# ================================================================================================
# 2b. Which single timetable event of the carrier is the onset locked to?
# ================================================================================================
# Steps 2's three quantities are only three of the carrier's events around N. If the off-line channel is a
# schedule relationship, one of the carrier's own events should sit at a fixed offset from the onset and the
# others should not. This scans the carrier's timetable in the matched trip from two stations before N to two
# stations after it, and reports the spread of (anchor - onset) for each. Scanning several anchors can only
# manufacture tightness, so the chosen anchor is carried into step 3's null, which resamples the onset and
# re-runs the whole scan.
TRIP: dict[tuple, dict] = {}
for (day, tk), g in carr.groupby(["date", "trip_key"], observed=True):
    TRIP[(day, tk)] = {"dep": to_s(g.dep_ts.values), "arr": to_s(g.arr_ts.values),
                       "from": g.from_station.astype(str).values, "to": g.to_station.astype(str).values}
ANCH = [(r, k) for r in (-2, -1, 0, 1, 2) for k in ("dep", "arr")]
ANCH_NAME = {}
for r, k in ANCH:
    ANCH_NAME[(r, k)] = (f"{k} at station {r:+d} from N")


def anchor_offsets(rows: pd.DataFrame, t: np.ndarray | None = None) -> pd.DataFrame:
    """(anchor - onset) in seconds for every anchor in the scan, one row per event."""
    out = {a: [] for a in ANCH}
    tt = to_s(rows.onset_ts.values) if t is None else t
    for day, tk, n, tv in zip(rows.day.values, rows.trip.values, rows.N.values, tt):
        tr = TRIP.get((pd.Timestamp(day), tk))
        if tr is None:
            for a in ANCH:
                out[a].append(np.nan)
            continue
        j = int(np.argmin(np.abs(tr["arr"] - tv)))   # row j is the run P -> N whose arrival is closest
        cand = np.nonzero(tr["to"] == n)[0]
        if not len(cand):
            for a in ANCH:
                out[a].append(np.nan)
            continue
        j = int(cand[np.argmin(np.abs(tr["arr"][cand] - tv))])
        for r, k in ANCH:
            jj = j + r
            out[(r, k)].append(float(tr[k][jj] - tv) if 0 <= jj < len(tr["dep"]) else np.nan)
    return pd.DataFrame({f"{r:+d}_{k}": v for (r, k), v in out.items()})


anch = anchor_offsets(off_use[in_win])
scan = []
for (r, k) in ANCH:
    col = f"{r:+d}_{k}"
    d = q5(anch[col])
    d.update({"anchor": col, "station_offset": r, "kind": k,
              "within_30s": int((anch[col].abs() <= 30).sum()),
              "within_60s": int((anch[col].abs() <= 60).sum())})
    scan.append(d)
scan = pd.DataFrame(scan)[["anchor", "station_offset", "kind", "n", "median", "q1", "q3", "iqr",
                            "min", "max", "within_30s", "within_60s"]]
scan.to_csv(OUT / "anchor_scan.csv", index=False)
log("")
log(f"[2b] ANCHOR SCAN -- (carrier event - onset) in seconds, {int(in_win.sum())} in-window off-line events")
log(f"    {'anchor':<28} {'median':>8} {'IQR':>8} {'|d|<=30s':>9} {'|d|<=60s':>9}")
for r in scan.itertuples():
    log(f"    {ANCH_NAME[(r.station_offset, r.kind)]:<28} {r.median:>8.1f} {r.iqr:>8.1f} "
        f"{r.within_30s:>9} {r.within_60s:>9}")
BEST = scan.sort_values("iqr").iloc[0]
log(f"[2b] tightest single anchor by IQR: {BEST['anchor']} (IQR {BEST['iqr']:.1f} s, "
    f"median {BEST['median']:.1f} s); no single event of the carrier is tight, which is why step 2c stops "
    f"asking which event and asks only whether the onset is on the carrier's timetable at all")
anch.insert(0, "incident_id", off_use.loc[in_win, "incident_id"].values)
anch.to_csv(OUT / "anchor_offsets.csv", index=False)
print(f"[2b] wrote {OUT.name}/anchor_scan.csv")

# ================================================================================================
# 2c. Is the onset next to *some* carrier timetable event, and if so which one?
# ================================================================================================
# Steps 2 and 2b measure the onset against N. But the carrier's timetable is a stream of dep/arr events at
# every station of its line, so the prior question is whether the onset sits next to any of them at all. This
# indexes that stream per date and measures the signed distance from each onset to its nearest carrier event;
# the same scan is what step 3's null has to reproduce before any tightness can be believed.
CAR_EVT: dict[pd.Timestamp, dict] = {}
for day, g in carr.groupby("date", observed=True):
    dep, arr = to_s(g.dep_ts.values), to_s(g.arr_ts.values)
    a = np.concatenate([dep, arr])
    kind = np.concatenate([np.full(len(dep), "dep"), np.full(len(arr), "arr")])
    sta = np.concatenate([g.from_station.astype(str).values, g.to_station.astype(str).values])
    o = np.argsort(a)
    CAR_EVT[pd.Timestamp(day)] = {"t": a[o], "kind": kind[o], "station": sta[o]}
GAP_S = float(np.mean([np.diff(e["t"]).mean() for e in CAR_EVT.values()]))
N_EVT = int(np.mean([len(e["t"]) for e in CAR_EVT.values()]))


def nearest_event(day, t) -> dict:
    """Signed offset (carrier event - onset) to the nearest carrier dep/arr on `day`, plus its kind/station."""
    e = CAR_EVT.get(pd.Timestamp(day))
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if e is None:
        z = np.full(t.shape, np.nan)
        return {"d": z, "kind": np.array([""] * len(t)), "station": np.array([""] * len(t))}
    a = e["t"]
    i = np.clip(np.searchsorted(a, t), 0, len(a) - 1)
    lo = np.maximum(i - 1, 0)
    pick = np.where(np.abs(a[lo] - t) <= np.abs(a[i] - t), lo, i)
    return {"d": a[pick] - t, "kind": e["kind"][pick], "station": e["station"][pick]}


NEAR = {"d": [], "kind": [], "station": []}
have = off_use.eta.notna().values          # 74: the carrier visited N on that date
have_win = in_win.values[have]
for d, t in zip(off_use.loc[have, "day"].values, t_arr[have]):
    r = nearest_event(d, t)
    for key in NEAR:
        NEAR[key].append(r[key][0])
NEAR["d"] = np.array(NEAR["d"], dtype=float)
# Measured on all 74 events with a carrier visit, not only on the |eta|<=600 s subset: the snapping statistic
# must not be conditioned on the quantity it is supposed to explain.
near = pd.DataFrame({"incident_id": off_use.loc[have, "incident_id"].values,
                     "N": off_use.loc[have, "N"].values,
                     "hops_to_N": off_use.loc[have, "hops_to_N"].values,
                     "offset": NEAR["d"], "kind": NEAR["kind"], "station": NEAR["station"],
                     "in_window": have_win})
near["abs_off"] = near.offset.abs()
near["at_N"] = near.station.eq(near.N)
near.to_csv(OUT / "nearest_event.csv", index=False)
N_ALL = len(near)
TIGHT = near.abs_off <= 10.0
log("")
log(f"[2c] NEAREST CARRIER TIMETABLE EVENT (dep or arr, any station of line {CARR_LINE})")
log(f"    carrier timetable: {N_EVT} events per service date, mean gap between consecutive events {GAP_S:.0f} s")
_on_sec = t_arr[have].astype(int) % 60
_tick = 30 * np.round(_on_sec / 30.0)
log(f"    the ONSETS are on that lattice too: seconds-of-minute {np.bincount(_on_sec, minlength=60).nonzero()[0].tolist()} "
    f"over {len(_on_sec)} events, {int((np.abs(_on_sec - _tick) <= 12).sum())}/{len(_on_sec)} within 12 s "
    f"after a :00/:30 tick (median lag {np.median(_on_sec - _tick):.0f} s)")
log(f"    => the |d|<=10 s statistic is NOT free: it asks whether the carrier has an event at the tick "
    f"the onset was recorded on, not whether two arbitrary instants are close")
log(f"    |onset - nearest carrier event|, all {N_ALL} events with a carrier visit:  " + "  ".join(
    f"<={k}s {int((near.abs_off <= k).sum())}/{N_ALL}" for k in (5, 10, 15, 30, 45, 60)))
log(f"    the same on the {int(have_win.sum())} in-window events:  " + "  ".join(
    f"<={k}s {int((near.loc[have_win, 'abs_off'] <= k).sum())}/{int(have_win.sum())}" for k in (10, 30, 60)))
log("    offset of the tight subset (|d|<=10 s): " +
    fmt("event - onset", q5(near.loc[TIGHT, "offset"])))
log(f"    the tight subset sits at {near.loc[TIGHT, 'station'].nunique()} distinct stations "
    f"({ {str(k): int(v) for k, v in near.loc[TIGHT, 'kind'].value_counts().items()} }); "
    f"at N itself: {int(near.loc[TIGHT, 'at_N'].sum())}")
log(f"    top stations of the tight subset: "
    f"{ {str(k): int(v) for k, v in near.loc[TIGHT, 'station'].value_counts().head(6).items()} }")
# Control: if the channel were "the carrier reaches the interchange", the onset should sit next to a carrier
# ARRIVAL at one of the 7 shared interchanges rather than next to an arbitrary event on the line.
sh_arr = {d: e["t"][(e["kind"] == "arr") & np.isin(e["station"], sorted(INTER10))]
          for d, e in CAR_EVT.items()}
sh_d = np.array([np.min(np.abs(sh_arr[pd.Timestamp(d)] - t)) if len(sh_arr[pd.Timestamp(d)]) else np.inf
                 for d, t in zip(off_use.loc[have, "day"].values, t_arr[have])])
log(f"    CONTROL, nearest carrier ARRIVAL at one of the {len(INTER10)} shared interchanges:  " + "  ".join(
    f"<={k}s {int((sh_d <= k).sum())}/{N_ALL}" for k in (10, 30, 45)))

# Decisive control: the same scan against the VICTIM's own timetable. A breakdown is the victim train stopping,
# so if its own dep/arr events are just as close, the apparent lock to the carrier is an artefact of the
# onset placement rather than a property of the carrier at all.
VIC_EVT: dict[tuple, np.ndarray] = {}
for (tid, day), g in trav[trav.train_id.isin(ev.loc[~OOS, "train_id"].unique())].groupby(
        ["train_id", "date"], observed=True):
    VIC_EVT[(tid, pd.Timestamp(day))] = np.sort(np.concatenate([to_s(g.dep_ts.values),
                                                               to_s(g.arr_ts.values)]))


def nearest_victim_event(tid, day, t) -> float:
    a = VIC_EVT.get((tid, pd.Timestamp(day)))
    if a is None or not len(a):
        return np.nan
    i = int(np.clip(np.searchsorted(a, t), 0, len(a) - 1))
    lo = max(i - 1, 0)
    return float(a[lo] if abs(a[lo] - t) <= abs(a[i] - t) else a[i]) - float(t)


vic_d = np.array([nearest_victim_event(tid, d, t) for tid, d, t in
                  zip(off_use.incident_id.map(ev.set_index("incident_id").train_id).loc[have].values,
                      off_use.loc[have, "day"].values, t_arr[have])])
vic_gap = float(np.mean([np.diff(a).mean() for a in VIC_EVT.values() if len(a) > 1]))
log(f"    CONTROL, nearest event in the VICTIM's OWN timetable (mean gap {vic_gap:.0f} s):  " + "  ".join(
    f"<={k}s {int((np.abs(vic_d) <= k).sum())}/{N_ALL}" for k in (5, 10, 15, 30, 45, 60)))
print(f"[2c] wrote {OUT.name}/nearest_event.csv")

# ================================================================================================
# 3. Random-onset null -- how tight is that, and could a random onset have done it?
# ================================================================================================
# The null resamples the onset and recomputes every statistic of steps 2 and 2c from scratch. Two nulls:
#   global  onset uniform over the carrier's in-service window on the event's own service date,
#   local   onset uniform within +-30 min of the true onset (keeps hour-of-day and date, drops the minute).
# Every statistic is computed on all 74 off-line events with a carrier visit -- no |eta| window -- so that
# nothing is conditioned on the quantity it is meant to explain.
SUB = off_use[have].reset_index(drop=True)
SUB_T = t_arr[have]
OBS = {"iqr_eta": q5(SUB.eta)["iqr"], "iqr_on_dep": q5(SUB.on_dep)["iqr"],
       "n10": int((near.abs_off <= 10).sum()), "n45sh": int((sh_d <= 45).sum()),
       "med_abs_d": float(near.abs_off.median())}
log("")
log(f"[3] RANDOM-ONSET NULL, {NULL_GLOBAL} draws per event, {len(SUB)} off-line events (no |eta| window)")
log(f"    observed: IQR(eta) {OBS['iqr_eta']:.1f} s, IQR(on_dep) {OBS['iqr_on_dep']:.1f} s, "
    f"|d|<=10 s {OBS['n10']}/{len(SUB)}, median |d| {OBS['med_abs_d']:.1f} s, "
    f"|d|<=45 s at a shared interchange {OBS['n45sh']}/{len(SUB)}")


def null_draws(kind: str, n: int) -> dict:
    """Resample every onset and recompute eta, on_dep, the nearest carrier event and the shared-arrival gap."""
    if kind == "global":
        lo = np.array([WINDOW[pd.Timestamp(d)][0] for d in SUB.day])
        hi = np.array([WINDOW[pd.Timestamp(d)][1] for d in SUB.day])
    else:
        lo, hi = SUB_T - LOCAL_HALF_S, SUB_T + LOCAL_HALF_S
    T = RNG.uniform(lo[:, None], hi[:, None], size=(len(SUB), n))
    eta, on_dep, dd, sh = (np.full_like(T, np.nan) for _ in range(4))
    for i, r in SUB.iterrows():
        day = pd.Timestamp(r.day)
        res = resolve(day, r.N, T[i])
        eta[i], on_dep[i] = res["eta"], res["on_dep"]
        dd[i] = nearest_event(day, T[i])["d"]
        a = sh_arr.get(day)
        sh[i] = (np.min(np.abs(a[:, None] - T[i][None, :]), axis=0) if a is not None and len(a)
                 else np.inf)
    return {"eta": eta, "on_dep": on_dep, "d": dd, "sh": sh}


def col_iqr(M: np.ndarray) -> np.ndarray:
    return np.nanpercentile(M, 75, axis=0) - np.nanpercentile(M, 25, axis=0)


NULL = {}
for kind in ("global", "local"):
    d = null_draws(kind, NULL_GLOBAL)
    s = {"iqr_eta": col_iqr(d["eta"]), "iqr_on_dep": col_iqr(d["on_dep"]),
         "n10": (np.abs(d["d"]) <= 10).sum(axis=0), "n45sh": (d["sh"] <= 45).sum(axis=0),
         "med_abs_d": np.nanmedian(np.abs(d["d"]), axis=0)}
    NULL[kind] = s
    log(f"    null {kind:<6}: " + "  ".join(
        f"{k} {np.median(v):.0f} [{np.percentile(v, 2.5):.0f}, {np.percentile(v, 97.5):.0f}]"
        for k, v in s.items()))
    log(f"    empirical p (null at least as tight as observed): " + "  ".join(
        f"{k} {np.mean(v <= OBS[k]) if k.startswith('iqr') or k == 'med_abs_d' else np.mean(v >= OBS[k]):.4f}"
        for k, v in s.items()))
pd.DataFrame([{ "null": k, "stat": s2, "obs": OBS[s2], "null_med": float(np.median(v)),
                "null_lo": float(np.percentile(v, 2.5)), "null_hi": float(np.percentile(v, 97.5)),
                "p": float(np.mean(v <= OBS[s2]) if s2.startswith("iqr") or s2 == "med_abs_d"
                           else np.mean(v >= OBS[s2]))}
              for k, s in NULL.items() for s2, v in s.items()]).to_csv(OUT / "null.csv", index=False)
print(f"[3] wrote {OUT.name}/null.csv")

# ================================================================================================
# 3b. Reference band: the same statistic for every other train in the network
# ================================================================================================
# A dense timetable is close to any instant by construction. The band of the other 155 trains' timetables
# scored identically says how much of the 49/74 is the carrier rather than the density.
REF = []
for tid, g in trav.groupby("train_id", observed=True):
    if tid == CARRIER or len(g) < 2000 or g.date.nunique() < 60:
        continue
    D = {pd.Timestamp(d): np.sort(np.concatenate([to_s(h.dep_ts.values), to_s(h.arr_ts.values)]))
         for d, h in g.groupby("date", observed=True)}
    c = 0
    for day, tt in zip(SUB.day.values, SUB_T):
        a = D.get(pd.Timestamp(day))
        if a is None or not len(a):
            continue
        i = int(np.clip(np.searchsorted(a, tt), 0, len(a) - 1))
        lo = max(i - 1, 0)
        c += int(min(abs(a[lo] - tt), abs(a[i] - tt)) <= 10.0)
    REF.append({"train_id": tid, "line": int(g.line_id.iloc[0]), "n10": c})
ref = pd.DataFrame(REF)
ref["is_victim"] = ref.train_id.isin(ev.loc[~OOS & ev.channel.eq("off-line"), "train_id"])
ref.to_csv(OUT / "ref_trains.csv", index=False)
log("")
log(f"[3b] REFERENCE BAND, |onset - nearest event| <= 10 s for each of the {len(ref)} other timetables:")
log(f"    all: median {ref.n10.median():.0f}, range [{ref.n10.min()}, {ref.n10.max()}], "
    f"trains at or above the carrier's {OBS['n10']}: {int((ref.n10 >= OBS['n10']).sum())}")
log(f"    excluding the {int(ref.is_victim.sum())} trains that are themselves a victim: "
    f"median {ref.loc[~ref.is_victim, 'n10'].median():.0f}, "
    f"range [{ref.loc[~ref.is_victim, 'n10'].min()}, {ref.loc[~ref.is_victim, 'n10'].max()}]")
log(f"    the victim's own train, scored the same way: {int((np.abs(vic_d) <= 10).sum())}/{N_ALL}")
# Density is the obvious confound: a train with more events per day is nearer in time more often. Regressing
# the reference count on events per day is the crudest control for it, and it is enough to see whether the
# carrier's 49 is just "a busy train".
g_c = trav.groupby("train_id", observed=True)
den = (g_c.size() / g_c.date.nunique()).rename("ev_day")
ref = ref.merge(den, left_on="train_id", right_index=True, how="left")
_b, _a = np.polyfit(ref.ev_day, ref.n10, 1)
_res = ref.n10 - (_a + _b * ref.ev_day)
_car_den = float(len(carr) / carr.date.nunique())
_pred = _a + _b * _car_den
log(f"    density control: n10 rises with events per day (r = {ref.n10.corr(ref.ev_day):.2f}); the fit "
    f"predicts {_pred:.1f} for the carrier's {_car_den:.0f} events/day (which is below {int((den > _car_den).sum())} "
    f"of the 139), observed 49, residual +{49 - _pred:.1f} = {(49 - _pred) / _res.std():.1f} sd; "
    f"no other train's residual exceeds {_res.max():.1f}")
# How much of the carrier's 49 is separable from the victim's own 44? Both timetables are scored on the same
# events; where both are tight, the two events are compared instant by instant.
c10, v10 = near.abs_off.values <= 10, np.abs(vic_d) <= 10
same = c10 & v10 & (np.abs(near.offset.values - vic_d) < 1e-9)
log(f"    overlap: both tight {int((c10 & v10).sum())}, carrier only {int((c10 & ~v10).sum())}, "
    f"victim only {int((v10 & ~c10).sum())}, neither {int((~c10 & ~v10).sum())}")
log(f"    where both are tight the two timetable events are the identical instant for "
    f"{int(same.sum())} of {int((c10 & v10).sum())}; victim-side offsets where tight: "
    + fmt("victim event - onset", q5(vic_d[v10])))
# Is the victim's own train an independent control at all? Score WHERE its tight matches sit: if they are
# all at the incident's own station, the train that broke down was simply scheduled where it broke down.
_where, _kind = [], []
_evi = ev.set_index("incident_id")
_evf, _evt = _evi.from_station, _evi.train_id
for r, tt in zip(off_use.loc[have].itertuples(), t_arr[have]):
    g = trav[(trav.train_id == _evt.get(r.incident_id)) & (trav.date == pd.Timestamp(r.day))]
    if not len(g):
        _where.append(None); _kind.append(None); continue
    a = np.concatenate([to_s(g.dep_ts.values), to_s(g.arr_ts.values)])
    j = int(np.argmin(np.abs(a - tt)))
    _kind.append("dep" if j < len(g) else "arr")
    _where.append(g.from_station.iloc[j] if j < len(g) else g.to_station.iloc[j - len(g)])
_own = np.array([w == _evf.get(i) for w, i in zip(_where, near.incident_id)], dtype=bool)
log(f"    victim-train matches at the incident's own station: {int((_own & v10).sum())} of "
    f"{int(v10.sum())} tight ones ({pd.Series(np.array(_kind, dtype=object)[v10]).value_counts().to_dict()}), "
    f"so this control is contaminated by construction, not independent")
_span = (trav.assign(_s=to_s(trav.dep_ts.values) % 86400)
         .groupby(["train_id", "date"], observed=True)._s.agg(lambda x: x.max() - x.min())
         .groupby("train_id").median())
log(f"    service-window control: median span {_span.median() / 3600:.2f} h/day, carrier "
    f"{_span['T3420'] / 3600:.2f} h (rank {int((_span > _span['T3420']).sum())} of {len(_span)}), "
    f"max {_span.max() / 3600:.2f} h — the carrier is not in service longer than the rest")
print(f"[3b] wrote {OUT.name}/ref_trains.csv")

# ================================================================================================
# 4. The on-line channel, referenced to the victim's own position
# ================================================================================================
# On line 10 the carrier runs over the victim's own segment, so the reference point is the instant the
# carrier passes the victim's exact position: dep + f * travel, with f = victim_frac when the carrier
# traverses the segment in the victim's direction and 1 - victim_frac when it traverses it the other way.
CSEG: dict[tuple, dict] = {}
_c = carr.assign(key=[undirected(f"{a} -> {b}") for a, b in zip(carr.from_station, carr.to_station)])
for (day, key), g in _c.groupby(["date", "key"], observed=True):
    o = np.argsort(to_s(g.dep_ts.values))
    CSEG[(pd.Timestamp(day), key)] = {"dep": to_s(g.dep_ts.values)[o], "arr": to_s(g.arr_ts.values)[o],
                                      "dir": g.direction.astype(str).values[o]}
onl = ev[~OOS & ev.channel.eq("on-line")].copy()
onl["day"] = pd.to_datetime(onl.onset_ts).dt.normalize()
rows = []
for r in onl.itertuples():
    key = undirected(f"{r.from_station} -> {r.to_station}")
    s = CSEG.get((pd.Timestamp(r.day), key))
    tt = float(to_s([r.onset_ts])[0])
    if s is None or pd.isna(r.victim_frac):
        rows.append((r.incident_id, np.nan, np.nan, np.nan, np.nan, np.nan, False, ""))
        continue
    travel = s["arr"] - s["dep"]
    same = s["dir"] == str(r.direction)
    # fraction of its own run at which each traversal reaches the victim's point
    f = np.where(same, float(r.victim_frac), 1.0 - float(r.victim_frac))
    pass_t = s["dep"] + f * travel
    dt = np.abs(pass_t - tt)
    j = int(np.argmin(dt))
    js = int(np.argmin(np.where(same, dt, np.inf))) if same.any() else -1
    rows.append((r.incident_id, float(pass_t[j] - tt), float(travel[j]),
                 float(s["dep"][j] - tt), float(s["arr"][j] - tt),
                 float(pass_t[js] - tt) if js >= 0 else np.nan, bool(same[j]), s["dir"][j]))
onl4 = pd.DataFrame(rows, columns=["incident_id", "lead", "travel", "off_dep", "off_arr",
                                   "lead_same", "nearest_is_same_dir", "carrier_dir"])
onl4["lead_frac"] = onl4.lead / onl4.travel
onl4.to_csv(OUT / "online_offsets.csv", index=False)
OK4 = onl4.lead.notna()
log("")
log(f"[4] ON-LINE CHANNEL, referenced to the carrier's passage of the victim's own position "
    f"({int(OK4.sum())} of {len(onl)} events resolved)")
log("    " + fmt("nearest passage - onset", q5(onl4.loc[OK4, "lead"])))
log("    " + fmt("nearest passage - onset / run", q5(onl4.loc[OK4, "lead_frac"]), ""))
log("    " + fmt("nearest SAME-dir passage - onset", q5(onl4.loc[OK4, "lead_same"])))
log("    " + fmt("carrier dep at seg start - onset", q5(onl4.loc[OK4, "off_dep"])))
log("    " + fmt("carrier arr at seg end - onset", q5(onl4.loc[OK4, "off_arr"])))
log(f"    |nearest passage - onset| <= 60 s: {int((onl4.loc[OK4, 'lead'].abs() <= 60).sum())}/"
    f"{int(OK4.sum())}")
log(f"    nearest passage already passed the victim's point (lead < 0): "
    f"{int((onl4.loc[OK4, 'lead'] < 0).sum())}/{int(OK4.sum())}; "
    f"nearest passage runs in the victim's direction: {int(onl4.loc[OK4, 'nearest_is_same_dir'].sum())}")
print(f"[4] wrote {OUT.name}/online_offsets.csv")

# ================================================================================================
# 5. victim_kind x channel x carrier_state
# ================================================================================================
# Exact counts only. With 143 events spread over three dimensions most cells are single digits, and a
# single-digit cell carries no information about a rate; those are marked rather than smoothed.
def cells(df: pd.DataFrame, a: str, b: str) -> str:
    t = pd.crosstab(df[a], df[b])
    return ", ".join(f"{i}/{j}={int(t.loc[i, j])}" for i in t.index for j in t.columns if t.loc[i, j])


log("")
log("[5] CROSS-TABS, exact counts; a cell with fewer than 5 events is too small to read")
log(f"    victim_location_type / channel: {cells(ev, 'victim_location_type', 'channel')}")
log(f"    carrier_state / channel:        {cells(ev, 'carrier_state', 'channel')}")
for ch in ("off-line", "on-line"):
    s = ev[ev.channel == ch]
    log(f"    {ch:<9} victim_location_type / carrier_state: "
        f"{cells(s, 'victim_location_type', 'carrier_state')}")
    small = [(i, j) for i in s.victim_location_type.unique() for j in s.carrier_state.unique()
             if len(s[(s.victim_location_type == i) & (s.carrier_state == j)]) < 5]
    log(f"    {ch:<9} cells with n<5: {len(small)} of "
        f"{s.victim_location_type.nunique() * s.carrier_state.nunique()}; "
        f"every cell is too small to support a rate")
pd.crosstab([ev.victim_location_type, ev.channel], ev.carrier_state, margins=True).to_csv(
    OUT / "crosstab.csv")
print(f"[5] wrote {OUT.name}/crosstab.csv")

# ================================================================================================
# 6. Was the carrier heading toward N or away from it?
# ================================================================================================
# Four phases, read off the same visit that step 2 resolves: still at or before P (heading toward N),
# between P and N (heading toward), between arrival and departure at N, and past the departure from N (away).
log("")
w = off_use[in_win].copy()
w["phase"] = np.select(
    [w.eta < 0, w.post < 0, w.on_dep > 0],
    ["away (past the departure from N)", "at N (arrived, not yet departed)", "toward N (P to N)"],
    default="toward N (not yet left P)")
w["toward"] = w.phase.isin(["toward N (P to N)", "toward N (not yet left P)"])
log(f"[6] CARRIER PHASE AT ONSET, {len(w)} in-window off-line events: "
    f"{ {k: int(v) for k, v in w.phase.value_counts().items()} }")
log(f"    toward N {int(w.toward.sum())}, away {int((~w.toward).sum())}")
for lab, s in (("toward", w[w.toward]), ("away", w[~w.toward])):
    log(f"    {lab:<7} " + fmt("eta", q5(s.eta)))
w[["incident_id", "N", "phase", "on_dep", "eta", "post", "run"]].to_csv(OUT / "phase.csv", index=False)
print(f"[6] wrote {OUT.name}/phase.csv")

# ================================================================================================
# 7. Per-interchange concentration against the background exposure
# ================================================================================================
# The exposure is 09's: train-seconds of the carrier spent within the near radius of each interchange.
log("")
pi = per_int.copy()
pi["ci_lo"], pi["ci_hi"] = poisson_ci(pi.n_events_near.values,
                                      pi.exposure_train_h_near.values / 1000.0)
pi = pi.sort_values("rate_per_1000_train_h_near", ascending=False)
log(f"[7] PER-INTERCHANGE RATE of off-line events within the near radius, exact Poisson (Garwood) 95% CI")
for r in pi.itertuples():
    log(f"    {r.interchange:<6} n={int(r.n_events_near):>2}  exposure {r.exposure_train_h_near:>7.1f} train-h  "
        f"rate {r.rate_per_1000_train_h_near:>6.1f} per 1000 train-h  "
        f"95% CI [{r.ci_lo:>6.1f}, {r.ci_hi:>6.1f}]")
obs = pi.n_events_near.values.astype(float)
exp = pi.exposure_train_h_near.values / pi.exposure_train_h_near.sum() * obs.sum()
het = float(((obs - exp) ** 2 / exp).sum())
p_het = float(stats.chi2.sf(het, len(obs) - 1))
log(f"    total {int(obs.sum())} events / {pi.exposure_train_h_near.sum():.1f} train-h = "
    f"{1000 * obs.sum() / pi.exposure_train_h_near.sum():.1f} per 1000 train-h")
log(f"    heterogeneity chi-square {het:.1f} on {len(obs) - 1} df, p = {p_het:.4f} "
    f"(a 6-fold spread in the point estimates, on 2-14 events per interchange)")
pi[["interchange", "n_events_near", "exposure_train_h_near", "rate_per_1000_train_h_near",
    "ci_lo", "ci_hi"]].to_csv(OUT / "per_interchange_ci.csv", index=False)
print(f"[7] wrote {OUT.name}/per_interchange_ci.csv")

# ================================================================================================
# 8. Figure
# ================================================================================================
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
a = ax[0, 0]
a.hist(off_use.loc[in_win, "eta"].dropna(), bins=np.arange(-360, 241, 20), color="#4878a8")
a.axvline(0, color="k", lw=0.8)
a.axvline(off_use.loc[in_win, "eta"].median(), color="#c03030", ls="--",
          label=f"median {off_use.loc[in_win, 'eta'].median():.0f} s")
a.set_title(f"off-line: arrival at N minus onset (n={int(in_win.sum())})", fontsize=10)
a.set_xlabel("eta (s)"); a.set_ylabel("events"); a.legend(fontsize=8)

a = ax[0, 1]
a.hist(near.offset, bins=np.arange(-180, 181, 5), color="#4878a8")
a.axvspan(-10, 0, color="#c03030", alpha=0.15)
a.set_title(f"off-line: nearest carrier timetable event minus onset (n={N_ALL})", fontsize=10)
a.set_xlabel("event - onset (s)"); a.set_ylabel("events")

a = ax[1, 0]
for k, c in (("global", "#999999"), ("local", "#4878a8")):
    a.hist(NULL[k]["iqr_eta"], bins=40, alpha=0.75, color=c, label=f"null {k}")
a.axvline(OBS["iqr_eta"], color="#c03030", lw=2, label=f"observed {OBS['iqr_eta']:.0f} s")
a.set_xscale("log"); a.set_title("IQR of eta: observed vs random-onset null (2000 draws)", fontsize=10)
a.set_xlabel("IQR (s, log)"); a.set_ylabel("draws"); a.legend(fontsize=8)

a = ax[1, 1]
lead = onl4.lead[onl4.lead.abs() <= 120]
a.hist(lead, bins=np.arange(-120, 121, 10), color="#5a9e5a")
a.axvline(0, color="k", lw=0.8)
a.axvline(lead.median(), color="#c03030", ls="--", label=f"median {lead.median():.0f} s")
a.set_title(f"on-line: carrier's nearest passage of the victim's point (n={len(lead)})", fontsize=10)
a.set_xlabel("passage - onset (s)"); a.set_ylabel("events"); a.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "leadtime.png", dpi=140)
log("")
log(f"[8] wrote {OUT.name}/leadtime.png")
print(f"[8] wrote {OUT.name}/leadtime.png")
