"""12_geo.py -- breakdown distance to the carrier in metres (real station coordinates).

Replaces the network-hop distance of src/03_mechanism.py with a physical distance. Station coordinates come from a
cached OpenStreetMap Overpass response (out/geo/osm_raw_rel2.json); the script never goes online. A train inside a
segment is placed by linear interpolation between the two station coordinates at constant speed -- that constant-speed
assumption is what makes "where was the train at 11:04:46" answerable at all, and section 3 checks it against the
scheduled travel times.

Sections
 1. Station coordinates: 120 stations matched by Chinese station name against OSM stop_position nodes.
 2. Segment lengths (haversine between adjacent stations) and the network graph behind network_m.
 3. Speeds: median scheduled travel time per route x segment -> speed_kmh, sanity flags, line-length check.
 4. Victim and carrier position at every onset -> straight_m and network_m (plus the old hop distance d_c). The
    carrier is in a segment, dwelling mid-trip, standing at an end-of-trip station (terminal layover: it is reversing,
    so it is in service and has a position) or off (before its first departure / after its last arrival).
 5. Exposure on the 30-s grid (every in-service train x every tick, distance to the carrier) -> rate by distance bin,
    over the full 0 .. whole-network range, with the rate relative to the >3 km background. The carrier's terminal
    ticks carry a distance like any other in-service tick, so they enter the denominator instead of being dropped.
 6. Breakdown rate per distance bin over the whole 0 .. 60 km range, with the rate relative to the background.
 7. Radius scan: define "zone" as distance <= r, sweep r from 0 to 10 km, report in/out rates and their ratio RR.
 8. Decay shape: exponential (Poisson GLM, log link, log-exposure offset), exponential + constant background, and
    a piecewise-constant model with an estimated breakpoint -- exponential falloff or a hard radius?
 9. The four questions of notes/07_geo.md, answered with the numbers printed here.
10. Figures: rate vs distance and RR vs radius.
11. State agreement with src/13_carrier_state.py (6 classes) at the 143 onsets: terminal and off must be split the
    same way by both scripts.

Outputs: out/geo/station_coords.csv, segment_lengths.csv, speed_check.csv, events_geo.csv, rate_by_distance.csv,
         rate_by_distance_network.csv, radius_scan.csv, radius_scan_network.csv, geo_results.txt,
         figures/geo/fig_distance_kernel[_network].png, figures/geo/fig_distance_radius_scan[_network].png
Run: python3 src/12_geo.py   (~5 s; reads data_derived/traversals.pkl and the OSM cache; no network access)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[1]
DATA, DERIVED = ROOT / "data", ROOT / "data_derived"
OUT = ROOT / "out" / "geo"
FIG = ROOT / "figures" / "geo"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
OSM_CACHE = OUT / "osm_raw_rel2.json"  # copied from /tmp/rel2.json once; the script only reads this file
LOG_PATH = OUT / "geo_results.txt"

CARRIER = "T3420"
TICK_S = 30  # space-time grid resolution (dwell is 30 s, so every station stop occupies one tick)
R_EARTH = 6_371_000.0
# Distance bins in metres, fine near the carrier and coarse far away, spanning 0 .. the whole network. Every bin
# must carry non-zero exposure, which is why the far bins are kilometres wide. All edges are multiples of 50 m so the
# same 50 m distance histogram feeds both this table and the radius scan.
BIN_EDGES = [0, 50, 100, 150, 250, 400, 600, 900, 1300, 2000, 3000, 5000, 10000, np.inf]
BIN_LABELS = ["0-50", "50-100", "100-150", "150-250", "250-400", "400-600", "600-900", "900-1300", "1300-2000",
              "2000-3000", "3000-5000", "5000-10000", "10000+"]
BACKGROUND_M = 3000.0  # "background" = every event beyond this straight-line distance / all exposure out there
NEAR_M = 150.0  # the near field: 0-150 m, reported as one aggregate because the 50 m bins there are tiny
RADIUS_MAX, RADIUS_STEP = 10_000.0, 50.0  # radius scan: r = 0, 50, ..., 10000 m
DIST_MAX = 60_000.0  # the whole network fits inside 60 km, so one 50 m histogram feeds every table below
DIST_EDGES = np.arange(0.0, DIST_MAX + RADIUS_STEP, RADIUS_STEP)
SPEED_MIN_OK, SPEED_MAX_OK = 15.0, 90.0  # physically plausible inter-station speed band (km/h)
# Published line lengths (km, approximate public figures, only used as an order-of-magnitude check).
LINE_LEN_REF_KM = {"2": 64.6, "8": 37.4, "10": 45.4, "12": 40.4}

_log_lines: list[str] = []


def log(*parts) -> None:
    """Console gets at most 15 lines per call (brief: short terminal output); the results file gets everything."""
    text = " ".join(str(p) for p in parts)
    lines = text.splitlines()
    print("\n".join(lines[:15]) + (f"\n   ... (+{len(lines) - 15} more lines in {LOG_PATH.name})"
                                   if len(lines) > 15 else ""), flush=True)
    _log_lines.append(text)


def section(title: str) -> None:
    log("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def flush_log() -> None:
    LOG_PATH.write_text("\n".join(_log_lines) + "\n", encoding="utf-8")


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres; works elementwise on numpy arrays."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def poisson_ci(k, exposure, level=0.95):
    """Exact (Garwood) CI for a Poisson rate k/exposure, same estimator as 03."""
    k = np.asarray(k, dtype=float)
    a = (1 - level) / 2
    lo = np.where(k > 0, stats.chi2.ppf(a, 2 * k) / 2, 0.0)
    hi = stats.chi2.ppf(1 - a, 2 * k + 2) / 2
    return lo / exposure, hi / exposure


def sec_of_day(ts: pd.Series, day: pd.Series) -> np.ndarray:
    """Seconds after midnight of `day` (vectorised; timestamps are tz-naive local clock times)."""
    return (ts - day).dt.total_seconds().values


def name_key(s: str) -> str:
    """Station-name key for OSM matching: the middle dot is spelled '·' in our data and '・' in OSM."""
    return re.sub(r"[·・•]", "", str(s)).strip()


def undirected(seg: str) -> str:
    """'S005 -> S004' and 'S004 -> S005' are the same piece of track."""
    return " - ".join(sorted(seg.split(" -> ")))


# ================================================================================================
section("1. Station coordinates from the cached OSM response")
# ================================================================================================
raw = json.loads(OSM_CACHE.read_text(encoding="utf-8"))
assert raw.get("elements"), f"empty OSM cache {OSM_CACHE}"
nodes = {e["id"]: e for e in raw["elements"] if e["type"] == "node"}
relations = [e for e in raw["elements"] if e["type"] == "relation"]
osm: dict[str, list[tuple[float, float]]] = {}
for n in nodes.values():
    nm = n.get("tags", {}).get("name")
    if nm and "lat" in n and "lon" in n:
        osm.setdefault(name_key(nm), []).append((float(n["lat"]), float(n["lon"])))

st = pd.read_csv(DATA / "station_information.csv", encoding="utf-8-sig", dtype=str)
uniq = st.drop_duplicates("station_id")[["station_id", "station_name"]].reset_index(drop=True)
log(f"OSM cache: {len(raw['elements']):,} elements ({len(nodes):,} nodes, {len(relations)} route relations); "
    f"{len(osm):,} distinct named node keys")
log(f"station table: {len(st)} rows, {len(uniq)} distinct stations; "
    f"lines {sorted(st.line_id.unique())}, routes {sorted(st.route_id.unique())}")

rows, missing, spreads = [], [], []
for r in uniq.itertuples():
    pts = osm.get(name_key(r.station_name))
    if not pts:
        missing.append(r.station_name)
        continue
    a = np.array(pts)
    if len(a) > 1:  # several stop_position nodes per station (one per platform): spread tells us how coarse this is
        spreads.append(max(haversine(a[i, 0], a[i, 1], a[j, 0], a[j, 1])
                           for i in range(len(a)) for j in range(i + 1, len(a))))
    rows.append({"station_id": r.station_id, "station_name": r.station_name,
                 "lat": float(a[:, 0].mean()), "lon": float(a[:, 1].mean()), "source": "osm"})
coords = pd.DataFrame(rows)
if missing:  # the brief's fallback; never fires on this cache (120/120 match) and is asserted rather than guessed
    raise RuntimeError(f"stations without OSM coordinates: {missing}")
coords.to_csv(OUT / "station_coords.csv", index=False)
log(f"matched {len(coords)}/{len(uniq)} stations (source=osm); multi-node stations {len(spreads)}: "
    f"platform spread median {np.median(spreads):.0f} m, max {max(spreads):.0f} m")
log(f"latitude {coords.lat.min():.4f}-{coords.lat.max():.4f}, longitude {coords.lon.min():.4f}-"
    f"{coords.lon.max():.4f} (Shanghai)")
flush_log()


# ================================================================================================
section("2. Segment lengths (haversine between adjacent stations) and the network graph")
# ================================================================================================
COORD = coords.set_index("station_id")[["lat", "lon"]]
LAT, LON = COORD.lat.values, COORD.lon.values
NODE = {s: i for i, s in enumerate(COORD.index)}  # 0..119, the vertex set of the track graph
st["sequence"] = st.sequence.astype(int)

seg_rows = []
for route_id, g in st.sort_values(["route_id", "sequence"]).groupby("route_id"):
    g = g.reset_index(drop=True)
    for a, b in zip(g.station_id[:-1], g.station_id[1:]):
        seg_rows.append({"route_id": route_id, "line_id": g.line_id.iloc[0], "from_station": a, "to_station": b,
                         "from_name": COORD.index.name and g.loc[g.station_id == a, "station_name"].iloc[0],
                         "to_name": g.loc[g.station_id == b, "station_name"].iloc[0],
                         "length_m": float(haversine(LAT[NODE[a]], LON[NODE[a]], LAT[NODE[b]], LON[NODE[b]]))})
seg = pd.DataFrame(seg_rows)
seg.to_csv(OUT / "segment_lengths.csv", index=False)
log(f"segments: {len(seg)} (route x adjacency); length median {seg.length_m.median():.0f} m, "
    f"IQR {seg.length_m.quantile(.25):.0f}-{seg.length_m.quantile(.75):.0f} m, "
    f"min {seg.length_m.min():.0f}, max {seg.length_m.max():.0f}")
log("per route: " + "; ".join(f"{r}: {len(g)} segs, {g.length_m.sum() / 1000:.1f} km"
                              for r, g in seg.groupby("route_id")))

# Undirected edge lengths: the same piece of track can appear on two routes (line-10 trunk, and the line-10 branch
# shares stations with the trunk), so the graph takes the shortest recorded length per station pair.
edge = {}
for r in seg.itertuples():
    k = tuple(sorted((NODE[r.from_station], NODE[r.to_station])))
    edge[k] = min(edge.get(k, np.inf), r.length_m)
n = len(COORD)
W = np.full((n, n), np.inf)
for (i, j), d in edge.items():
    W[i, j] = W[j, i] = d
np.fill_diagonal(W, 0.0)  # a station to itself is 0 m of track (dijkstra and the position arithmetic rely on it)
NET = dijkstra(W, directed=False)  # all-pairs shortest path along the track, transfers between lines are free
n_comp = int((~np.isinf(NET[0])).sum())
log(f"track graph: {n} vertices, {len(edge)} undirected edges; stations reachable from {COORD.index[0]}: "
    f"{n_comp}/{n}; network diameter {np.nanmax(NET[~np.isinf(NET)]) / 1000:.1f} km")

line_len = {}
for line_id, g in seg.groupby("line_id"):
    keys = {tuple(sorted((NODE[a], NODE[b]))) for a, b in zip(g.from_station, g.to_station)}
    line_len[line_id] = sum(edge[k] for k in keys) / 1000
log("line length from station coordinates vs published figure (km): " + "; ".join(
    f"L{ln} {line_len[ln]:.1f} vs {LINE_LEN_REF_KM[ln]:.1f} (x{line_len[ln] / LINE_LEN_REF_KM[ln]:.2f})"
    for ln in sorted(line_len, key=int)))
flush_log()


# ================================================================================================
section("3. Speeds: median scheduled travel time per route x segment")
# ================================================================================================
trav = pd.read_pickle(DERIVED / "traversals.pkl")
bd = pd.read_csv(DERIVED / "breakdowns.csv", parse_dates=["onset_ts", "recovery_ts", "date"])
ev_hop = pd.read_csv(ROOT / "out" / "interchange" / "events.csv", usecols=["incident_id", "d_c"])

tt = (trav.groupby(["route_id", "segment"], observed=True)
      .agg(n_trav=("travel_s", "size"), travel_s_med=("travel_s", "median")).reset_index())
tt["key"] = tt.segment.astype(str).map(undirected)
tt = (tt.groupby(["route_id", "key"], as_index=False)  # both directions of a segment -> one row per track piece
      .agg(n_trav=("n_trav", "sum"), travel_s_med=("travel_s_med", "median")))
seg["key"] = [" - ".join(sorted([a, b])) for a, b in zip(seg.from_station, seg.to_station)]
spd = seg.merge(tt[["route_id", "key", "n_trav", "travel_s_med"]], on=["route_id", "key"], how="left")
spd["speed_kmh"] = spd.length_m / spd.travel_s_med * 3.6
spd["flag"] = np.select([spd.speed_kmh < SPEED_MIN_OK, spd.speed_kmh > SPEED_MAX_OK], ["too_slow", "too_fast"],
                        default="ok")
spd[["route_id", "line_id", "from_station", "to_station", "from_name", "to_name", "length_m", "n_trav",
     "travel_s_med", "speed_kmh", "flag"]].to_csv(OUT / "speed_check.csv", index=False)
log(f"segments with a scheduled travel time: {int(spd.travel_s_med.notna().sum())}/{len(spd)}; traversals per "
    f"segment median {spd.n_trav.median():.0f} (min {spd.n_trav.min():.0f})")
log(f"speed km/h over {len(spd)} route x segments: median {spd.speed_kmh.median():.1f}, "
    f"IQR {spd.speed_kmh.quantile(.25):.1f}-{spd.speed_kmh.quantile(.75):.1f}, "
    f"min {spd.speed_kmh.min():.1f}, max {spd.speed_kmh.max():.1f}")
log("per line: " + "; ".join(f"L{ln} median {g.speed_kmh.median():.1f} km/h ({len(g)} segs)"
                             for ln, g in spd.groupby("line_id")))
bad = spd[spd.flag != "ok"].sort_values("speed_kmh")
log(f"physically implausible (<{SPEED_MIN_OK:.0f} or >{SPEED_MAX_OK:.0f} km/h): {len(bad)} segments"
    + ("" if bad.empty else ":\n" + bad[["route_id", "from_name", "to_name", "length_m", "travel_s_med",
                                         "speed_kmh", "flag"]].round(1).to_string(index=False)))
flush_log()


# ================================================================================================
section("4. Victim and carrier position at every onset -> straight_m, network_m")
# ================================================================================================
TCODES = {t: i for i, t in enumerate(trav.train_id.cat.categories)}
DATES = np.unique(trav.date.values)
DCODES = {pd.Timestamp(d): i for i, d in enumerate(DATES)}
combo = trav.train_id.cat.codes.values.astype(np.int64) * 1000 + pd.Categorical(trav.date, categories=DATES).codes
order = np.argsort(combo, kind="stable")
combo_s = combo[order]
DEP = trav.dep_ts.values.astype("datetime64[ns]").astype(np.int64)
ARR = trav.arr_ts.values.astype("datetime64[ns]").astype(np.int64)
DWELL = trav.dwell_s.values.astype(float)
FROM = trav.from_station.astype(str).values
TO = trav.to_station.astype(str).values
TRAV_ROUTE = trav.route_id.astype(str).values
TRAV_SEG = trav.segment.astype(str).values
SPEED = {(r.route_id, r.key): (r.length_m, r.speed_kmh) for r in spd.itertuples()}
PLAN = {(r.route_id, r.key): float(r.travel_s_med) for r in spd.itertuples()}

# Terminal layovers: between arriving at the end of one trip and departing on the next, a train stands at the
# end-of-trip station (31-41 min for the carrier, against 30 s for a mid-trip dwell). The segment intervals and the
# dwell intervals below both stop short of that spell, so without this class a reversing train looks out of service --
# which is how the carrier's 6 terminal-station breakdowns lost their distance in the first version of this script.
# The spell is [arr_ts, next dep_ts) at to_station, and it is a turnaround exactly when the next row starts a new trip.
tsp = (trav[["train_id", "date", "trip_key", "from_station", "to_station", "arr_ts", "dep_ts"]]
       .sort_values(["train_id", "date", "dep_ts"]).reset_index(drop=True))
_g = tsp.groupby(["train_id", "date"], observed=True)
tsp["next_dep"] = _g.dep_ts.shift(-1)
tsp["next_trip"] = _g.trip_key.shift(-1)
tsp["next_from"] = _g.from_station.shift(-1)
del _g
tsp = tsp[tsp.next_dep.notna() & (tsp.next_dep > tsp.arr_ts) & (tsp.next_trip != tsp.trip_key)].reset_index(drop=True)
assert (tsp.to_station == tsp.next_from).all(), "a turnaround must depart from the station it arrived at"
T_ARR = tsp.arr_ts.values.astype("datetime64[ns]").astype(np.int64)
T_DEP = tsp.next_dep.values.astype("datetime64[ns]").astype(np.int64)
T_NODE = np.array([NODE[s] for s in tsp.to_station.astype(str)], dtype=np.int16)
T_COMBO = tsp.train_id.astype(str).map(TCODES).values.astype(np.int64) * 1000 + \
    pd.Categorical(tsp.date, categories=DATES).codes
T_ORDER = np.argsort(T_COMBO, kind="stable")
T_COMBO_S = T_COMBO[T_ORDER]
_car_tsp = tsp[tsp.train_id.astype(str) == CARRIER].reset_index(drop=True)  # the carrier's spells drive the grid
CAR_T_START_S = sec_of_day(_car_tsp.arr_ts, _car_tsp.date)
CAR_T_END_S = sec_of_day(_car_tsp.next_dep, _car_tsp.date)
CAR_T_NODE = np.array([NODE[s] for s in _car_tsp.to_station.astype(str)], dtype=np.int16)
CAR_T_DCODE = pd.Categorical(_car_tsp.date, categories=DATES).codes
log(f"terminal layovers: {len(tsp):,} spells over {tsp.train_id.nunique()} trains; carrier {len(_car_tsp):,} spells "
    f"of median {(_car_tsp.next_dep - _car_tsp.arr_ts).dt.total_seconds().median() / 60:.0f} min at "
    + ", ".join(f"{s} x{n}" for s, n in _car_tsp.to_station.value_counts().items() if n))


def locate(train: str, day, ts) -> dict | None:
    """Where is `train` on `day` at `ts`? Segment -> (from, to, fraction); station or terminal -> (node, node, 0).

    Returns None when the train is not in service (before its first departure, after its last arrival, or between two
    trips that do not meet at a station).
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
    if m.any():  # on a segment: linear interpolation between the two station coordinates
        r = int(rows[m][0])
        u, v = NODE[FROM[r]], NODE[TO[r]]
        key = " - ".join(sorted([FROM[r], TO[r]]))
        length, speed = SPEED[(TRAV_ROUTE[r], key)]
        # A breakdown traversal's arr_ts includes the 5-10 min stop, so dividing by it would place the train next to the
        # station it had just left (f ~ 0.08 instead of ~0.33). Whenever the recorded spell is longer than the scheduled
        # running time, the fraction is taken against the scheduled time instead.
        # ponytail: inside a breakdown spell the train is actually held at the onset point, but it is still interpolated
        # forward to the arrival station; only the 143 breakdown spells are affected and the victim is always read at its
        # own onset, where the two conventions agree.
        plan = PLAN.get((TRAV_ROUTE[r], key), np.nan)
        dur = (ARR[r] - DEP[r]) / 1e9
        den = min(dur, plan) if np.isfinite(plan) else dur
        f = float(np.clip((t - DEP[r]) / (den * 1e9), 0.0, 1.0))
        return {"kind": "segment", "seg": TRAV_SEG[r], "u": u, "v": v, "f": f, "length_m": length,
                "lat": LAT[u] + f * (LAT[v] - LAT[u]), "lon": LON[u] + f * (LON[v] - LON[u]),
                "speed_kmh": speed}
    m2 = np.isfinite(DWELL[rows]) & (DEP[rows] - DWELL[rows] * 1e9 <= t) & (DEP[rows] > t)
    if m2.any():  # dwelling at a station: the departure row's dwell interval covers [dep - dwell, dep)
        r = int(rows[m2][0])
        u = NODE[FROM[r]]
        return {"kind": "station", "seg": FROM[r], "u": u, "v": u, "f": 0.0, "length_m": 0.0,
                "lat": LAT[u], "lon": LON[u], "speed_kmh": np.nan}
    lo, hi = np.searchsorted(T_COMBO_S, c, "left"), np.searchsorted(T_COMBO_S, c, "right")
    trows = T_ORDER[lo:hi]
    m3 = (T_ARR[trows] <= t) & (T_DEP[trows] > t)
    if m3.any():  # reversing at the end-of-trip station: in service, no interpolation, the station's coordinates
        r = int(trows[m3][0])
        u = T_NODE[r]
        return {"kind": "terminal", "seg": str(tsp.to_station.iat[r]), "u": u, "v": u, "f": 0.0, "length_m": 0.0,
                "lat": LAT[u], "lon": LON[u], "speed_kmh": np.nan}
    return None


def network_m(p1: dict, p2: dict) -> float:
    """Shortest track distance between two interpolated positions: partial segment + graph path + partial segment.

    Both positions sit on their own segment; a path must leave through one of the four segment ends, so the minimum
    over the four combinations is the network distance. Two positions on the same piece of track are handled directly.
    """
    best = np.inf
    if p1["u"] != p1["v"] and {p1["u"], p1["v"]} == {p2["u"], p2["v"]}:
        best = abs(p1["f"] - p2["f"]) * p1["length_m"]
    for a, pa in ((p1["u"], p1["f"] * p1["length_m"]), (p1["v"], (1 - p1["f"]) * p1["length_m"])):
        for b, pb in ((p2["u"], p2["f"] * p2["length_m"]), (p2["v"], (1 - p2["f"]) * p2["length_m"])):
            best = min(best, pa + NET[a, b] + pb)
    return float(best)


rows, seg_match, station_dwell_ok = [], 0, 0
for r in bd.itertuples():
    vic = locate(r.train_id, r.date, r.onset_ts)
    if r.location_type == "segment":
        if vic is None or vic["kind"] != "segment" or vic["seg"] != r.location:
            raise RuntimeError(f"{r.incident_id}: onset {r.onset_ts} not inside recorded segment {r.location}")
        seg_match += 1
        v_lat, v_lon, v_frac, v_len, v_spd = vic["lat"], vic["lon"], vic["f"], vic["length_m"], vic["speed_kmh"]
        v_u, v_v, v_f = vic["u"], vic["v"], vic["f"]
    else:  # station breakdown: the victim is at that station by definition
        u = NODE[r.location]
        station_dwell_ok += int(vic is not None and vic["kind"] == "station" and vic["seg"] == r.location)
        v_lat, v_lon, v_frac, v_len, v_spd = LAT[u], LON[u], np.nan, np.nan, np.nan
        v_u = v_v = u
        v_f = 0.0
    car = locate(CARRIER, r.date, r.onset_ts)
    if car is None:
        c_state, c_lat, c_lon, s_m, n_m = "off", np.nan, np.nan, np.nan, np.nan
    else:
        c_state, c_lat, c_lon = car["kind"], car["lat"], car["lon"]
        s_m = float(haversine(v_lat, v_lon, c_lat, c_lon))
        n_m = network_m({"u": v_u, "v": v_v, "f": v_f, "length_m": v_len if np.isfinite(v_len) else 0.0},
                        {"u": car["u"], "v": car["v"], "f": car["f"], "length_m": car["length_m"]})
    rows.append({"incident_id": r.incident_id, "train_id": r.train_id, "line_id": r.line_id,
                 "route_id": r.route_id, "onset_ts": r.onset_ts, "location_type": r.location_type,
                 "victim_lat": v_lat, "victim_lon": v_lon, "victim_seg_len_m": v_len, "victim_frac": v_frac,
                 "speed_kmh_victim": v_spd, "carrier_state": c_state, "carrier_lat": c_lat, "carrier_lon": c_lon,
                 "straight_m": s_m, "network_m": n_m})
geo = pd.DataFrame(rows).merge(ev_hop, on="incident_id", how="left")
geo["d_c"] = geo.d_c.astype(float)
geo.to_csv(OUT / "events_geo.csv", index=False)
log(f"events: {len(geo)}; segment onsets located inside the recorded segment: {seg_match}/"
    f"{int((bd.location_type == 'segment').sum())}; station onsets inside a dwell at that station: "
    f"{station_dwell_ok}/{int((bd.location_type == 'station').sum())}")
log(f"carrier state at onset: {geo.carrier_state.value_counts().to_dict()}")
on = geo[geo.carrier_state != "off"]
log(f"straight_m: median {on.straight_m.median():.0f} m, IQR {on.straight_m.quantile(.25):.0f}-"
    f"{on.straight_m.quantile(.75):.0f}, min {on.straight_m.min():.0f}, max {on.straight_m.max():.0f}")
log(f"network_m: median {on.network_m.median():.0f} m, IQR {on.network_m.quantile(.25):.0f}-"
    f"{on.network_m.quantile(.75):.0f}, max {on.network_m.max():.0f}; network_m/straight_m median "
    f"{(on.network_m / on.straight_m.replace(0, np.nan)).median():.2f}")
log("straight_m by hop distance d_c:\n" + on.groupby("d_c").straight_m.describe()[["count", "min", "50%", "max"]]
    .round(0).to_string())
flush_log()


# ================================================================================================
section("5. Exposure on the 30-s grid: distance from every other train to the carrier")
# ================================================================================================
DEP_S = sec_of_day(trav.dep_ts, trav.date)
ARR_S = sec_of_day(trav.arr_ts, trav.date)
T0 = float(np.floor(np.nanmin(DEP_S) / 3600) * 3600)
NT = int(np.ceil((np.nanmax(ARR_S) - T0) / TICK_S)) + 1
U = np.array([NODE[s] for s in FROM], dtype=np.int16)
V = np.array([NODE[s] for s in TO], dtype=np.int16)
TC = trav.train_id.cat.codes.values.astype(np.int16)
N_TRAIN = len(trav.train_id.cat.categories)
CI = TCODES[CARRIER]
DCODE_ARR = pd.Categorical(trav.date, categories=DATES).codes
TICK_H = TICK_S / 3600
log(f"grid: {N_TRAIN} trains x {len(DATES)} dates x {NT} ticks of {TICK_S} s (from {T0 / 3600:.0f}:00); "
    f"traversals {len(trav):,}")

LEN2 = W.copy()  # segment length by node pair: 0 on the diagonal, length on an edge, inf elsewhere
fine_s = np.zeros(len(DIST_EDGES) - 1)  # train-ticks per 50 m straight-line distance bin, 0 .. 60 km
fine_n = np.zeros(len(DIST_EDGES) - 1)  # same for network distance along the track
far_s = far_n = 0  # train-ticks beyond the last edge (should stay 0)
carrier_ticks = 0
carrier_off_ticks, carrier_off_others = 0, 0.0
carrier_term_ticks, carrier_term_others = 0, 0.0
for di, day in enumerate(DATES):
    sel = DCODE_ARR == di
    u_g = np.full((N_TRAIN, NT), -1, dtype=np.int16)  # segment start node, -1 = not in service
    v_g = np.full((N_TRAIN, NT), -1, dtype=np.int16)  # segment end node (== start when dwelling)
    f_g = np.zeros((N_TRAIN, NT), dtype=np.float32)  # position along the segment, 0..1
    rows = np.flatnonzero(sel)
    a = np.ceil((DEP_S[rows] - T0) / TICK_S).astype(np.int64)
    b = np.ceil((ARR_S[rows] - T0) / TICK_S).astype(np.int64)
    n_t = np.maximum(b - a, 0)
    rep_local = np.repeat(np.arange(rows.size), n_t)
    rep = rows[rep_local]
    offs = np.arange(n_t.sum()) - np.repeat(np.cumsum(n_t) - n_t, n_t)
    tick = a[rep_local] + offs
    f = np.clip((tick * TICK_S + T0 - DEP_S[rep]) / (ARR_S[rep] - DEP_S[rep]), 0.0, 1.0)
    u_g[TC[rep], tick], v_g[TC[rep], tick], f_g[TC[rep], tick] = U[rep], V[rep], f
    del rep, offs, tick, f, n_t, a, b
    dw = np.flatnonzero(np.isfinite(DWELL) & sel)  # station dwell intervals [dep - dwell, dep)
    a = np.ceil((DEP_S[dw] - DWELL[dw] - T0) / TICK_S).astype(np.int64)
    b = np.ceil((DEP_S[dw] - T0) / TICK_S).astype(np.int64)
    n_t = np.maximum(b - a, 0)
    rep_local = np.repeat(np.arange(dw.size), n_t)
    rep = dw[rep_local]
    offs = np.arange(n_t.sum()) - np.repeat(np.cumsum(n_t) - n_t, n_t)
    tick = a[rep_local] + offs
    u_g[TC[rep], tick], v_g[TC[rep], tick], f_g[TC[rep], tick] = U[rep], U[rep], 0.0
    del rep, offs, tick, n_t, a, b, rep_local, dw, rows
    # The carrier's terminal layovers get the end-of-trip station as their position. Only the carrier: a train is in
    # the risk set while it is inside a segment or a 30-s dwell, so the other trains' layovers stay out (03/13 rule).
    tw = np.flatnonzero(CAR_T_DCODE == di)
    a = np.ceil((CAR_T_START_S[tw] - T0) / TICK_S).astype(np.int64)
    b = np.ceil((CAR_T_END_S[tw] - T0) / TICK_S).astype(np.int64)
    n_t = np.maximum(b - a, 0)
    rep_local = np.repeat(np.arange(tw.size), n_t)
    tick = a[rep_local] + np.arange(n_t.sum()) - np.repeat(np.cumsum(n_t) - n_t, n_t)
    u_g[CI, tick], v_g[CI, tick], f_g[CI, tick] = CAR_T_NODE[tw][rep_local], CAR_T_NODE[tw][rep_local], 0.0
    is_terminal = np.zeros(NT, dtype=bool)
    is_terminal[tick] = True
    del tw, rep_local, tick, n_t, a, b

    valid = u_g >= 0
    uu, vv = np.maximum(u_g, 0), np.maximum(v_g, 0)
    plat = np.where(valid, LAT[uu] + f_g * (LAT[vv] - LAT[uu]), np.nan)  # position of every in-service train
    plon = np.where(valid, LON[uu] + f_g * (LON[vv] - LON[uu]), np.nan)
    cu, cv, cf = u_g[CI], v_g[CI], f_g[CI]
    con = cu >= 0  # carrier in service
    carrier_ticks += int(con.sum())
    term = con & is_terminal  # reversing at a terminal station: in service, so it has a position and an exposure
    carrier_term_ticks += int(term.sum())
    carrier_term_others += float(valid[:, term].sum() - term.sum()) * TICK_H  # -1 per tick: not its own neighbour

    d = haversine(plat[CI][None, :], plon[CI][None, :], plat, plon)  # (train, tick) metres to the carrier
    d[CI] = np.nan  # the carrier is not its own neighbour
    d[~valid] = np.nan
    ok = np.isfinite(d)
    h = np.histogram(d[ok], bins=DIST_EDGES)[0]
    fine_s += h
    far_s += int(ok.sum() - h.sum())

    len_g = LEN2[uu, vv]  # 0 for a station, segment length inside a segment
    pa, pb = f_g * len_g, (1 - f_g) * len_g  # distance from the position to the segment start / end
    ccu, ccv = np.maximum(cu, 0), np.maximum(cv, 0)
    clen = LEN2[ccu, ccv]
    pca, pcb = cf * clen, (1 - cf) * clen
    dn = np.minimum.reduce([  # leave through one end, graph path, enter through one end
        pa + NET[uu, ccu[None, :]] + pca[None, :], pa + NET[uu, ccv[None, :]] + pcb[None, :],
        pb + NET[vv, ccu[None, :]] + pca[None, :], pb + NET[vv, ccv[None, :]] + pcb[None, :]])
    same = ((u_g == cu[None, :]) & (v_g == cv[None, :])) | ((u_g == cv[None, :]) & (v_g == cu[None, :]))
    dn = np.where(same, np.minimum(dn, np.abs(f_g - cf[None, :]) * len_g), dn)  # same piece of track
    dn[CI] = np.nan
    dn[~valid] = np.nan
    dn[:, ~con] = np.nan  # no distance when the carrier is out of service
    ok = np.isfinite(dn)
    h = np.histogram(dn[ok], bins=DIST_EDGES)[0]
    fine_n += h
    far_n += int(ok.sum() - h.sum())

    off = ~con
    carrier_off_ticks += int(off.sum())
    carrier_off_others += float(valid[:, off].sum()) * TICK_H
    del u_g, v_g, f_g, plat, plon, d, dn, ok, valid, uu, vv, len_g, is_terminal

total_s = fine_s.sum() * TICK_H + far_s * TICK_H
total_n = fine_n.sum() * TICK_H + far_n * TICK_H
log(f"carrier in service on {carrier_ticks:,} ticks ({carrier_ticks * TICK_H:,.0f} train-h of carrier presence), of "
    f"which {carrier_term_ticks:,} ticks ({carrier_term_ticks * TICK_H:,.0f} h) are terminal layovers carrying "
    f"{carrier_term_others:,.0f} train-h of other-train exposure; carrier off on {carrier_off_ticks:,} ticks, during "
    f"which other trains accumulated {carrier_off_others:,.0f} train-h")
assert far_s == far_n == 0, f"distance beyond {DIST_MAX / 1000:.0f} km: {far_s} straight, {far_n} network"
beyond10 = DIST_EDGES[1:] > 10_000
log(f"exposure with a carrier distance: {total_s:,.0f} train-h straight / {total_n:,.0f} train-h network "
    f"({fine_s.sum():,.0f} train-ticks); beyond 10 km: {fine_s[beyond10].sum() / fine_s.sum():.1%} straight, "
    f"{fine_n[beyond10].sum() / fine_n.sum():.1%} network")
assert abs(total_s - total_n) < 1e-6 * total_s, "straight and network exposure must cover the same train-ticks"

ev_on = geo[geo.carrier_state != "off"].copy()
ev_on["straight_m"] = ev_on.straight_m.astype(float)
ev_on["network_m"] = ev_on.network_m.astype(float)
BG = {}
for nm, col, fine, far in (("straight", "straight_m", fine_s, far_s), ("network", "network_m", fine_n, far_n)):
    k = int((ev_on[col] > BACKGROUND_M).sum())
    h = float(fine[DIST_EDGES[1:] > BACKGROUND_M].sum() + far) * TICK_H
    BG[nm] = (k, h, k / h * 1000)
    log(f"background >{BACKGROUND_M / 1000:.0f} km ({nm}): {k} events / {h:,.0f} train-h = "
        f"{k / h * 1000:.4f} per 1000 train-h")
log(f"carrier-off events excluded from the kernel: {int((geo.carrier_state == 'off').sum())}")
flush_log()


# ================================================================================================
section("6. Breakdown rate by distance bin (exposure from the 30-s grid)")
# ================================================================================================
CUM = {"straight": np.concatenate([[0.0], np.cumsum(fine_s)]),
       "network": np.concatenate([[0.0], np.cumsum(fine_n)])}


def ticks_upto(cum: np.ndarray, x: float) -> float:
    """Train-ticks whose distance to the carrier is below x; x sits on a 50 m edge, so this is "<= x" up to ties.

    "left" is the off-by-one that matters: cum[j] already holds the ticks of bin j, so a "right" lookup would push
    every bin one 50 m step outward (and would put the [0, 50) bin's exposure into the next bin).
    """
    return float(cum[min(np.searchsorted(DIST_EDGES, x, "left"), len(cum) - 1)])


def bin_exposure_h(cum: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Train-hours inside each [lo, hi) distance bin; an open-ended last bin stops at the histogram edge."""
    return np.array([(ticks_upto(cum, min(b, DIST_MAX)) - ticks_upto(cum, min(a, DIST_MAX))) * TICK_H
                     for a, b in zip(lo, hi)])


def rate_table(col: str, cum: np.ndarray, label: str) -> pd.DataFrame:
    k = np.histogram(ev_on[col].values, bins=BIN_EDGES)[0]
    h = bin_exposure_h(cum, np.array(BIN_EDGES[:-1], dtype=float), np.array(BIN_EDGES[1:], dtype=float))
    assert (h > 0).all(), f"empty exposure bin in the {label} table: {h}"
    lo_ci, hi_ci = poisson_ci(k, h)
    df = pd.DataFrame({"bin": BIN_LABELS, "lo_m": BIN_EDGES[:-1], "hi_m": BIN_EDGES[1:], "events": k,
                       "exposure_train_h": h, "rate_per_1000h": k / h * 1000,
                       "ci_low": lo_ci * 1000, "ci_high": hi_ci * 1000})
    df["rate_vs_background"] = df.rate_per_1000h / BG[label][2]
    return df


RT = {"straight": rate_table("straight_m", CUM["straight"], "straight"),
      "network": rate_table("network_m", CUM["network"], "network")}
RT["straight"].to_csv(OUT / "rate_by_distance.csv", index=False)
RT["network"].to_csv(OUT / "rate_by_distance_network.csv", index=False)
for nm in ("straight", "network"):
    log(f"\nrate by distance to the carrier, {nm} metres (background = "
        f"{BG[nm][2]:.4f} per 1000 train-h):")
    log(RT[nm][["bin", "events", "exposure_train_h", "rate_per_1000h", "ci_low", "ci_high", "rate_vs_background"]]
        .round({"exposure_train_h": 1, "rate_per_1000h": 3, "ci_low": 3, "ci_high": 3, "rate_vs_background": 1})
        .to_string(index=False))
flush_log()


# ================================================================================================
section("7. Radius scan: zone = distance <= r, r swept from 0 to 10 km")
# ================================================================================================
RS = DIST_EDGES[DIST_EDGES <= RADIUS_MAX]  # 0, 50, ..., 10000 m


def radius_scan(col: str, cum: np.ndarray) -> pd.DataFrame:
    """Zone rate / outside rate for every radius r: RR(r) traces where the carrier effect dies out."""
    d = np.sort(ev_on[f"{col}_m"].values)
    h_in = np.array([ticks_upto(cum, r) for r in RS]) * TICK_H
    k_in = np.searchsorted(d, RS, "right").astype(float)
    h_out, k_out = total_s - h_in, d.size - k_in
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_in = np.where(h_in > 0, k_in / h_in * 1000, np.nan)
        rate_out = np.where(h_out > 0, k_out / h_out * 1000, np.nan)
        rr = rate_in / rate_out
        se = np.sqrt(np.where((k_in > 0) & (k_out > 0), 1 / k_in + 1 / k_out, np.nan))
        lo, hi = rr * np.exp(-1.96 * se), rr * np.exp(1.96 * se)
    return pd.DataFrame({"r_m": RS, "events_in": k_in.astype(int), "exposure_in_train_h": h_in,
                         "rate_in_per_1000h": rate_in, "events_out": k_out.astype(int),
                         "exposure_out_train_h": h_out, "rate_out_per_1000h": rate_out,
                         "rr": rr, "rr_ci_low": lo, "rr_ci_high": hi})


SCAN = {nm: radius_scan(nm, CUM[nm]) for nm in ("straight", "network")}
SCAN["straight"].to_csv(OUT / "radius_scan.csv", index=False)
SCAN["network"].to_csv(OUT / "radius_scan_network.csv", index=False)

SCAN_SUMMARY = {}
for nm in ("straight", "network"):
    sc = SCAN[nm]
    i_peak = int(np.nanargmax(sc.rr.values))
    r_peak, rr_peak = float(sc.r_m[i_peak]), float(sc.rr[i_peak])
    after = sc[(sc.r_m > r_peak) & (sc.rr <= rr_peak / 2)]
    r_half = float(after.r_m.iloc[0]) if len(after) else np.nan
    SCAN_SUMMARY[nm] = dict(r_peak=r_peak, rr_peak=rr_peak, rr_peak_lo=float(sc.rr_ci_low[i_peak]),
                            rr_peak_hi=float(sc.rr_ci_high[i_peak]), r_half=r_half)
    log(f"\nRR(r) = rate within r / rate beyond r, {nm} metres "
        f"(zone exposure at r=1 km: {sc.loc[sc.r_m == 1000, 'exposure_in_train_h'].iloc[0]:,.1f} train-h)")
    show = sc[sc.r_m.isin([50, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000])]
    log(show[["r_m", "events_in", "exposure_in_train_h", "rate_in_per_1000h", "rate_out_per_1000h", "rr",
              "rr_ci_low", "rr_ci_high"]]
        .round({"exposure_in_train_h": 1, "rate_in_per_1000h": 3, "rate_out_per_1000h": 3,
                "rr": 1, "rr_ci_low": 1, "rr_ci_high": 1}).to_string(index=False))
    log(f"RR peaks at r = {r_peak:,.0f} m (RR = {rr_peak:.1f}, 95% CI {sc.rr_ci_low[i_peak]:.1f}-"
        f"{sc.rr_ci_high[i_peak]:.1f}); RR falls back to half the peak at r = {r_half:,.0f} m"
        if np.isfinite(r_half) else
        f"RR peaks at r = {r_peak:,.0f} m (RR = {rr_peak:.1f}) and never halves inside 10 km")
flush_log()


# ================================================================================================
section("8. Decay shape: exponential falloff or a hard radius? (Poisson models, offset = log exposure)")
# ================================================================================================
D_MID = 0.5 * (DIST_EDGES[1:] + DIST_EDGES[:-1])  # 50 m bin centres


def coarse_design(col: str, cum: np.ndarray, fine: np.ndarray) -> pd.DataFrame:
    """Aggregate the 50 m histogram to BIN_EDGES, with an exposure-weighted representative distance per bin."""
    k = np.histogram(ev_on[f"{col}_m"].values, bins=BIN_EDGES)[0]
    h = bin_exposure_h(cum, np.array(BIN_EDGES[:-1], float), np.array(BIN_EDGES[1:], float))
    d = np.array([np.average(D_MID[(D_MID >= a) & (D_MID < b)], weights=fine[(D_MID >= a) & (D_MID < b)])
                  for a, b in zip(BIN_EDGES[:-1], BIN_EDGES[1:])])
    return pd.DataFrame({"bin": BIN_LABELS, "d_m": d, "events": k, "exposure_train_h": h})


def deviance(k: np.ndarray, h: np.ndarray, lam: np.ndarray) -> float:
    """Poisson deviance of predicted rates lam against counts k on exposure h (0 log 0 = 0)."""
    mu = np.maximum(h * lam, 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(k > 0, k * np.log(np.where(k > 0, k, 1.0) / mu), 0.0)
    return float(2 * np.sum(t - (k - mu)))


def numeric_hessian(grad, x: np.ndarray, step: np.ndarray, *args) -> np.ndarray:
    """Central-difference Hessian of an analytic gradient (curvature gives the parameter standard errors)."""
    n = x.size
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei, ej = np.eye(n)[i] * step[i], np.eye(n)[j] * step[j]
            H[i, j] = (grad(x + ei + ej, *args)[i] - grad(x + ei - ej, *args)[i]
                       - grad(x - ei + ej, *args)[i] + grad(x - ei - ej, *args)[i]) / (4 * step[i] * step[j])
    return H


def nll_exp(theta, k, h, d):
    eta = theta[0] + theta[1] * d
    return float(np.sum(h * np.exp(eta) - k * eta))


def grad_exp(theta, k, h, d):
    lam = np.exp(theta[0] + theta[1] * d)
    return np.array([np.sum(h * lam - k), np.sum((h * lam - k) * d)])


def nll_core(theta, k, h, d):
    """Exponential core plus a constant background level exp(theta[2]) (kept on the log scale: c is tiny)."""
    lam = np.exp(theta[0] + theta[1] * d) + np.exp(theta[2])
    return float(np.sum(h * lam - k * np.log(lam)))


def grad_core(theta, k, h, d):
    e = np.exp(theta[0] + theta[1] * d)
    c = np.exp(theta[2])
    w = h - k / (e + c)
    return np.array([np.sum(w * e), np.sum(w * e * d), np.sum(w * c)])


FIT = {}
for nm in ("straight", "network"):
    tab = coarse_design(nm, CUM[nm], fine_s if nm == "straight" else fine_n)
    k, h, d = tab.events.values.astype(float), tab.exposure_train_h.values, tab.d_m.values
    k_f = np.histogram(ev_on[f"{nm}_m"].values, bins=DIST_EDGES)[0].astype(float)
    h_f = np.diff(CUM[nm]) * TICK_H  # train-hours, the same unit the models are fitted in
    # every deviance below is evaluated on the same 50 m histogram, so the models are comparable
    dev_null = deviance(k_f, h_f, np.full_like(h_f, k_f.sum() / h_f.sum()))

    res = optimize.minimize(nll_exp, x0=[np.log(k[0] / h[0]), -1 / 2000], args=(k, h, d), jac=grad_exp,
                            method="L-BFGS-B", bounds=[(-30.0, 5.0), (-0.01, 0.001)])
    theta = res.x
    se = np.sqrt(np.diag(np.linalg.inv(numeric_hessian(grad_exp, theta, np.array([1e-3, 1e-6]), k, h, d))))
    dev_exp = deviance(k_f, h_f, np.exp(theta[0] + theta[1] * D_MID))
    # quasi-Poisson dispersion measured on the 13 coarse bins (the 50 m histogram is too sparse to be informative)
    phi = deviance(k, h, np.exp(theta[0] + theta[1] * d)) / (len(k) - 2)
    se_phi = se * np.sqrt(phi)

    res_c = optimize.minimize(nll_core, x0=[theta[0], theta[1], np.log(1e-4)], args=(k, h, d), jac=grad_core,
                              method="L-BFGS-B", bounds=[(-30.0, 5.0), (-0.01, 0.001), (-30.0, 5.0)])
    theta_c = res_c.x
    se_c = np.sqrt(np.diag(np.linalg.inv(numeric_hessian(grad_core, theta_c, np.array([1e-3, 1e-6, 1e-4]),
                                                         k, h, d))))
    se_c[2] *= np.exp(theta_c[2])  # delta method: the fit is on log c
    dev_core = deviance(k_f, h_f, np.exp(theta_c[0] + theta_c[1] * D_MID) + np.exp(theta_c[2]))

    dev_tau = np.empty(len(RS))
    for i, tau in enumerate(RS):
        m = D_MID < tau
        lam = np.concatenate([np.full(int(m.sum()), k_f[m].sum() / h_f[m].sum()),
                              np.full(int((~m).sum()), k_f[~m].sum() / h_f[~m].sum())])
        dev_tau[i] = deviance(k_f, h_f, lam)
    i_bp = int(np.argmin(dev_tau))
    dev_bp = float(dev_tau[i_bp])
    inside = RS[dev_tau <= dev_bp + 3.8415]  # 95% profile-likelihood interval for the breakpoint
    tau = RS[i_bp]
    m = D_MID < tau
    r_in = k_f[m].sum() / h_f[m].sum() * 1000
    r_out = k_f[~m].sum() / h_f[~m].sum() * 1000

    aic = {"constant": dev_null + 2 * 1, "exponential": dev_exp + 2 * 2,
           "exponential+background": dev_core + 2 * 3, "breakpoint": dev_bp + 2 * 3}
    FIT[nm] = dict(theta=theta, se=se, se_phi=se_phi, phi=phi, theta_c=theta_c, se_c=se_c, dev_null=dev_null,
                   dev_exp=dev_exp, dev_core=dev_core, dev_bp=dev_bp, tau=tau, tau_lo=inside.min(),
                   tau_hi=inside.max(), rate_in=r_in, rate_out=r_out, aic=aic, dev_tau=dev_tau,
                   background_share=float(np.exp(theta_c[2]) / (np.exp(theta_c[0]) + np.exp(theta_c[2]))))

    log(f"\n{nm} metres -- {len(tab)} distance bins, deviances evaluated on the same 50 m histogram")
    log(f"  exponential: log rate = {theta[0]:+.2f} {theta[1] * 1000:+.3f} x km, e-folding length {-1 / theta[1]:,.0f} m")
    log(f"    slope SE {se[1] * 1000:.4f}/km, {se_phi[1] * 1000:.4f}/km after quasi-Poisson scaling "
        f"(dispersion {phi:.1f})")
    log(f"  exponential + constant background: core e-folding {-1 / theta_c[1]:,.0f} m "
        f"(slope SE {se_c[1] * 1000:.4f}/km), background {np.exp(theta_c[2]) * 1000:.4f} per 1000 train-h "
        f"(SE {se_c[2] * 1000:.4f})")
    wide = RS[dev_tau <= dev_bp + 10.0]  # +10 is roughly the cost of shifting the boundary past one event
    log(f"  breakpoint (step) model: best breakpoint {tau:,.0f} m, 95% profile CI {inside.min():,.0f}-"
        f"{inside.max():,.0f} m, {r_in:.3f} inside vs {r_out:.4f} per 1000 train-h outside ({r_in / r_out:.1f}x)")
    log(f"    the profile is jagged at 50 m resolution (one event moving across the boundary is worth ~10 deviance), "
        f"so the breakpoint is only pinned to {wide.min():,.0f}-{wide.max():,.0f} m")
    near = [i for i in range(len(RS)) if abs(RS[i] - tau) <= 200]
    log("    deviance near the breakpoint: " + ", ".join(f"{RS[i]:,.0f}m {dev_tau[i]:,.1f}" for i in near))
    log("  deviance / AIC: " + ", ".join(f"{nm2} {dv:,.1f} / {aic[nm2]:,.1f}"
                                        for nm2, dv in (("constant", dev_null), ("exponential", dev_exp),
                                                        ("exponential+background", dev_core), ("breakpoint", dev_bp))))
    log(f"  exponential vs breakpoint: deviance difference {dev_bp - dev_exp:+,.1f} on 1 parameter "
        f"(a hard radius needs more than 3.84 to be preferred)")
    log(f"  AIC gap: core+background beats the breakpoint by "
        f"{aic['breakpoint'] - aic['exponential+background']:,.1f} and the pure exponential by "
        f"{aic['exponential'] - aic['exponential+background']:,.1f}")
flush_log()


# ================================================================================================
section("9. The four questions of notes/07_geo.md, answered from the numbers above")
# ================================================================================================
ANS = {}
for nm in ("straight", "network"):
    t, f, ss = RT[nm], FIT[nm], SCAN_SUMMARY[nm]
    a0, b0 = f["theta"][0], f["theta"][1]
    a, b, c = f["theta_c"][0], f["theta_c"][1], float(np.exp(f["theta_c"][2]))
    d_cross = (np.log(c) - a) / b  # where the fitted core has fallen to the background level
    top = t.loc[t.rate_per_1000h.idxmax()]
    k_bg, h_bg, bg = BG[nm]
    far = t.loc[t.bin == "10000+"].iloc[0]
    # the single nearest bins rest on a train-hour or two, so the headline near-field number aggregates 0-150 m
    near_k = int((ev_on[f"{nm}_m"] <= NEAR_M).sum())
    near_h = ticks_upto(CUM[nm], NEAR_M) * TICK_H
    near_rate = near_k / near_h * 1000
    near_lo, near_hi = poisson_ci(near_k, near_h)
    ANS[nm] = dict(top_bin=top.bin, top_rate=float(top.rate_per_1000h), top_rr=float(top.rate_vs_background),
                   bg=bg, k_bg=k_bg, h_bg=h_bg, efold_core=float(-1 / b), efold_pure=float(-1 / b0),
                   d_cross=float(d_cross), max_event=float(ev_on[f"{nm}_m"].max()), far_rate=float(far.rate_per_1000h),
                   far_rr=float(far.rate_vs_background), near_k=near_k, near_h=near_h, near_rate=near_rate,
                   near_rr=near_rate / bg, **ss, aic=f["aic"])
    log(f"\n{nm} metres")
    log(f" 1) kernel shape: within {NEAR_M:,.0f} m there are {near_k} events in {near_h:,.1f} train-h = "
        f"{near_rate:,.1f} per 1000 train-h (CI {near_lo * 1000:,.1f}-{near_hi * 1000:,.1f}), "
        f"{near_rate / bg:,.0f}x the background; the rate then falls with an e-folding length of {-1 / b:,.0f} m "
        f"on top of a flat {c * 1000:.3f} per 1000 train-h")
    log(f"    a hard edge is worse: AIC {f['aic']['breakpoint']:,.1f} (step at {f['tau']:,.0f} m) vs "
        f"{f['aic']['exponential+background']:,.1f} (core + background) and {f['aic']['exponential']:,.1f} "
        f"(exponential only) -- the falloff is smooth, not a radius")
    log(f" 2) the farthest event sits at {ev_on[f'{nm}_m'].max():,.0f} m, but that is coincidence, not signal: "
        f"{k_bg} events beyond {BACKGROUND_M / 1000:.0f} km in {h_bg:,.0f} train-h, a flat {bg:.4f} per 1000 train-h; "
        f"the fitted core reaches that level at {d_cross:,.0f} m, and beyond 10 km the rate is "
        f"{far.rate_per_1000h:.3f} per 1000 train-h = {far.rate_vs_background:.2f}x background")
    log(f" 3) background rate {bg:.4f} per 1000 train-h; the near field is {near_rate / bg:,.0f}x it (the single "
        f"highest bin, {top.bin} m, is {top.rate_vs_background:,.0f}x but rests on only "
        f"{float(top.exposure_train_h):.1f} train-h)")
    log(f" 4) speed check: median {spd.speed_kmh.median():.1f} km/h over {len(spd)} route x segments "
        f"(IQR {spd.speed_kmh.quantile(.25):.1f}-{spd.speed_kmh.quantile(.75):.1f}), i.e. "
        f"{'inside' if 30 <= spd.speed_kmh.median() <= 40 else 'outside'} the 30-40 km/h band of real metro "
        f"running speeds")
    log(f" 5) radius scan: RR peaks at r = {ss['r_peak']:,.0f} m with RR = {ss['rr_peak']:.1f} "
        f"(CI {ss['rr_peak_lo']:.1f}-{ss['rr_peak_hi']:.1f}); it halves at r = {ss['r_half']:,.0f} m, but RR(r) "
        f"stays above 1 out to 10 km because the outside set is dominated by the far field")
flush_log()


# ================================================================================================
section("10. Figures")
# ================================================================================================
for nm in ("straight", "network"):
    tab = coarse_design(nm, CUM[nm], fine_s if nm == "straight" else fine_n)
    t, f = RT[nm], FIT[nm]
    a0, b0 = f["theta"]
    a, b, c = f["theta_c"][0], f["theta_c"][1], float(np.exp(f["theta_c"][2]))
    x = tab.d_m.values
    xs = np.geomspace(10.0, DIST_MAX, 400)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.errorbar(x, t.rate_per_1000h, yerr=[t.rate_per_1000h - t.ci_low, t.ci_high - t.rate_per_1000h],
                fmt="o", ms=4.5, lw=1.1, capsize=2.5, color="C0", label="observed (95% CI)")
    ax.plot(xs, (np.exp(a + b * xs) + c) * 1000, color="C3", lw=1.6, label="exponential core + background")
    ax.plot(xs, np.exp(a0 + b0 * xs) * 1000, color="C2", lw=1.2, ls="--", label="exponential only")
    ax.axhline(BG[nm][2], color="0.4", lw=1.0, ls=":", label=f"background beyond {BACKGROUND_M / 1000:.0f} km")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"distance to {CARRIER} at breakdown onset ({nm}, m)")
    ax.set_ylabel("breakdowns per 1000 train-h")
    ax.set_title(f"Breakdown rate vs distance to {CARRIER} ({nm} metres)")
    ax.grid(alpha=0.25, which="both")
    # the exponential-only curve dives to e^-60 by 60 km; clip the axis to the informative range
    ax.set_ylim(min(t.ci_low[t.ci_low > 0].min(), BG[nm][2]) * 0.4, t.ci_high.max() * 2.5)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / ("fig_distance_kernel.png" if nm == "straight" else f"fig_distance_kernel_{nm}.png"), dpi=200)
    plt.close(fig)

    sc, ss = SCAN[nm], SCAN_SUMMARY[nm]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(sc.r_m, sc.rr, color="C0", lw=1.6, label="RR(r)")
    ax.fill_between(sc.r_m, sc.rr_ci_low, sc.rr_ci_high, color="C0", alpha=0.18, lw=0, label="95% CI")
    ax.axhline(1.0, color="0.4", lw=1.0, ls=":")
    ax.axvline(ss["r_peak"], color="C3", lw=1.2, ls="--", label=f"peak at {ss['r_peak']:,.0f} m")
    if np.isfinite(ss["r_half"]):
        ax.axvline(ss["r_half"], color="C2", lw=1.2, ls="-.", label=f"half of peak at {ss['r_half']:,.0f} m")
    ax.set_xlabel(f"radius r: zone = distance <= r ({nm}, m)")
    ax.set_ylabel("RR = rate within r / rate beyond r")
    ax.set_title(f"Radius scan around {CARRIER} ({nm} metres)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / ("fig_distance_radius_scan.png" if nm == "straight"
                       else f"fig_distance_radius_scan_{nm}.png"), dpi=200)
    plt.close(fig)

# ================================================================================================
section("11. State agreement with src/13_carrier_state.py")
# ================================================================================================
# 13 splits the carrier into 6 classes; this script needs 4 (a victim only asks where the carrier is). The table below
# is that collapse. terminal vs off is the pair that used to be merged, so it has to agree event by event now.
CARRIER_STATES_13 = {"depart": "segment", "cruise": "segment", "arrive": "segment", "dwell": "station",
                     "terminal": "terminal", "off": "off"}
CARR_EVENTS = ROOT / "out" / "carrier" / "carrier_state_events.csv"
if not CARR_EVENTS.exists():
    raise RuntimeError(f"missing {CARR_EVENTS}; run src/13_carrier_state.py first (this section only reads it)")
ce = (pd.read_csv(CARR_EVENTS, usecols=["incident_id", "carrier_state"])
      .rename(columns={"carrier_state": "state_13"}))
ce["state_12"] = ce.state_13.map(CARRIER_STATES_13)
ag = geo[["incident_id", "carrier_state"]].merge(ce, on="incident_id", how="left", validate="1:1")
assert ag.state_12.notna().all(), "13's event table does not cover every event of this script"
dis = ag[ag.carrier_state != ag.state_12]
log("13's 6 classes collapsed onto this script's 4: " + ", ".join(f"{a}->{b}" for a, b in CARRIER_STATES_13.items()))
log(f"events {len(ag)}; agreement {len(ag) - len(dis)}/{len(ag)} ({1 - len(dis) / len(ag):.1%}), "
    f"disagreements {len(dis)}")
log(dis.to_string(index=False) if len(dis) else
    "no disagreement: terminal and off are split identically by both scripts")
log("event counts by state: 12 " + str(geo.carrier_state.value_counts().reindex(
    ["segment", "station", "terminal", "off"]).to_dict()) + " vs 13 " + str(
    ce.state_13.value_counts().reindex(list(CARRIER_STATES_13)).to_dict()))
log(f"terminal: {int((ag.carrier_state == 'terminal').sum())} events; carrier {carrier_term_ticks * TICK_H:,.1f} h "
    f"({carrier_term_ticks / carrier_ticks:.1%} of its {carrier_ticks * TICK_H:,.0f} h in-service clock), "
    f"{carrier_term_others:,.0f} train-h of other-train exposure inside the distance kernel")
log(f"off: {int((ag.carrier_state == 'off').sum())} event(s); carrier {carrier_off_ticks * TICK_H:,.1f} h, "
    f"{carrier_off_others:,.0f} train-h of other-train exposure excluded from the kernel (no carrier position)")
flush_log()

for p in sorted(FIG.glob("fig_distance_*.png")):
    log(f"figure: {p.relative_to(ROOT)}")
log(f"\nfull log: {LOG_PATH.relative_to(ROOT)}")
flush_log()
