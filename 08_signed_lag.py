"""08_signed_lag.py -- signed lag from each zone breakdown to the carrier's occupancy of the relevant track.

Read-only reuse of the 30-s position grid built in src/03_mechanism.py (same exposure intervals, same
station/segment labelling, same distance definition). For each breakdown inside the 2.5-hop carrier zone
(d_c <= 2.5 at the onset tick) it computes the signed lag, in minutes on the 0.5-min tick grid, to the
nearest tick at which carrier T3420 occupied

  - the victim's own undirected segment, if the victim broke down on a segment ('S005 -> S004' and
    'S004 -> S005' are the same track), or
  - any segment incident to the victim's station, if the victim broke down at a station
    (a station has no segment of its own, so its two adjacent tracks are the relevant piece of track).

Sign convention, identical to the lag scan in 03_mechanism.py ("distance between a train now and the
carrier lag_min later"):  lag = t_carrier - t_event, so
    lag < 0  the carrier occupied that track BEFORE the breakdown (victim breaks after the carrier)
    lag = 0  same tick
    lag > 0  the carrier reaches that track AFTER the breakdown (victim breaks first)

Outputs: out/signed_lag/signed_lag_events.csv (one row per zone event) and the four summary answers.
Run: python3 src/08_signed_lag.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, DERIVED = ROOT / "data", ROOT / "data_derived"
OUT = ROOT / "out" / "signed_lag"
OUT.mkdir(parents=True, exist_ok=True)
CARRIER, TICK_S, ZONE_D, OFF = "T3420", 30, 2.5, 99.0
TICK_MIN = TICK_S / 60
EXPECTED_ZONE = 127  # as reported by src/03_mechanism.py ("zone events inside a carrier trip: 127/127")

trav = pd.read_pickle(DERIVED / "traversals.pkl")
bd = pd.read_csv(DERIVED / "breakdowns.csv", parse_dates=["onset_ts", "recovery_ts", "date"])
st = pd.read_csv(DATA / "station_information.csv", encoding="utf-8-sig", dtype=str)


def undirected(loc: str) -> str:
    """'S005 -> S004' and 'S004 -> S005' are the same piece of track; stations stay as they are."""
    return " - ".join(sorted(loc.split(" -> "))) if " -> " in loc else loc


# ---------------------------------------------------------------- exposure intervals + 30-s grid (as in 03)
dw = trav[trav.dwell_s.notna()]
iv = pd.concat([
    pd.DataFrame({"train_id": dw.train_id.values, "line_id": dw.line_id.values, "date": dw.date.values,
                  "loc": dw.from_station.astype(str).values, "kind": "station",
                  "start": (dw.dep_ts - pd.to_timedelta(dw.dwell_s, unit="s")).values, "end": dw.dep_ts.values}),
    pd.DataFrame({"train_id": trav.train_id.values, "line_id": trav.line_id.values, "date": trav.date.values,
                  "loc": trav.segment.astype(str).values, "kind": "segment",
                  "start": trav.dep_ts.values, "end": trav.arr_ts.values}),
], ignore_index=True)
del dw
for c in ["train_id", "line_id", "loc", "kind"]:
    iv[c] = iv[c].astype(str).astype("category")
iv["date"] = pd.to_datetime(iv.date)

LABELS = list(iv["loc"].cat.categories)              # directed segments 'A -> B' and stations 'S...'
TRAINS = list(iv.train_id.cat.categories)
DATES = np.array(sorted(iv.date.unique()))
DATE_IDX = {pd.Timestamp(d): i for i, d in enumerate(DATES)}
TRAIN_IDX = {t: i for i, t in enumerate(TRAINS)}
sec_of_day = lambda ts, day: (ts - day).dt.total_seconds().values  # noqa: E731
t_start_s = float(np.floor(sec_of_day(iv.start, iv.date).min() / 3600) * 3600)
NT = int(np.ceil((sec_of_day(iv.end, iv.date).max() - t_start_s) / TICK_S)) + 1

s_rel = (sec_of_day(iv.start, iv.date) - t_start_s) / TICK_S
e_rel = (sec_of_day(iv.end, iv.date) - t_start_s) / TICK_S
a_t, b_t = np.ceil(s_rel).astype(np.int64), np.ceil(e_rel).astype(np.int64)
n_t = np.maximum(b_t - a_t, 0)
rep = np.repeat(np.arange(len(iv)), n_t)
offs = np.arange(n_t.sum()) - np.repeat(np.cumsum(n_t) - n_t, n_t)
GRID = np.full((len(TRAINS), len(DATES), NT), -1, dtype=np.int16)  # label code, -1 = not in service
GRID[iv.train_id.cat.codes.values[rep], iv.date.map(DATE_IDX).values[rep], a_t[rep] + offs] = \
    iv["loc"].cat.codes.values[rep]
del rep, offs, s_rel, e_rel, a_t, b_t, n_t

# ---------------------------------------------------------------- track keys and the station -> segment map
KEYS = sorted({undirected(lb) for lb in LABELS if " -> " in lb})  # undirected segments
KID = {k: i for i, k in enumerate(KEYS)}
lab_kid = np.array([KID.get(undirected(lb), -1) if " -> " in lb else -1 for lb in LABELS], dtype=np.int16)
incident: dict[str, list[int]] = {}
for i, lb in enumerate(LABELS):
    if " -> " in lb:
        for node in lb.split(" -> "):
            incident.setdefault(node, []).append(int(lab_kid[i]))

ci = TRAIN_IDX[CARRIER]
carr_on = GRID[ci] >= 0
carr_kid = np.full(carr_on.shape, -1, dtype=np.int16)
carr_kid[carr_on] = lab_kid[GRID[ci][carr_on]]
print(f"grid {GRID.shape}; carrier {CARRIER} in service on {int(carr_on.any(1).sum())} of {len(DATES)} dates")

# ---------------------------------------------------------------- distance from a victim position to the carrier
nodes = sorted(set(st.station_id))
adj = {s: set() for s in nodes}
for a, b in zip(st.station_id, st.next_station_id):
    if isinstance(b, str):
        adj[a].add(b)
        adj[b].add(a)
HOP = pd.DataFrame(9999, index=nodes, columns=nodes, dtype=np.int32)
for s in nodes:  # BFS over the undirected track graph
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
        HOP.at[s, v] = x


def dist_label(i: int, j: int) -> float:
    """03's distance: station=node, same undirected segment=0, station-segment=hops+0.5, else 1+hops."""
    if i < 0 or j < 0:
        return OFF
    a, b = LABELS[i].split(" -> "), LABELS[j].split(" -> ")
    h = float(HOP.loc[a, b].values.min())
    if len(a) == 2 and len(b) == 2:
        return 0.0 if set(a) == set(b) else 1.0 + h
    return 0.5 + h if (len(a) == 2 or len(b) == 2) else h


# ---------------------------------------------------------------- events, zone membership, signed lag
bd["ti"] = bd.train_id.map(TRAIN_IDX)
bd["di"] = bd.date.map(DATE_IDX)
bd["tick"] = np.floor((sec_of_day(bd.onset_ts, bd.date) - t_start_s) / TICK_S).astype(np.int64)
ev_lab = GRID[bd.ti.values, bd.di.values, bd.tick.values]
cr_lab = np.where(carr_on[bd.di.values, bd.tick.values], GRID[ci][bd.di.values, bd.tick.values], -1)
bd["d_c"] = [dist_label(int(a), int(b)) for a, b in zip(ev_lab, cr_lab)]
zone = bd[bd.d_c <= ZONE_D].copy()
print(f"events {len(bd)}; inside zone d<={ZONE_D}: {len(zone)} (03_mechanism.py reports {EXPECTED_ZONE})")
print("zone events by d_c:", zone.d_c.value_counts().sort_index().to_dict())

rows = []
for r in zone.itertuples():
    at_station = " -> " not in r.location
    kids = incident.get(r.location, []) if at_station else [KID[undirected(r.location)]]
    mask = np.isin(carr_kid[r.di], kids) if kids else np.zeros(NT, dtype=bool)
    hits = np.nonzero(mask)[0]
    lag, tie = np.nan, False
    if len(hits):
        d = (hits - r.tick) * TICK_MIN
        k = int(np.argmin(np.abs(d)))
        lag = float(d[k])
        tie = bool((np.abs(d) == abs(lag)).sum() > 1)  # carrier passes the track at +x and at -x
    rows.append({"incident_id": r.incident_id, "train_id": r.train_id, "line_id": str(r.line_id),
                 "date": r.date.date(), "onset_ts": r.onset_ts, "location": r.location,
                 "victim_at": "station" if at_station else "segment",
                 "target_track": "; ".join(KEYS[k] for k in kids), "d_c": r.d_c,
                 "carrier_on_date": bool(carr_on[r.di].any()), "lag_min": lag, "tie": tie,
                 "n_carrier_passes": int(len(hits))})
ev = pd.DataFrame(rows)
ev["grp"] = np.where(ev.line_id == "10", "Line 10", "Lines 2/8/12")
ev["sign"] = np.select([ev.lag_min < 0, ev.lag_min == 0, ev.lag_min > 0],
                       ["carrier first (lag<0)", "same tick (lag=0)", "carrier later (lag>0)"], "unassignable")
ev.to_csv(OUT / "signed_lag_events.csv", index=False)


def describe(x: pd.Series) -> str:
    x = x.dropna()
    if not len(x):
        return "n/a"
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return f"median {q2:+.1f} min, IQR [{q1:+.1f}, {q3:+.1f}]"


order = ["carrier first (lag<0)", "same tick (lag=0)", "carrier later (lag>0)", "unassignable"]
for title, sub in [("ALL zone events", ev), ("|lag| <= 2 min", ev[ev.lag_min.abs() <= 2])]:
    tab = pd.crosstab(sub.grp, sub.sign).reindex(index=["Line 10", "Lines 2/8/12"], columns=order,
                                                 fill_value=0).fillna(0).astype(int)
    tab["total"] = tab.sum(axis=1)
    print(f"\n== {title} ==\n{tab.to_string()}")
    for g in ["Line 10", "Lines 2/8/12"]:
        print(f"   {g:12s} {describe(sub.loc[sub.grp == g, 'lag_min'])}")
print(f"\nties at +-x with equal |lag|: {int(ev.tie.sum())}; events with no carrier occupancy of the target "
      f"track on that date: {int(ev.lag_min.isna().sum())} "
      f"(carrier never in service on the event date: {int((~ev.carrier_on_date).sum())})")
