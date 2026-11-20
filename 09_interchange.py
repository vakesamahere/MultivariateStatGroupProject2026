"""09_interchange.py -- decompose the 2.5-hop carrier zone into an interchange mechanism.

Read-only reuse of the 30-s position grid and the distance definition of src/03_mechanism.py. For every
breakdown (143) it builds the covariates that separate "the victim is near the carrier" from "the victim's
line meets the carrier's line at a station and the carrier is at / approaching / leaving that station", then
answers five questions: contingency table, coverage/precision of a one-sentence rule (with the negative
control and a real exposure denominator), per-interchange concentration, a rate per 1000 train-h with a
date-clustered bootstrap CI, and the residual distance effect once interchange structure is held fixed.

Definitions fixed here (derived, never hardcoded):
  interchange            station served by >= 2 of the four lines, from the traversals
  interchange10          interchanges that the carrier's line (line 10) serves
  carrier-occupiable     every station / segment the carrier's line traverses
  relevant node          carrier-occupiable label closest to the victim's position (03's distance), and
                         among ties the one the carrier occupies closest in time. For a same-line victim this
                         is the victim's own track; for a cross-line victim it is the shared station, because
                         the carrier cannot physically reach another line's track.
  signed lag             lag = t_carrier - t_event in minutes on the 0.5-min tick grid (<0 carrier first).

Outputs: out/interchange/events.csv, contingency.csv, rule_precision.csv, per_interchange.csv, rates.csv
Run: python3 src/09_interchange.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, DERIVED = ROOT / "data", ROOT / "data_derived"
OUT = ROOT / "out" / "interchange"
OUT.mkdir(parents=True, exist_ok=True)
CARRIER, TICK_S, ZONE_D, OFF, W_MIN = "T3420", 30, 2.5, 99.0, 2.0
TICK_MIN, N_BOOT = TICK_S / 60, 2000
RNG = np.random.default_rng(20260302)
EXPECTED = {"events": 143, "zone": 127}

trav = pd.read_pickle(DERIVED / "traversals.pkl")
bd = pd.read_csv(DERIVED / "breakdowns.csv", parse_dates=["onset_ts", "recovery_ts", "date"])
st = pd.read_csv(DATA / "station_information.csv", encoding="utf-8-sig", dtype=str)
CARR_LINE = str(trav.loc[trav.train_id == CARRIER, "line_id"].iloc[0])


def undirected(loc: str) -> str:
    """'S005 -> S004' and 'S004 -> S005' are the same piece of track; stations stay as they are."""
    return " - ".join(sorted(loc.split(" -> "))) if " -> " in loc else loc


# ================================================================================================
# 1. 30-s position grid (identical construction to 03 / 08) + station, interchange and track maps
# ================================================================================================
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
print(f"[1] grid {GRID.shape}; {len(LABELS)} labels; {len(DATES)} dates; carrier line {CARR_LINE}")

# --- stations, lines at each station, interchanges (derived from the traversals, not hardcoded) -----------
# Every station appears as a from_ or to_station of some traversal, including trip origins/termini that have no
# dwell interval (e.g. the line-10 terminus S003), so the traversals -- not the exposure intervals -- are the
# authority on which lines serve which station.
st_lines: dict[str, set[str]] = {}
for s, ln in zip(pd.concat([trav.from_station, trav.to_station]).astype(str),
                 pd.concat([trav.line_id, trav.line_id]).astype(str)):
    st_lines.setdefault(s, set()).add(ln)
STATIONS = sorted(st_lines)
INTER = {s for s, ls in st_lines.items() if len(ls) >= 2}
L10_ST = {s for s, ls in st_lines.items() if CARR_LINE in ls}
INTER10 = INTER & L10_ST
ALL_LINES = sorted(set().union(*st_lines.values()))
SHARED_WITH = {ln: {s for s in INTER10 if ln in st_lines[s]} for ln in ALL_LINES}
print(f"[1] stations {len(STATIONS)}; interchanges {len(INTER)}: "
      f"{ {s: '/'.join(sorted(st_lines[s])) for s in sorted(INTER)} }")
print(f"[1] carrier-line interchanges {len(INTER10)}: {sorted(INTER10)}")
print("[1] shared-with-carrier stations per line: "
      f"{ {ln: sorted(v) for ln, v in SHARED_WITH.items() if ln != CARR_LINE} }")

# --- undirected track keys, incident-segment map, hop matrix -----------------------------------------------
KEYS = sorted({undirected(lb) for lb in LABELS if " -> " in lb})
KID = {k: i for i, k in enumerate(KEYS)}
lab_kid = np.array([KID.get(undirected(lb), -1) if " -> " in lb else -1 for lb in LABELS], dtype=np.int16)
L10_KEYS = {KID[undirected(sg)] for sg in trav.loc[trav.line_id == CARR_LINE, "segment"].astype(str).unique()}
incident: dict[str, set[int]] = {}
for i, lb in enumerate(LABELS):
    if " -> " in lb:
        for node in lb.split(" -> "):
            incident.setdefault(node, set()).add(int(lab_kid[i]))

adj: dict[str, set[str]] = {s: set() for s in STATIONS}
for a, b in zip(st.station_id, st.next_station_id):
    if isinstance(b, str) and a in adj and b in adj:
        adj[a].add(b)
        adj[b].add(a)
HOP = pd.DataFrame(9999, index=STATIONS, columns=STATIONS, dtype=np.int32)
for s in STATIONS:  # BFS over the undirected track graph
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
    return float(DL[i, j])


# --- full label x label distance matrix, so nothing downstream needs a per-pair python call --------------
ST_IX = {s: i for i, s in enumerate(STATIONS)}
NL = len(LABELS)
lab_nodes = [lb.split(" -> ") if " -> " in lb else [lb] for lb in LABELS]
lab_is_st = np.array([len(ns) == 1 for ns in lab_nodes])
H1 = np.array([ST_IX[ns[0]] for ns in lab_nodes])
H2 = np.array([ST_IX[ns[-1]] for ns in lab_nodes])
HOPf = HOP.values.astype(np.float32)
MN = np.minimum.reduce([HOPf[np.ix_(H1, H2)], HOPf[np.ix_(H1, H1)],
                        HOPf[np.ix_(H2, H1)], HOPf[np.ix_(H2, H2)]])
DL = np.where(lab_is_st[:, None] & lab_is_st[None, :], MN,
              np.where(lab_is_st[:, None] ^ lab_is_st[None, :], 0.5 + MN, 1.0 + MN)).astype(np.float32)
seg_m = ~lab_is_st
DL[(lab_kid[:, None] == lab_kid[None, :]) & seg_m[:, None] & seg_m[None, :]] = 0.0
ST_CODE_ALL = {lb: i for i, lb in enumerate(LABELS) if lab_is_st[i]}


# --- labels the carrier's line can physically occupy, and the carrier's own state sequence ----------------
OCC = [i for i, lb in enumerate(LABELS)
       if (" -> " not in lb and lb in L10_ST) or (" -> " in lb and lab_kid[i] in L10_KEYS)]
ci = TRAIN_IDX[CARRIER]
carr_on = GRID[ci] >= 0
carr_lab = GRID[ci]
carr_kid = np.where(carr_on, lab_kid[np.maximum(carr_lab, 0)], -1).astype(np.int16)
STAT_CODE = {LABELS[i]: i for i in OCC if " -> " not in LABELS[i]}
print(f"[1] carrier-occupiable labels {len(OCC)} ({len(STAT_CODE)} stations, {len(L10_KEYS)} tracks); "
      f"carrier in service on {int(carr_on.any(1).sum())}/{len(DATES)} dates")

# --- carrier occupancy of each station: dwell ticks plus the arrival/departure instants --------------------
# The grid's station interval is [dep_ts - dwell_s, dep_ts), which misses trip origins (dwell_s is NaN there,
# e.g. the line-10 terminus S003) and the departure instant itself. Both are added so that "carrier at S003"
# is observable; otherwise an interchange where the carrier only turns around would look never visited.
ST_AT = {s: np.zeros((len(DATES), NT), dtype=bool) for s in L10_ST}
for s in L10_ST:
    if s in STAT_CODE:
        ST_AT[s][carr_lab == STAT_CODE[s]] = True
carr_trav = trav[trav.train_id == CARRIER]   # the carrier only, not every line-10 train
st_ts = pd.concat([carr_trav.from_station, carr_trav.to_station]).astype(str).values
st_day = pd.to_datetime(pd.concat([carr_trav.date, carr_trav.date])).values
st_tick = pd.concat([carr_trav.dep_ts, carr_trav.arr_ts])
for s, day, ts in zip(st_ts, st_day, st_tick):
    tick = int(np.ceil((ts - pd.Timestamp(day)).total_seconds() / TICK_S - t_start_s / TICK_S))
    if 0 <= tick < NT and s in ST_AT:
        ST_AT[s][DATE_IDX[pd.Timestamp(day)], tick] = True

# --- carrier state at every tick: the one-sentence rule's left-hand side ----------------------------------
STATE = np.full((len(DATES), NT), "out_of_service", dtype=object)
SNODE = np.full((len(DATES), NT), "", dtype=object)
for di in range(len(DATES)):
    for code in np.unique(carr_lab[di][carr_lab[di] >= 0]):
        m = carr_lab[di] == code
        lb = LABELS[code]
        if " -> " not in lb:
            if lb in INTER10:
                STATE[di][m], SNODE[di][m] = "dwelling_at_interchange", lb
            else:
                STATE[di][m] = "at_ordinary_station"
        else:
            a, b = lb.split(" -> ")
            if b in INTER10:  # interchange ahead in the direction of travel
                STATE[di][m], SNODE[di][m] = "on_segment_approaching_interchange", b
            elif a in INTER10:
                STATE[di][m], SNODE[di][m] = "on_segment_leaving_interchange", a
            else:
                STATE[di][m] = "on_ordinary_segment"
print("[1] carrier ticks by state:",
      {k: int(v) for k, v in pd.Series(STATE.ravel()).value_counts().items()})

# ================================================================================================
# 2. Per-event covariates: location type, shared station, carrier state, signed lag to the relevant node
# ================================================================================================
def hops_to(loc: str, targets) -> int:
    ns = loc.split(" -> ") if " -> " in loc else [loc]
    return int(min(HOP.at[a, b] for a in ns for b in targets)) if len(targets) else 9999


def nearest_ticks(mask: np.ndarray, tick: int) -> tuple[float, int]:
    """Signed lag in minutes to the nearest tick where `mask` holds, plus the number of such ticks."""
    hits = np.nonzero(mask)[0]
    if not len(hits):
        return np.nan, 0
    d = (hits - tick) * TICK_MIN
    return float(d[np.argmin(np.abs(d))]), int(len(hits))


bd["ti"] = bd.train_id.map(TRAIN_IDX)
bd["di"] = bd.date.map(DATE_IDX)
bd["tick"] = np.floor((sec_of_day(bd.onset_ts, bd.date) - t_start_s) / TICK_S).astype(np.int64)
bd["ev_lab"] = GRID[bd.ti.values, bd.di.values, bd.tick.values]
bd["d_c"] = [dist_label(int(a), int(carr_lab[d, t])) for a, d, t in
              zip(bd.ev_lab, bd.di.values, bd.tick.values)]
bd["line"] = bd.line_id.astype(str)
print(f"[2] events {len(bd)}; inside zone d<={ZONE_D}: {int((bd.d_c <= ZONE_D).sum())} (expected {EXPECTED['zone']})")
print("[2] d_c histogram:", bd.d_c.value_counts().sort_index().to_dict())

rows = []
for r in bd.itertuples():
    at_st = " -> " not in r.location
    nodes = [r.location] if at_st else r.location.split(" -> ")
    # victim location type
    if at_st:
        vtype = "interchange_station" if r.location in INTER else "ordinary_station"
    else:
        vtype = "segment_adjacent_to_interchange" if any(n in INTER for n in nodes) \
            else "ordinary_segment"
    # shared stations between the victim's line and the carrier's line
    shared = sorted(SHARED_WITH[r.line]) if r.line != CARR_LINE else sorted(INTER10)
    # the rule needs BOTH sides: the victim at a shared station or on a track incident to it, and the carrier
    # at that same station within +-W_MIN. Candidate stations are therefore restricted to hops <= 1.
    cands = [s for s in shared if hops_to(r.location, {s}) <= 1]
    tight = [s for s in shared if hops_to(r.location, {s}) == 0]
    near_shared = min(cands, key=lambda s: hops_to(r.location, {s})) if cands else \
        (min(shared, key=lambda s: hops_to(r.location, {s})) if shared else "")
    lag_shared = {s: nearest_ticks(ST_AT[s][r.di], r.tick) for s in cands}
    fires = sorted(((abs(v[0]), s, v[0]) for s, v in lag_shared.items()
                    if not np.isnan(v[0]) and abs(v[0]) <= W_MIN))
    tight_fires = []
    for s in tight:
        lag, _ = nearest_ticks(ST_AT[s][r.di], r.tick)
        if not np.isnan(lag) and abs(lag) <= W_MIN:
            tight_fires.append((abs(lag), s, lag))
    tight_fires.sort()
    # relevant node: carrier-occupiable label closest to the victim, ties broken by nearest in time. Directed
    # labels are collapsed to their undirected track first, otherwise 'A -> B' and 'B -> A' both qualify at
    # distance 0 and every same-line event looks like a tie.
    dists = DL[int(r.ev_lab), np.array(OCC)]
    seen_key, cand = set(), []
    for o in np.array(OCC)[dists == dists.min()]:
        lb = LABELS[o]
        key = undirected(lb)
        if key not in seen_key:
            seen_key.add(key)
            cand.append((lb, o))
    lags = []
    for lb, o in cand:
        mask = ST_AT[lb][r.di] if " -> " not in lb else (carr_kid[r.di] == lab_kid[o])
        lag, _ = nearest_ticks(mask, r.tick)
        if not np.isnan(lag):
            lags.append((lb, lag))
    best = min(lags, key=lambda x: abs(x[1])) if lags else ("", np.nan)
    scope = min(lag_shared, key=lambda s: abs(lag_shared[s][0])) if lag_shared else ""
    carr_here = int(carr_lab[r.di, r.tick])
    rows.append({
        "incident_id": r.incident_id, "train_id": r.train_id, "line": r.line, "date": r.date.date(),
        "onset_ts": r.onset_ts, "location": r.location, "victim_at": "station" if at_st else "segment",
        "d_c": r.d_c, "in_zone": bool(r.d_c <= ZONE_D), "victim_location_type": vtype,
        "victim_line_shares_station_with_carrier": bool(shared),
        "near_shared_station": near_shared,
        "hops_to_near_shared": hops_to(r.location, {near_shared}) if near_shared else np.nan,
        "carrier_state": STATE[r.di, r.tick], "carrier_state_node": SNODE[r.di, r.tick],
        "rule_fires": bool(fires), "rule_station": fires[0][1] if fires else "",
        "rule_lag_min": fires[0][2] if fires else np.nan,
        "tight_fires": bool(tight_fires), "tight_station": tight_fires[0][1] if tight_fires else "",
        "tight_set": " ".join(tight), "ev_lab": int(r.ev_lab),
        "scope_station": scope,
        "scope_lag_min": lag_shared[scope][0] if scope else np.nan,
        "carrier_dist_to_scope": float(DL[ST_CODE_ALL[scope], carr_here]) if (scope and carr_here >= 0)
        else OFF,
        "relevant_node": best[0], "relevant_node_dist": float(dists.min()),
        "lag_min": best[1], "lag_tie": bool(len(lags) > 1 and sum(abs(l) == abs(best[1]) for _, l in lags) > 1),
    })
ev = pd.DataFrame(rows)
ev["grp"] = np.where(ev.line == CARR_LINE, f"Line {CARR_LINE}", "Lines 2/8/12")
ev["sign"] = np.select([ev.lag_min < 0, ev.lag_min == 0, ev.lag_min > 0],
                       ["carrier first (lag<0)", "same tick (lag=0)", "carrier later (lag>0)"], "unassignable")
ev.to_csv(OUT / "events.csv", index=False)
print(f"[2] unassignable {int((ev.sign == 'unassignable').sum())}; "
      f"lag tie {int(ev.lag_tie.sum())}; rule fires {int(ev.rule_fires.sum())}/{len(ev)}")
print("[2] relevant node of the cross-line zone events:",
      ev[ev.grp != f"Line {CARR_LINE}"].relevant_node.value_counts().head(8).to_dict())

# ================================================================================================
# 3. Contingency table: victim location type x carrier state
# ================================================================================================
VORDER = ["interchange_station", "segment_adjacent_to_interchange", "ordinary_station", "ordinary_segment"]
SORDER = ["dwelling_at_interchange", "on_segment_approaching_interchange", "on_segment_leaving_interchange",
          "at_ordinary_station", "on_ordinary_segment", "out_of_service"]
tab = []
for scope_name, sub in [("zone (d<=2.5)", ev[ev.in_zone]), ("cross-line zone", ev[ev.in_zone & (ev.line != CARR_LINE)]),
                        ("non-zone", ev[~ev.in_zone]), ("all", ev)]:
    ct = pd.crosstab(sub.victim_location_type, sub.carrier_state).reindex(
        index=VORDER, columns=SORDER, fill_value=0)
    print(f"\n[3] {scope_name} events n={len(sub)}\n{ct.to_string()}")
    flat = ct.stack()
    (r_, c_), v_ = flat.idxmax(), int(flat.max())
    print(f"[3]   dominant cell {r_} x {c_} = {v_}/{len(sub)} ({v_ / len(sub):.1%}); "
          f"row totals {dict(zip(VORDER, ct.sum(1).tolist()))}")
    out = ct.copy()
    out.insert(0, "scope", scope_name)
    tab.append(out.reset_index())
pd.concat(tab).to_csv(OUT / "contingency.csv", index=False)

# ================================================================================================
# 4/5. Rule coverage, and precision against a real exposure denominator (carrier-tick x train-tick)
# ================================================================================================
def window(mask: np.ndarray, w: int) -> np.ndarray:
    """Ticks within +-w ticks of `mask` (inclusive), per date row."""
    out = mask.copy()
    for k in range(1, w + 1):
        out[:, k:] |= mask[:, :-k]
        out[:, :-k] |= mask[:, k:]
    return out


W_T = int(round(W_MIN / TICK_MIN))
LINE_OF_TRAIN = dict(zip(trav.train_id.astype(str), trav.line_id.astype(str)))
ev["tick"] = np.floor((sec_of_day(ev.onset_ts, pd.to_datetime(ev.date)) - t_start_s) / TICK_S).astype(int)
ev["di"] = pd.to_datetime(ev.date).map(DATE_IDX)

print(f"\n[4] rule coverage (victim at/incident to shared I AND carrier at I within +-{W_MIN:g} min)")
for name, sub in [("all", ev), ("cross-line", ev[ev.line != CARR_LINE]), ("line 10", ev[ev.line == CARR_LINE]),
                  ("cross-line zone", ev[(ev.line != CARR_LINE) & ev.in_zone]),
                  ("line 10 zone", ev[(ev.line == CARR_LINE) & ev.in_zone])]:
    print(f"[4]   {name:17s} n={len(sub):3d} covered {int(sub.tight_fires.sum()):3d} ({sub.tight_fires.mean():5.1%})")

per_station, N_all = [], {k: np.zeros(len(DATES)) for k in ("near", "mid", "far", "ordst")}
K_all = {k: np.zeros(len(DATES)) for k in ("near", "mid", "far", "ordst")}
TRAIN_LINE = np.array([LINE_OF_TRAIN[t] for t in TRAINS])
for I in INTER10:
    lines = sorted(ln for ln in SHARED_WITH if I in SHARED_WITH[ln] and ln != CARR_LINE)
    codes = [ST_CODE_ALL[I]] + [i for i, lb in enumerate(LABELS)
                                if " -> " in lb and lab_kid[i] in incident.get(I, set())]
    cd = np.full(carr_lab.shape, OFF, dtype=np.float32)
    cd[carr_on] = DL[ST_CODE_ALL[I], carr_lab[carr_on]]
    near = window(ST_AT[I], W_T)
    mid = (cd <= ZONE_D) & ~near
    far = ~near & ~mid
    ordst = (near | mid | far) & (STATE == "at_ordinary_station")
    cls = {"near": near, "mid": mid, "far": far, "ordst": ordst}
    N = {k: np.zeros(len(DATES)) for k in cls}
    for ln in lines:                                     # exposure: other-line trains in scope of I
        for ti in np.nonzero(TRAIN_LINE == ln)[0]:
            m = np.isin(GRID[ti], codes)
            for k in cls:
                N[k] += (m & cls[k]).sum(1)
    # an event whose victim track is incident to two interchanges is assigned to the first of them only,
    # so that the per-station event counts sum to the number of distinct events (no double counting)
    E = ev[ev.tight_set.apply(lambda s: s.split()[0] == I if s.split() else False)]
    K = {k: np.zeros(len(DATES)) for k in cls}
    for e in E.to_dict("records"):
        lag, _ = nearest_ticks(ST_AT[I][e["di"]], e["tick"])
        k = "near" if (not np.isnan(lag) and abs(lag) <= W_MIN) else \
            ("mid" if cd[e["di"], e["tick"]] <= ZONE_D else "far")
        K[k][e["di"]] += 1
        K["ordst"][e["di"]] += int(STATE[e["di"], e["tick"]] == "at_ordinary_station")
    for k in cls:
        N_all[k] += N[k]
        K_all[k] += K[k]
    hours = N["near"] * TICK_S / 3600.0
    per_station.append({
        "interchange": I, "lines": "/".join(lines), "n_events_in_scope": len(E),
        "n_events_near": int(K["near"].sum()), "exposure_ticks_near": int(N["near"].sum()),
        "exposure_train_h_near": hours.sum(),
        "rate_per_1000_train_h_near": 1000 * K["near"].sum() / hours.sum() if hours.sum() else np.nan,
        "precision_near_pct": 100 * K["near"].sum() / N["near"].sum() if N["near"].sum() else np.nan,
        "n_events_mid": int(K["mid"].sum()), "exposure_ticks_mid": int(N["mid"].sum()),
        "n_events_far": int(K["far"].sum()), "exposure_ticks_far": int(N["far"].sum()),
        "precision_mid_pct": 100 * K["mid"].sum() / N["mid"].sum() if N["mid"].sum() else np.nan,
        "precision_far_pct": 100 * K["far"].sum() / N["far"].sum() if N["far"].sum() else np.nan,
    })
    print(f"[4]   {I} lines {'/'.join(lines)}: events {len(E):3d}, near {int(K['near'].sum()):3d} "
          f"mid {int(K['mid'].sum()):2d} far {int(K['far'].sum()):2d} | exposure near {int(N['near'].sum()):7d} "
          f"far {int(N['far'].sum()):8d} ticks")

per = pd.DataFrame(per_station)
per.to_csv(OUT / "per_interchange.csv", index=False)


def rates(N, K):
    h = {k: v * TICK_S / 3600.0 for k, v in N.items()}
    out = {}
    for k in ("near", "mid", "far"):
        out[f"rate_{k}"] = 1000 * K[k].sum() / h[k].sum()
        out[f"prec_{k}"] = 100 * K[k].sum() / N[k].sum()
    out["prec_ord"] = 100 * K["ordst"].sum() / N["ordst"].sum()
    return out


pt = rates(N_all, K_all)
keys = list(pt)
boot = {k: [] for k in keys}
for _ in range(N_BOOT):                                    # date-clustered bootstrap
    d = RNG.integers(0, len(DATES), len(DATES))
    r = rates({k: v[d] for k, v in N_all.items()}, {k: v[d] for k, v in K_all.items()})
    for k in keys:
        boot[k].append(r[k])
ci = {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in boot.items()}
rr = np.array(boot["rate_near"]) / np.array(boot["rate_mid"])
print(f"\n[4] event rate per 1000 train-h at scope(I), by carrier position at the same tick")
print(f"[4]   carrier AT I (+-{W_MIN:g} min):  {pt['rate_near']:.3f} [{ci['rate_near'][0]:.2f},{ci['rate_near'][1]:.2f}]"
      f"  n={int(K_all['near'].sum())} events / {N_all['near'].sum() * TICK_S / 3600:,.0f} train-h")
print(f"[4]   carrier within 2.5 hops only:   {pt['rate_mid']:.3f} [{ci['rate_mid'][0]:.2f},{ci['rate_mid'][1]:.2f}]"
      f"  n={int(K_all['mid'].sum())} / {N_all['mid'].sum() * TICK_S / 3600:,.0f} train-h")
print(f"[4]   carrier >2.5 hops (control):    {pt['rate_far']:.3f} [{ci['rate_far'][0]:.2f},{ci['rate_far'][1]:.2f}]"
      f"  n={int(K_all['far'].sum())} / {N_all['far'].sum() * TICK_S / 3600:,.0f} train-h")
print(f"[4]   RR at-I / band-only = {pt['rate_near'] / pt['rate_mid']:.1f} "
      f"[{np.nanpercentile(rr, 2.5):.1f},{np.nanpercentile(rr, 97.5):.1f}]")
print(f"[4] precision per exposure tick: at-I {pt['prec_near']:.4f}% {ci['prec_near']} | band-only "
      f"{pt['prec_mid']:.4f}% {ci['prec_mid']} | far {pt['prec_far']:.4f}% {ci['prec_far']} | "
      f"carrier-at-ordinary-station {pt['prec_ord']:.4f}% {ci['prec_ord']}")
pd.DataFrame([{"class": k, "rate_per_1000_train_h": pt[f"rate_{k}"], "n_events": int(K_all[k].sum()),
               "exposure_train_h": float(N_all[k].sum() * TICK_S / 3600),
               "rate_ci_lo": ci[f"rate_{k}"][0], "rate_ci_hi": ci[f"rate_{k}"][1],
               "precision_pct": pt[f"prec_{k}"], "prec_ci_lo": ci[f"prec_{k}"][0], "prec_ci_hi": ci[f"prec_{k}"][1]}
              for k in ("near", "mid", "far")]).to_csv(OUT / "rates.csv", index=False)

# --- 4b. matched negative control: the same geometry at a station that is NOT shared --------------------
# Arm A counts other-line trains standing at an interchange I while the carrier is at I; arm B counts
# other-line trains standing at an ordinary station of their own line while the carrier is at an ordinary
# station of its own line. Same "both trains at a station at the same time" opportunity, no shared node.
ord_c = window(STATE == "at_ordinary_station", W_T)
# 8 of 120 stations are line terminuses that never appear as a standalone grid label (no dwell row);
# none of them is an interchange, so the control arm simply omits them.
ORD_ST = [st for st in STATIONS if st not in INTER and st in ST_CODE_ALL]
ord_codes = np.array([ST_CODE_ALL[st] for st in ORD_ST], dtype=np.int16)
print(f"[4b] control arm covers {len(ORD_ST)}/{len(STATIONS) - len(INTER)} ordinary stations")
NA = np.zeros(len(DATES)); KA = np.zeros(len(DATES))
NB = np.zeros(len(DATES)); KB = np.zeros(len(DATES))
for I in INTER10:
    sc = np.array([ST_CODE_ALL[I]], dtype=np.int16)
    for ti in np.nonzero(TRAIN_LINE != CARR_LINE)[0]:
        NA += (np.isin(GRID[ti], sc) & window(ST_AT[I], W_T)).sum(1)
for ti in np.nonzero(TRAIN_LINE != CARR_LINE)[0]:
    NB += (np.isin(GRID[ti], ord_codes) & ord_c).sum(1)
at_st = ev[ev.victim_at == "station"]
for e in at_st.to_dict("records"):
    if e["line"] == CARR_LINE:
        continue
    if e["victim_location_type"] == "interchange_station" and e["carrier_state"] == "dwelling_at_interchange":
        lag, _ = nearest_ticks(ST_AT[e["location"]][e["di"]], e["tick"])
        KA[e["di"]] += int(not np.isnan(lag) and abs(lag) <= W_MIN)
    if e["victim_location_type"] == "ordinary_station" and ord_c[e["di"], e["tick"]]:
        KB[e["di"]] += 1


def _r(k, n):
    return 1000 * k.sum() / (n.sum() * TICK_S / 3600)


rrb = []
for _ in range(N_BOOT):
    d = RNG.integers(0, len(DATES), len(DATES))
    rrb.append(_r(KA[d], NA[d]) / _r(KB[d], NB[d]))
print(f"[4b] matched control: shared station A {int(KA.sum())}/{NA.sum() * TICK_S / 3600:,.0f} train-h = "
      f"{_r(KA, NA):.3f}/1000 train-h | ordinary station B {int(KB.sum())}/{NB.sum() * TICK_S / 3600:,.0f} = "
      f"{_r(KB, NB):.3f}/1000 train-h | RR {_r(KA, NA) / _r(KB, NB):.2f} "
      f"[{np.nanpercentile(rrb, 2.5):.2f},{np.nanpercentile(rrb, 97.5):.2f}]")
pd.DataFrame([{"arm": "A_shared_station", "n_events": int(KA.sum()), "train_h": float(NA.sum() * TICK_S / 3600),
               "rate_per_1000_train_h": _r(KA, NA)},
              {"arm": "B_ordinary_station_control", "n_events": int(KB.sum()),
               "train_h": float(NB.sum() * TICK_S / 3600), "rate_per_1000_train_h": _r(KB, NB)}]).to_csv(
    OUT / "matched_control.csv", index=False)
