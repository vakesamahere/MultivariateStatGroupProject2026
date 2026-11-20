"""05 Permutation / bootstrap validation of the T3420 "carrier" proximity effect.

Stages
  --stage export   (local; needs pandas) build a compact, version-safe .npz/.json export of the 30-s position grid,
                   breakdown events and network distances in data_derived/perm_export/, then recompute the observed
                   statistic from the export alone and check it against notes/03_mechanism.md.
  --stage permute  (cluster; numpy only) null scheme A (whole-day circular shift of the carrier's trajectory),
                   null scheme B (carrier label permutation), date-cluster bootstrap of the distance kernel.
  --stage figure   (local; matplotlib) figures/mechanism/fig_permutation.png from out/permutation/.

Grid and distance definitions follow src/03_mechanism.py section 3 exactly: a tick is the instant
05:30 + i*30 s; an exposure interval [s, e) (station dwell [dep_ts - dwell_s, dep_ts), segment [dep_ts, arr_ts),
terminal layovers excluded) covers ticks ceil(s)..ceil(e)-1. Distance between two positions in network hops on the
undirected track graph: station-station = hops; station-segment = 0.5 + hops to the nearer end; segment-segment = 0 if
the same undirected piece of track, else 1 + hops between the nearest ends. Distances are stored as half-hops (int).

Statistic (primary): breakdowns of trains other than the carrier per 1000 train-h those trains spend within
d <= 2.5 hops of the carrier ("zone rate"); also the zone RR vs everything else (d >= 3 or carrier out of service)
and the scan rate within d <= 1.5 hops (03's carrier-scan score).

Run:  python src/05_permutation.py --stage export
      python src/05_permutation.py --stage permute --export-dir data/perm_export --out-dir out/permutation
      python src/05_permutation.py --stage figure
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TICK_S = 30
GRID_START_S = 5 * 3600 + 30 * 60  # 05:30
GRID_END_S = 24 * 3600  # 24:00 (exclusive)
NT = (GRID_END_S - GRID_START_S) // TICK_S  # 2220 ticks
CARRIER = "T3420"
SEED = 20260305
R_SHIFT = 2000  # null scheme A replicates
R_LABEL = 2000  # null scheme B replicates
B_BOOT = 2000  # date-cluster bootstrap replicates
# Distance classes in half-hops: 0..5 = d 0, 0.5, ..., 2.5; 6 = d >= 3; 7 = carrier (or victim) out of service.
D_LABELS = ["0", "0.5", "1", "1.5", "2", "2.5", ">=3", "carrier off"]
N_CLS = 8
ZONE_MAX_CLS = 5  # d <= 2.5
NEAR_MAX_CLS = 3  # d <= 1.5
EXPECTED = {"k_zone": 127, "h_zone": 6159, "rate_zone": 20.6, "k_far": 16, "h_far": 164220, "rate_far": 0.097,
            "rr_zone": 212, "rate_near_1p5": 45.2}


# ================================================================================================
# shared statistics (numpy only)
# ================================================================================================
def stats_from_counts(k: np.ndarray, h_ticks: np.ndarray) -> dict:
    """k, h_ticks: (..., N_CLS) event counts and train-ticks by distance class -> summary statistics."""
    k = np.asarray(k, dtype=np.float64)
    h = np.asarray(h_ticks, dtype=np.float64) * TICK_S / 3600
    kz, hz = k[..., :ZONE_MAX_CLS + 1].sum(-1), h[..., :ZONE_MAX_CLS + 1].sum(-1)
    kf, hf = k[..., ZONE_MAX_CLS + 1:].sum(-1), h[..., ZONE_MAX_CLS + 1:].sum(-1)
    kn, hn = k[..., :NEAR_MAX_CLS + 1].sum(-1), h[..., :NEAR_MAX_CLS + 1].sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rz = np.where(hz > 0, kz / hz * 1000, 0.0)
        rf = kf / hf * 1000
        rn = np.where(hn > 0, kn / hn * 1000, 0.0)
        rr = np.where(kf > 0, rz / rf, np.inf)
    return {"k_zone": kz, "h_zone": hz, "rate_zone": rz, "k_far": kf, "h_far": hf, "rate_far": rf,
            "rr_zone": rr, "k_near_1p5": kn, "h_near_1p5": hn, "rate_near_1p5": rn}


def class_counts(grid_d: np.ndarray, car_traj: np.ndarray, dcls: np.ndarray, exclude_train: int) -> np.ndarray:
    """Train-ticks by distance class on one date.

    grid_d: (n_trains, NT) location codes (-1 = not in service) of the victims on that date;
    car_traj: (NT,) carrier location codes (-1 = off) used against them; dcls: (n_loc + 1, n_loc + 1) class table
    with the last row/col = out of service. Victim ticks out of service and the carrier's own row are excluded.
    """
    sent = dcls.shape[0] - 1
    g = np.where(grid_d < 0, sent, grid_d)
    c = np.where(car_traj < 0, sent, car_traj)
    cls = dcls[g, c[None, :]]
    m = grid_d >= 0
    m[exclude_train] = False
    return np.bincount(cls[m], minlength=N_CLS).astype(np.int64)


# ================================================================================================
# stage: export (local, pandas)
# ================================================================================================
def stage_export(export_dir: Path) -> None:
    import pandas as pd

    t0 = time.time()
    export_dir.mkdir(parents=True, exist_ok=True)
    derived, data = ROOT / "data_derived", ROOT / "data"
    trav = pd.read_pickle(derived / "traversals.pkl")
    bd = pd.read_csv(derived / "breakdowns.csv", parse_dates=["onset_ts", "date"])
    st = pd.read_csv(data / "station_information.csv", encoding="utf-8-sig", dtype=str)
    trains_info = pd.read_csv(data / "train_information.csv", encoding="utf-8-sig", dtype=str)

    # ---- exposure intervals (same as 03 section 1) ----
    dw = trav[trav.dwell_s.notna()]
    iv = pd.concat([
        pd.DataFrame({"train_id": dw.train_id.astype(str).values, "date": dw.date.values,
                      "loc": dw.from_station.astype(str).values,
                      "start": (dw.dep_ts - pd.to_timedelta(dw.dwell_s, unit="s")).values, "end": dw.dep_ts.values}),
        pd.DataFrame({"train_id": trav.train_id.astype(str).values, "date": trav.date.values,
                      "loc": trav.segment.astype(str).values, "start": trav.dep_ts.values, "end": trav.arr_ts.values}),
    ], ignore_index=True)
    del dw
    iv["date"] = pd.to_datetime(iv.date)
    total_h = float((iv.end - iv.start).dt.total_seconds().sum() / 3600)

    # ---- codes ----
    nodes = sorted(st.station_id.unique())
    nidx = {s: i for i, s in enumerate(nodes)}
    labels = sorted(iv["loc"].unique())  # stations 'Sxxx' and directed segments 'A -> B'
    lidx = {s: i for i, s in enumerate(labels)}
    trains = sorted(iv.train_id.unique())
    tidx = {t: i for i, t in enumerate(trains)}
    dates = sorted(iv.date.unique())
    didx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    assert len(trains) == 140 and len(dates) == 105, (len(trains), len(dates))

    # ---- grid ----
    sec = lambda ts, day: (ts - day).dt.total_seconds().values  # noqa: E731
    s_rel = (sec(iv.start, iv.date) - GRID_START_S) / TICK_S
    e_rel = (sec(iv.end, iv.date) - GRID_START_S) / TICK_S
    a_t, b_t = np.ceil(s_rel).astype(np.int64), np.ceil(e_rel).astype(np.int64)
    n_t = np.maximum(b_t - a_t, 0)
    assert a_t[n_t > 0].min() >= 0 and b_t[n_t > 0].max() <= NT, "exposure outside the 05:30-24:00 grid"
    rep = np.repeat(np.arange(len(iv)), n_t)
    offs = np.arange(n_t.sum()) - np.repeat(np.cumsum(n_t) - n_t, n_t)
    grid = np.full((len(trains), len(dates), NT), -1, dtype=np.int16)
    loc_code = iv["loc"].map(lidx).values.astype(np.int16)
    grid[iv.train_id.map(tidx).values[rep], iv.date.map(didx).values[rep], a_t[rep] + offs] = loc_code[rep]
    n_cells = int(n_t.sum())
    assert int((grid >= 0).sum()) == n_cells, "overlapping exposure intervals on the grid"
    del rep, offs

    # ---- network ----
    edges = sorted({tuple(sorted((a, b))) for a, b in zip(st.station_id, st.next_station_id) if isinstance(b, str)})
    edge_arr = np.array([[nidx[a], nidx[b]] for a, b in edges], dtype=np.int16)
    adj = [[] for _ in nodes]
    for a, b in edge_arr:
        adj[a].append(b)
        adj[b].append(a)
    hop = np.full((len(nodes), len(nodes)), -1, dtype=np.int16)
    for s in range(len(nodes)):
        hop[s, s], frontier = 0, [s]
        while frontier:
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if hop[s, v] < 0:
                        hop[s, v] = hop[s, u] + 1
                        nxt.append(v)
            frontier = nxt
    assert (hop >= 0).all(), "track graph not connected"
    loc_a = np.array([nidx[lb.split(" -> ")[0]] for lb in labels], dtype=np.int16)
    loc_b = np.array([nidx[lb.split(" -> ")[1]] if " -> " in lb else -1 for lb in labels], dtype=np.int16)
    is_seg = loc_b >= 0
    nl = len(labels)
    dist2 = np.empty((nl, nl), dtype=np.int16)  # distance in half-hops
    for i in range(nl):
        ni = [loc_a[i]] + ([loc_b[i]] if is_seg[i] else [])
        for j in range(nl):
            nj = [loc_a[j]] + ([loc_b[j]] if is_seg[j] else [])
            h = int(hop[np.ix_(ni, nj)].min())
            if is_seg[i] and is_seg[j]:
                dist2[i, j] = 0 if set(ni) == set(nj) else 2 * (1 + h)
            elif is_seg[i] or is_seg[j]:
                dist2[i, j] = 2 * h + 1
            else:
                dist2[i, j] = 2 * h

    # ---- events ----
    ev_tick = np.floor((sec(bd.onset_ts, bd.date) - GRID_START_S) / TICK_S).astype(np.int64)
    ev_train = bd.train_id.map(tidx).values.astype(np.int16)
    ev_date = bd.date.map(didx).values.astype(np.int16)
    ev_loc = bd.location.map(lidx).values
    assert not np.isnan(ev_loc.astype(float)).any(), "event location not in the label set"
    ev_loc = ev_loc.astype(np.int16)
    ev_grid_loc = grid[ev_train, ev_date, ev_tick]
    match = float((ev_grid_loc == ev_loc).mean())
    line_of = trains_info.set_index("train_id").line_id.reindex(trains).values
    train_line_is10 = (line_of == "10").astype(np.int8)

    np.savez_compressed(
        export_dir / "perm_export.npz",
        grid=grid, occupied=grid >= 0,
        ev_train=ev_train, ev_date=ev_date, ev_tick=ev_tick.astype(np.int16), ev_loc=ev_loc,
        ev_grid_loc=ev_grid_loc.astype(np.int16),
        loc_node_a=loc_a, loc_node_b=loc_b, edges=edge_arr, node_hops=hop, loc_dist_halfhops=dist2,
        train_line_is10=train_line_is10,
    )
    meta = {
        "description": "Compact export for src/05_permutation.py --stage permute (integer codes only).",
        "grid": {"shape": list(grid.shape), "axes": ["train", "date", "tick"], "dtype": "int16",
                 "value": "location code (index into labels) while in service, -1 = not in service",
                 "tick_s": TICK_S, "tick0_clock": "05:30:00", "n_ticks": NT,
                 "tick_rule": "tick i = instant 05:30 + 30*i s; interval [s,e) covers ticks ceil(s)..ceil(e)-1",
                 "exposure": "station dwell [dep_ts-dwell_s, dep_ts) at from_station; segment [dep_ts, arr_ts); "
                             "terminal layovers excluded",
                 "occupied_train_ticks": n_cells, "exact_interval_train_h": round(total_h, 2)},
        "locations": {"labels": labels, "kind": ["segment" if s else "station" for s in is_seg],
                      "loc_node_a": "from-node (station: the node)", "loc_node_b": "to-node of a segment, -1 = station"},
        "loc_dist_halfhops": "network-hop distance x 2 between location codes: station-station 2*hops; "
                             "station-segment 2*hops_to_nearer_end+1; same undirected segment 0; other segments "
                             "2*(1+hops between nearest ends)",
        "nodes": nodes, "edges": "undirected node-index pairs from station_information adjacency",
        "trains": trains, "dates": [str(pd.Timestamp(d).date()) for d in dates],
        "carrier": CARRIER, "carrier_code": tidx[CARRIER],
        "events": {"n": int(len(bd)), "incident_id": bd.incident_id.tolist(),
                   "tick_rule": "floor((onset - 05:30)/30 s)",
                   "grid_location_matches_recorded": match},
        "statistics": {
            "distance_classes": D_LABELS,
            "rate_zone": "events of trains != carrier at d<=2.5 hops of the carrier per 1000 train-h those trains "
                         "spend at d<=2.5 (exposure: in-service grid ticks x 30 s, carrier's own ticks excluded)",
            "rate_far": "same for d>=3 hops or carrier out of service",
            "rr_zone": "rate_zone / rate_far",
            "rate_near_1p5": "same as rate_zone with d<=1.5 (03 carrier-scan score)",
            "expected_from_03": EXPECTED},
    }
    (export_dir / "perm_export.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"export written in {time.time() - t0:.0f}s: grid {grid.shape}, {n_cells:,} occupied train-ticks "
          f"({n_cells * TICK_S / 3600:,.0f} train-h; exact intervals {total_h:,.0f}); {nl} location codes, "
          f"{len(nodes)} nodes, {len(edge_arr)} edges; {len(bd)} events, grid location at onset matches recorded "
          f"location {match:.1%}")
    for f in sorted(export_dir.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.2f} MB")
    verify_export(export_dir)


def load_export(export_dir: Path) -> tuple[dict, dict]:
    z = np.load(export_dir / "perm_export.npz", allow_pickle=False)
    arrs = {k: z[k] for k in z.files}
    meta = json.loads((export_dir / "perm_export.json").read_text(encoding="utf-8"))
    return arrs, meta


def class_table(dist2: np.ndarray) -> np.ndarray:
    """(n_loc + 1)^2 int8 distance-class table; last row/col = out of service -> class 7."""
    nl = dist2.shape[0]
    t = np.full((nl + 1, nl + 1), N_CLS - 1, dtype=np.int8)
    t[:nl, :nl] = np.minimum(dist2, 6)
    return t


def event_classes(arrs: dict, dcls: np.ndarray, car_traj_of_date: np.ndarray) -> np.ndarray:
    """Distance class of every event given the carrier trajectory per date (n_dates, NT)."""
    sent = dcls.shape[0] - 1
    c = car_traj_of_date[arrs["ev_date"], arrs["ev_tick"]]
    return dcls[arrs["ev_grid_loc"], np.where(c < 0, sent, c)].astype(np.int64)


def observed_counts(arrs: dict, dcls: np.ndarray, ci: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-date (k, h_ticks) by distance class for carrier ci: arrays (n_dates, N_CLS)."""
    grid = arrs["grid"]
    nd = grid.shape[1]
    h = np.stack([class_counts(grid[:, d], grid[ci, d], dcls, ci) for d in range(nd)])
    ecl = event_classes(arrs, dcls, grid[ci])
    keep = arrs["ev_train"] != ci
    k = np.zeros((nd, N_CLS), np.int64)
    np.add.at(k, (arrs["ev_date"][keep].astype(np.int64), ecl[keep]), 1)
    return k, h


def verify_export(export_dir: Path) -> None:
    arrs, meta = load_export(export_dir)
    assert np.array_equal(arrs["occupied"], arrs["grid"] >= 0)
    assert (arrs["ev_grid_loc"] == arrs["ev_loc"]).all(), "event position on grid != recorded location"
    dcls = class_table(arrs["loc_dist_halfhops"])
    k, h = observed_counts(arrs, dcls, meta["carrier_code"])
    s = stats_from_counts(k.sum(0), h.sum(0))
    print(f"recomputed from export: zone k={s['k_zone']:.0f}, h={s['h_zone']:,.0f}, rate={s['rate_zone']:.2f}/1000h;"
          f" far k={s['k_far']:.0f}, h={s['h_far']:,.0f}, rate={s['rate_far']:.4f}; RR={s['rr_zone']:.1f}; "
          f"d<=1.5 rate={s['rate_near_1p5']:.2f}")
    e = EXPECTED
    checks = {"k_zone": s["k_zone"] == e["k_zone"], "k_far": s["k_far"] == e["k_far"],
              "h_zone": abs(s["h_zone"] - e["h_zone"]) < 1, "h_far": abs(s["h_far"] - e["h_far"]) < 1,
              "rate_zone": round(float(s["rate_zone"]), 1) == e["rate_zone"],
              "rate_far": round(float(s["rate_far"]), 3) == e["rate_far"],
              "rr_zone": round(float(s["rr_zone"])) == e["rr_zone"],
              "rate_near_1p5": round(float(s["rate_near_1p5"]), 1) == e["rate_near_1p5"]}
    print("checks vs 03:", checks)
    assert all(checks.values()), "export does not reproduce the observed statistic"


# ================================================================================================
# stage: permute (cluster, numpy only)
# ================================================================================================
_G: dict = {}  # read-only globals shared with forked workers


def _shift_unit(d_victim: int) -> np.ndarray:
    """Scheme A unit: train-ticks by class on victim date d_victim against the carrier trajectory of every date."""
    grid, dcls, ci = _G["grid"], _G["dcls"], _G["ci"]
    return np.stack([class_counts(grid[:, d_victim], grid[ci, d_src], dcls, ci) for d_src in range(grid.shape[1])])


def _label_unit(c: int) -> np.ndarray:
    """Scheme B unit: train-ticks by class (summed over dates) with train c as the pseudo-carrier."""
    grid, dcls = _G["grid"], _G["dcls"]
    return np.sum([class_counts(grid[:, d], grid[c, d], dcls, c) for d in range(grid.shape[1])], axis=0)


def n_workers() -> int:
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        if os.environ.get(var):
            return max(1, int(os.environ[var]))
    return max(1, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)


def p_upper(null: np.ndarray, obs: float) -> float:
    null = np.asarray(null, dtype=np.float64)
    return float((1 + np.sum(null >= obs * (1 - 1e-12))) / (1 + len(null)))


def stage_permute(export_dir: Path, out_dir: Path, workers: int) -> None:
    import multiprocessing as mp

    t_start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    arrs, meta = load_export(export_dir)
    grid = arrs["grid"]
    n_tr, nd, nt = grid.shape
    ci = int(meta["carrier_code"])
    dcls = class_table(arrs["loc_dist_halfhops"])
    _G.update(grid=grid, dcls=dcls, ci=ci)
    ss = np.random.SeedSequence(SEED)
    rng_a, rng_b, rng_boot = (np.random.default_rng(s) for s in ss.spawn(3))
    timings = {}
    print(f"[{time.strftime('%H:%M:%S')}] loaded export grid {grid.shape}, carrier {meta['trains'][ci]}, "
          f"{workers} workers, numpy {np.__version__}", flush=True)

    # ---- observed ----
    k_obs_d, h_obs_d = observed_counts(arrs, dcls, ci)
    obs = stats_from_counts(k_obs_d.sum(0), h_obs_d.sum(0))
    print(f"observed: zone {obs['k_zone']:.0f}/{obs['h_zone']:,.0f} h = {obs['rate_zone']:.2f}/1000h, "
          f"RR {obs['rr_zone']:.1f}, d<=1.5 {obs['rate_near_1p5']:.2f}", flush=True)

    ctx = mp.get_context("fork")
    # ---- scheme A: whole-day circular shift of the carrier trajectory ----
    # Under shift k the carrier on victim date d follows its own trajectory of date (d + k) mod nd; every other
    # train, the events and the clock times are untouched. All 104 non-zero shifts are enumerated exactly
    # (H[d_victim, d_src, class]); the R_SHIFT replicates draw k ~ U{1..nd-1} from that exact table.
    t0 = time.time()
    with ctx.Pool(workers) as pool:
        H = np.stack(pool.map(_shift_unit, range(nd), chunksize=1))  # (nd victim, nd src, N_CLS)
    sent = dcls.shape[0] - 1
    keep = arrs["ev_train"] != ci
    ev_d, ev_t, ev_l = (arrs[x][keep].astype(np.int64) for x in ("ev_date", "ev_tick", "ev_grid_loc"))
    shifts = np.arange(nd)
    k_shift = np.zeros((nd, N_CLS), np.int64)
    h_shift = np.zeros((nd, N_CLS), np.int64)
    for s in shifts:
        src = (np.arange(nd) + s) % nd
        h_shift[s] = H[np.arange(nd), src].sum(0)
        c = grid[ci][(ev_d + s) % nd, ev_t]
        np.add.at(k_shift[s], dcls[ev_l, np.where(c < 0, sent, c)].astype(np.int64), 1)
    st_shift = stats_from_counts(k_shift, h_shift)
    assert st_shift["k_zone"][0] == obs["k_zone"] and np.allclose(st_shift["h_zone"][0], obs["h_zone"])
    k_draw_a = rng_a.integers(1, nd, size=R_SHIFT)
    timings["scheme_A_s"] = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] scheme A done in {timings['scheme_A_s']:.1f}s", flush=True)

    # ---- scheme B: carrier label permutation ----
    # The pseudo-carrier c is a random train other than T3420; its own ticks and its own events are excluded from
    # the risk set (as for the real carrier). All 139 alternatives are enumerated exactly; R_LABEL draws from them.
    t0 = time.time()
    with ctx.Pool(workers) as pool:
        h_label = np.stack(pool.map(_label_unit, range(n_tr), chunksize=1))  # (n_tr, N_CLS)
    k_label = np.zeros((n_tr, N_CLS), np.int64)
    for c in range(n_tr):
        ecl = event_classes(arrs, dcls, grid[c])
        np.add.at(k_label[c], ecl[arrs["ev_train"] != c], 1)
    st_label = stats_from_counts(k_label, h_label)
    assert st_label["k_zone"][ci] == obs["k_zone"] and np.allclose(st_label["h_zone"][ci], obs["h_zone"])
    others = np.array([c for c in range(n_tr) if c != ci])
    c_draw_b = rng_b.choice(others, size=R_LABEL, replace=True)
    timings["scheme_B_s"] = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] scheme B done in {timings['scheme_B_s']:.1f}s", flush=True)

    # ---- cluster bootstrap over service dates ----
    t0 = time.time()
    w = rng_boot.multinomial(nd, np.full(nd, 1 / nd), size=B_BOOT).astype(np.float64)  # resampled date counts
    kb, hb = w @ k_obs_d, w @ h_obs_d  # (B, N_CLS)
    hb_h = hb * TICK_S / 3600
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_b = np.where(hb_h > 0, kb / hb_h * 1000, np.nan)
    st_boot = stats_from_counts(kb, hb)
    timings["bootstrap_s"] = time.time() - t0

    # ---- p-values and outputs ----
    stat_names = ["rate_zone", "rr_zone", "rate_near_1p5", "k_zone"]
    res_p = {}
    for name in stat_names:
        o = float(obs[name])
        res_p[name] = {
            "observed": o,
            "A_time_shift": {"p_mc": p_upper(st_shift[name][k_draw_a], o),
                             "p_exact_enum": p_upper(st_shift[name][1:], o),
                             "null_max": float(np.max(st_shift[name][1:])),
                             "null_median": float(np.median(st_shift[name][1:])),
                             "null_q99": float(np.percentile(st_shift[name][k_draw_a], 99))},
            "B_label_perm": {"p_mc": p_upper(st_label[name][c_draw_b], o),
                             "p_exact_enum": p_upper(st_label[name][others], o),
                             "null_max": float(np.max(st_label[name][others])),
                             "null_median": float(np.median(st_label[name][others])),
                             "null_q99": float(np.percentile(st_label[name][c_draw_b], 99))},
        }
    h_k = h_obs_d.sum(0) * TICK_S / 3600
    kernel = []
    for j, lab in enumerate(D_LABELS):
        lo, hi = np.nanpercentile(rate_b[:, j], [2.5, 97.5])
        kernel.append({"d_to_carrier": lab, "k": int(k_obs_d.sum(0)[j]), "h": float(h_k[j]),
                       "rate_per_1000h": float(k_obs_d.sum(0)[j] / h_k[j] * 1000), "boot_lo": float(lo),
                       "boot_hi": float(hi), "n_boot_defined": int(np.isfinite(rate_b[:, j]).sum())})
    summ_boot = {}
    for name in ["rate_zone", "rate_far", "rr_zone", "rate_near_1p5"]:
        v = st_boot[name]
        fin = np.isfinite(v)
        summ_boot[name] = {"observed": float(obs[name]), "lo": float(np.percentile(v[fin], 2.5)),
                           "hi": float(np.percentile(v[fin], 97.5)), "n_finite": int(fin.sum())}

    def save_csv(path: Path, header: list[str], rows) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                f.write(",".join(str(x) for x in r) + "\n")

    trains = meta["trains"]
    save_csv(out_dir / "null_A_time_shift_exact.csv", ["shift_days", "k_zone", "h_zone", "rate_zone", "rr_zone",
                                                        "rate_near_1p5"],
             [(s, int(st_shift["k_zone"][s]), round(float(st_shift["h_zone"][s]), 3),
               round(float(st_shift["rate_zone"][s]), 5), round(float(st_shift["rr_zone"][s]), 5),
               round(float(st_shift["rate_near_1p5"][s]), 5)) for s in shifts])
    save_csv(out_dir / "null_A_time_shift_draws.csv", ["rep", "shift_days", "rate_zone", "rr_zone", "rate_near_1p5"],
             [(r, int(s), round(float(st_shift["rate_zone"][s]), 5), round(float(st_shift["rr_zone"][s]), 5),
               round(float(st_shift["rate_near_1p5"][s]), 5)) for r, s in enumerate(k_draw_a)])
    save_csv(out_dir / "null_B_label_exact.csv", ["train_id", "is_carrier", "k_zone", "h_zone", "rate_zone",
                                                   "rr_zone", "rate_near_1p5"],
             [(trains[c], int(c == ci), int(st_label["k_zone"][c]), round(float(st_label["h_zone"][c]), 3),
               round(float(st_label["rate_zone"][c]), 5), round(float(st_label["rr_zone"][c]), 5),
               round(float(st_label["rate_near_1p5"][c]), 5)) for c in range(n_tr)])
    save_csv(out_dir / "null_B_label_draws.csv", ["rep", "train_id", "rate_zone", "rr_zone", "rate_near_1p5"],
             [(r, trains[c], round(float(st_label["rate_zone"][c]), 5), round(float(st_label["rr_zone"][c]), 5),
               round(float(st_label["rate_near_1p5"][c]), 5)) for r, c in enumerate(c_draw_b)])
    save_csv(out_dir / "bootstrap_kernel_ci.csv", list(kernel[0].keys()), [tuple(r.values()) for r in kernel])
    np.savez_compressed(out_dir / "bootstrap_draws.npz", rate_by_class=rate_b.astype(np.float32),
                        rate_zone=st_boot["rate_zone"], rr_zone=st_boot["rr_zone"],
                        rate_near_1p5=st_boot["rate_near_1p5"], d_labels=np.array(D_LABELS))
    timings["total_s"] = time.time() - t_start
    summary = {
        "seed": SEED, "rng": "numpy.random.SeedSequence(seed).spawn(3) -> PCG64 streams for A, B, bootstrap",
        "R_shift": R_SHIFT, "R_label": R_LABEL, "B_boot": B_BOOT, "n_dates": nd, "n_trains": n_tr,
        "carrier": trains[ci], "workers": workers, "timings_s": {k: round(v, 2) for k, v in timings.items()},
        "host": platform.node(), "python": sys.version.split()[0], "numpy": np.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "export_sha256": hashlib.sha256((export_dir / "perm_export.npz").read_bytes()).hexdigest(),
        "observed": {k: float(v) for k, v in obs.items()}, "p_values": res_p,
        "bootstrap_ci": summ_boot, "kernel": kernel,
        "n_distinct_null_values": {"A_time_shift": nd - 1, "B_label_perm": n_tr - 1},
    }
    (out_dir / "permutation_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for name in stat_names[:3]:
        r = res_p[name]
        print(f"{name}: obs {r['observed']:.2f} | A p_mc={r['A_time_shift']['p_mc']:.5f} exact="
              f"{r['A_time_shift']['p_exact_enum']:.5f} null max {r['A_time_shift']['null_max']:.2f} | B p_mc="
              f"{r['B_label_perm']['p_mc']:.5f} exact={r['B_label_perm']['p_exact_enum']:.5f} null max "
              f"{r['B_label_perm']['null_max']:.2f}")
    print("bootstrap:", {k: (round(v["lo"], 3), round(v["hi"], 3)) for k, v in summ_boot.items()})
    print(f"[{time.strftime('%H:%M:%S')}] done; timings {summary['timings_s']}", flush=True)


# ================================================================================================
# stage: figure (local)
# ================================================================================================
def stage_figure(out_dir: Path, fig_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = json.loads((out_dir / "permutation_summary.json").read_text(encoding="utf-8"))

    def read_col(path: Path, col: str) -> np.ndarray:
        lines = path.read_text(encoding="utf-8").splitlines()
        j = lines[0].split(",").index(col)
        return np.array([float(x.split(",")[j]) for x in lines[1:]])

    obs = s["observed"]["rate_zone"]
    a = read_col(out_dir / "null_A_time_shift_draws.csv", "rate_zone")
    b = read_col(out_dir / "null_B_label_draws.csv", "rate_zone")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for axi, null, title, key in [(ax[0], a, "A: carrier trajectory shifted by k whole days", "A_time_shift"),
                                  (ax[1], b, "B: random other train as pseudo-carrier", "B_label_perm")]:
        bins = np.linspace(0, max(null.max(), obs) * 1.05, 60)
        axi.hist(null, bins=bins, color="0.6", label=f"null (R = {len(null)})")
        axi.axvline(obs, color="C3", lw=2, label=f"observed T3420 = {obs:.1f}")
        p = s["p_values"]["rate_zone"][key]["p_mc"]
        axi.set_title(f"{title}\np = {p:.4f}", fontsize=9)
        axi.set_xlabel("breakdowns per 1000 train-h within 2.5 hops")
        axi.set_ylabel("replicates")
        axi.legend(fontsize=8)
    kern = s["kernel"]
    x = np.arange(len(kern))
    r = np.array([k["rate_per_1000h"] for k in kern])
    lo = np.array([k["boot_lo"] for k in kern])
    hi = np.array([k["boot_hi"] for k in kern])
    pos = r > 0
    ax[2].errorbar(x[pos], r[pos], yerr=[r[pos] - np.maximum(lo[pos], 1e-3), hi[pos] - r[pos]], fmt="o", color="C0",
                   capsize=3)
    ax[2].set_yscale("log")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels([k["d_to_carrier"] for k in kern], rotation=30)
    ax[2].set_xlabel("network-hop distance to T3420")
    ax[2].set_ylabel("breakdowns per 1000 train-h")
    ax[2].set_title(f"distance kernel, 95% date-cluster bootstrap CI (B = {s['B_boot']})", fontsize=9)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"figure written: {fig_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=["export", "permute", "figure"])
    ap.add_argument("--export-dir", type=Path, default=ROOT / "data_derived" / "perm_export")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out" / "permutation")
    ap.add_argument("--fig", type=Path, default=ROOT / "figures" / "mechanism" / "fig_permutation.png")
    ap.add_argument("--workers", type=int, default=0, help="worker processes (0 = from SLURM / affinity)")
    a = ap.parse_args()
    if a.stage == "export":
        stage_export(a.export_dir)
    elif a.stage == "permute":
        stage_permute(a.export_dir, a.out_dir, a.workers or n_workers())
    else:
        stage_figure(a.out_dir, a.fig)


if __name__ == "__main__":
    main()
