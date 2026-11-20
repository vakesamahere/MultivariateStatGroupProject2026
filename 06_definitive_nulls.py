"""06 Definitive-scale negative controls of the T3420 carrier proximity effect (Null A and Null B of 03 section 4).

Stages
  --stage permute  (cluster or local; numpy only) Null A event-position permutation (R draws) and Null B within-day
                   circular shift of the carrier trajectory (exact enumeration of every shift + S random shifts).
  --stage figure   (local; matplotlib) figures/mechanism/fig5_null_tests.png from out/definitive/ (same layout as
                   the figure code in src/03_mechanism.py).

Input: the compact export of src/05_permutation.py --stage export (data_derived/perm_export/perm_export.{npz,json};
integer codes only, no pandas objects). That export uses a 2,220-tick grid starting at 05:30; src/03_mechanism.py
uses a grid starting at the floor hour of the first exposure (05:00) with NT = ceil((t_end - t_start)/30 s) + 1 =
2,222 ticks. The export grid is mapped onto 03's grid by a constant offset of 60 ticks (05:00 -> 05:30), which is
exact for every in-service cell and event; only the circular shift of Null B depends on NT, and 03's NT is used.

Statistic (03 section 3/4): for a candidate carrier c, R_c = k_c / h_c x 1000, where k_c = breakdowns of trains
other than c whose onset position is within NEAR_D = 1.5 network hops of c's position at the onset tick, and h_c =
train-hours other trains spend within 1.5 hops of c (30-s grid). Observed R_T3420 = 45.177 (115 / 2,545.5 h).

Null A (event-position permutation). Daily event counts fixed; each event's (train, tick) drawn independently and
uniformly from the in-service (train, tick) cells of its own date. k_c recomputed for all 140 trains per draw;
statistics: R_T3420 and the max-scan statistic max_c R_c (h_c fixed). Draws are generated in fixed chunks, each with
its own SeedSequence child stream, so results do not depend on the number of workers.

Null B (within-day circular shift). Events fixed; the carrier's position at tick t on every date is replaced by its
position at tick (t + delta) mod NT, one delta for all dates; k and h recomputed exactly. delta = 0 is the unit
check. The admissible set delta in [60, NT - 60) (30 min .. 22.9 h) has only NT - 120 = 2,102 elements, so the null
is enumerated exactly (exact p) and S random shifts are drawn from it with replacement (Monte Carlo p, as specified).

p = (1 + #{null >= observed}) / (1 + R).

Run:  python3 src/06_definitive_nulls.py --stage permute --perm 20000 --shifts 4000 --workers 64
      python3 src/06_definitive_nulls.py --stage figure
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TICK_S = 30
CARRIER = "T3420"
SEED = 20260306
NEAR_HALFHOPS = 3  # NEAR_D = 1.5 hops, distances stored in half-hops
GRID_PAD = 60  # export tick 0 = 05:30; 03's tick 0 = 05:00
NT_03 = 2222  # 03's grid length (mechanism_results_perm.txt: "grid (140, 105, 2222) ... from 5.0h")
SHIFT_MIN_TICKS = 60  # 30 min
CHUNK = 250  # Null A draws per work unit
EXPECTED = {"k": 115, "h": 2545.5, "rate": 45.177}

_G: dict = {}  # read-only data shared with forked workers


# ================================================================================================
# data
# ================================================================================================
def load(export_dir: Path) -> dict:
    z = np.load(export_dir / "perm_export.npz", allow_pickle=False)
    meta = json.loads((export_dir / "perm_export.json").read_text(encoding="utf-8"))
    grid = z["grid"]
    n_tr, n_d, nt_exp = grid.shape
    occ_ticks = np.flatnonzero((grid >= 0).any(axis=(0, 1)))
    # 03 grid: tick0 = floor-hour of first exposure (05:00), NT = last exposure tick + 1 + 1
    assert meta["grid"]["tick0_clock"] == "05:30:00" and occ_ticks.min() == 0, "export grid does not start at 05:30"
    assert occ_ticks.max() + GRID_PAD + 2 == NT_03, (occ_ticks.max(), "grid end inconsistent with 03's NT")
    nl = z["loc_dist_halfhops"].shape[0]
    g = np.full((n_tr, n_d, NT_03), nl, dtype=np.int16)  # sentinel nl = not in service
    keep = min(nt_exp, NT_03 - GRID_PAD)
    assert (grid[:, :, keep:] < 0).all()
    gg = grid[:, :, :keep]
    g[:, :, GRID_PAD:GRID_PAD + keep] = np.where(gg < 0, nl, gg)
    near = np.zeros((nl + 1, nl + 1), dtype=bool)  # near[a, b]: positions a, b within 1.5 hops; sentinel -> False
    near[:nl, :nl] = z["loc_dist_halfhops"] <= NEAR_HALFHOPS
    ev_tick = z["ev_tick"].astype(np.int64) + GRID_PAD
    ev_train, ev_date = z["ev_train"].astype(np.int64), z["ev_date"].astype(np.int64)
    assert (g[ev_train, ev_date, ev_tick] == z["ev_loc"]).all(), "event position on grid != recorded location"
    return {"g": g, "near": near, "nl": nl, "ev_train": ev_train, "ev_date": ev_date, "ev_tick": ev_tick,
            "ci": int(meta["carrier_code"]), "trains": meta["trains"], "dates": meta["dates"]}


def near_counts(g_d: np.ndarray, near: np.ndarray, nl: int, exclude: int | None = None) -> np.ndarray:
    """(nl + 1, NT) float: number of in-service trains (optionally excluding one) within 1.5 hops of position a."""
    nt = g_d.shape[1]
    m = g_d < nl
    if exclude is not None:
        m = m.copy()
        m[exclude] = False
    flat = g_d[m].astype(np.int64) * nt + np.broadcast_to(np.arange(nt), g_d.shape)[m]
    occ = np.bincount(flat, minlength=nl * nt).reshape(nl, nt).astype(np.float64)
    return near[:, :nl].astype(np.float64) @ occ


def exposure_all(D: dict) -> np.ndarray:
    """h_c (train-h other trains spend within 1.5 hops of c) for every train c."""
    g, near, nl = D["g"], D["near"], D["nl"]
    n_tr, n_d, nt = g.shape
    h = np.zeros(n_tr)
    ticks = np.arange(nt)
    for d in range(n_d):
        nc = near_counts(g[:, d], near, nl)  # includes c itself where c is in service
        gd = g[:, d]
        on = gd < nl
        v = nc[gd, ticks[None, :]]  # (n_tr, nt)
        h += np.where(on, v - 1.0, 0.0).sum(axis=1)  # subtract c's own presence (near itself, d = 0)
    return h * TICK_S / 3600


def observed_k(D: dict) -> np.ndarray:
    """k_c for every candidate c from the observed events."""
    g, near = D["g"], D["near"]
    pc = g[:, D["ev_date"], D["ev_tick"]]  # (n_tr, n_ev) candidate position at each onset
    pv = g[D["ev_train"], D["ev_date"], D["ev_tick"]]  # (n_ev,)
    hit = near[pc, pv[None, :]]
    hit[D["ev_train"], np.arange(len(pv))] = False  # own breakdowns excluded
    return hit.sum(axis=1).astype(np.float64)


# ================================================================================================
# Null A: event-position permutation
# ================================================================================================
def _null_a_chunk(args: tuple[int, int, np.random.SeedSequence]) -> np.ndarray:
    """k_c (n_tr, reps) for one chunk of permutation draws."""
    _, reps, ss = args
    g, near, cells, n_by_date = _G["g"], _G["near"], _G["cells"], _G["n_by_date"]
    rng = np.random.default_rng(ss)
    n_tr, _, nt = g.shape
    k = np.zeros((n_tr, reps), dtype=np.int64)
    rows = np.arange(n_tr)[:, None]
    for d, n in n_by_date:
        cd = cells[d]
        pick = cd[rng.integers(0, len(cd), size=reps * n)]  # uniform over in-service (train, tick) of date d
        vt, tk = pick // nt, pick % nt
        gd = g[:, d]
        pc = gd[:, tk]  # (n_tr, reps*n) candidate positions
        pv = gd[vt, tk]  # victim positions
        hit = near[pc, pv[None, :]]
        hit[vt, np.arange(len(vt))] = False  # the drawn victim is not counted for itself as candidate
        k += hit.reshape(n_tr, reps, n).sum(axis=2)
    return k


def run_null_a(D: dict, n_perm: int, seed_seq: np.random.SeedSequence, workers: int) -> np.ndarray:
    g, nl = D["g"], D["nl"]
    n_tr, n_d, nt = g.shape
    cells = [np.flatnonzero((g[:, d] < nl).ravel()) for d in range(n_d)]  # flat index train*nt + tick
    dates, counts = np.unique(D["ev_date"], return_counts=True)
    _G.update(g=g, near=D["near"], cells=cells, n_by_date=list(zip(dates.tolist(), counts.tolist())))
    n_chunks = -(-n_perm // CHUNK)
    children = seed_seq.spawn(n_chunks)
    units = [(i, min(CHUNK, n_perm - i * CHUNK), children[i]) for i in range(n_chunks)]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            parts = pool.map(_null_a_chunk, units)
    else:
        parts = [_null_a_chunk(u) for u in units]
    return np.concatenate(parts, axis=1)


# ================================================================================================
# Null B: within-day circular shift of the carrier trajectory
# ================================================================================================
def run_null_b(D: dict) -> tuple[np.ndarray, np.ndarray]:
    """Exact (k, h) for every shift delta = 0..NT-1."""
    g, near, nl, ci = D["g"], D["near"], D["nl"], D["ci"]
    n_tr, n_d, nt = g.shape
    deltas = np.arange(nt)
    ticks = np.arange(nt)
    shifted = (ticks[None, :] + deltas[:, None]) % nt  # (n_delta, nt)
    h = np.zeros(nt)
    for d in range(n_d):
        nc = near_counts(g[:, d], near, nl, exclude=ci)  # other trains near position a at tick t
        cp = g[ci, d][shifted]  # (n_delta, nt) shifted carrier position (sentinel row of nc is 0)
        h += nc[cp, ticks[None, :]].sum(axis=1)
    h *= TICK_S / 3600
    keep = D["ev_train"] != ci
    ed, et, ev = D["ev_date"][keep], D["ev_tick"][keep], D["ev_train"][keep]
    pv = g[ev, ed, et]
    cp_ev = g[ci, ed[:, None], (et[:, None] + deltas[None, :]) % nt]  # (n_ev, n_delta)
    k = near[cp_ev, pv[:, None]].sum(axis=0).astype(np.float64)
    return k, h


# ================================================================================================
# stage: permute
# ================================================================================================
def p_upper(null: np.ndarray, obs: float) -> float:
    null = np.asarray(null, dtype=np.float64)
    return float((1 + np.sum(null >= obs * (1 - 1e-12))) / (1 + len(null)))


def summ(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {"mean": float(x.mean()), "median": float(np.median(x)), "p97_5": float(np.percentile(x, 97.5)),
            "max": float(x.max())}


def default_workers() -> int:
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        if os.environ.get(var):
            return max(1, int(os.environ[var]))
    return 1


def stage_permute(export_dir: Path, out_dir: Path, n_perm: int, n_shift: int, seed: int, workers: int) -> None:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    D = load(export_dir)
    ci, trains = D["ci"], D["trains"]
    assert trains[ci] == CARRIER
    t_load = time.time() - t0

    # ---- observed statistic and unit checks ----
    h_c = exposure_all(D)
    k_c = observed_k(D)
    rate_c = k_c / h_c * 1000
    obs_carrier, obs_max = float(rate_c[ci]), float(rate_c.max())
    assert int(np.argmax(rate_c)) == ci and (h_c > 0).all()
    unit = {"k": float(k_c[ci]), "h": float(h_c[ci]), "rate": obs_carrier}
    assert unit["k"] == EXPECTED["k"] and round(unit["h"], 1) == EXPECTED["h"] \
        and round(obs_carrier, 3) == EXPECTED["rate"], unit
    order = np.argsort(-rate_c)
    t_obs = time.time() - t0
    print(f"observed: {CARRIER} k={k_c[ci]:.0f} h={h_c[ci]:,.1f} rate={obs_carrier:.3f}; runner-up "
          f"{trains[order[1]]} {rate_c[order[1]]:.3f} [{t_obs:.1f}s]", flush=True)

    ss = np.random.SeedSequence(seed)
    ss_a, ss_b = ss.spawn(2)

    # ---- Null B (exact enumeration + S random shifts) ----
    tb = time.time()
    k_b, h_b = run_null_b(D)
    nt = len(k_b)
    rate_b = k_b / np.maximum(h_b, 1e-9) * 1000
    assert k_b[0] == unit["k"] and abs(h_b[0] - unit["h"]) < 1e-6, (k_b[0], h_b[0])  # delta = 0 unit check
    adm = np.arange(SHIFT_MIN_TICKS, nt - SHIFT_MIN_TICKS)
    rng_b = np.random.default_rng(ss_b)
    draw_b = rng_b.choice(adm, size=n_shift, replace=True)
    t_b = time.time() - tb
    print(f"Null B: delta=0 check k={k_b[0]:.0f} h={h_b[0]:,.1f} rate={rate_b[0]:.3f}; {len(adm)} admissible shifts, "
          f"exact max {rate_b[adm].max():.2f}; {n_shift} draws [{t_b:.1f}s]", flush=True)

    # ---- Null A ----
    ta = time.time()
    k_a = run_null_a(D, n_perm, ss_a, workers)
    rate_a = k_a / h_c[:, None] * 1000
    car_a, max_a = rate_a[ci], rate_a.max(axis=0)
    t_a = time.time() - ta
    print(f"Null A: {n_perm} draws, carrier null max {car_a.max():.2f}, max-scan null max {max_a.max():.2f} "
          f"[{t_a:.1f}s, {workers} workers]", flush=True)

    # ---- outputs ----
    np.savez_compressed(out_dir / "null_A_draws.npz", k_carrier=k_a[ci].astype(np.int16),
                        max_rate=max_a.astype(np.float32), argmax_train=rate_a.argmax(axis=0).astype(np.int16))
    with open(out_dir / "null_A_draws.csv", "w") as f:
        f.write("draw,k_carrier,rate_carrier,max_rate,argmax_train\n")
        am = rate_a.argmax(axis=0)
        for i in range(n_perm):
            f.write(f"{i},{int(k_a[ci, i])},{car_a[i]:.6f},{max_a[i]:.6f},{trains[am[i]]}\n")
    with open(out_dir / "null_B_shift_exact.csv", "w") as f:
        f.write("shift_ticks,shift_min,k,h,rate_per_1000h,admissible\n")
        for dl in range(nt):
            a = int(SHIFT_MIN_TICKS <= dl < nt - SHIFT_MIN_TICKS)
            f.write(f"{dl},{dl * TICK_S / 60:.1f},{k_b[dl]:.0f},{h_b[dl]:.4f},{rate_b[dl]:.6f},{a}\n")
    np.savez_compressed(out_dir / "null_B_shift_draws.npz", shift_ticks=draw_b.astype(np.int16))

    rb_draw, rb_exact = rate_b[draw_b], rate_b[adm]
    summary = {
        "script": "src/06_definitive_nulls.py --stage permute",
        "seed": seed, "seed_scheme": "SeedSequence(seed).spawn(2) -> (Null A, Null B); Null A further spawned into "
                                     f"{-(-n_perm // CHUNK)} chunks of {CHUNK} draws (worker-count independent)",
        "statistic": "R_c = k_c / h_c x 1000; k_c = breakdowns of trains != c within 1.5 hops of c at onset tick; "
                     "h_c = train-h other trains spend within 1.5 hops of c (30-s grid, 03 definitions)",
        "grid": {"shape": list(D["g"].shape), "tick0": "05:00", "tick_s": TICK_S,
                 "note": "05 export (05:30, 2220 ticks) padded by 60 leading ticks to 03's grid (NT=2222)"},
        "observed": {"carrier": CARRIER, "k": unit["k"], "h_train_h": unit["h"], "rate": obs_carrier,
                     "max_scan": obs_max, "max_scan_train": trains[int(np.argmax(rate_c))],
                     "runner_up": trains[order[1]], "runner_up_rate": float(rate_c[order[1]]),
                     "median_rate_all_trains": float(np.median(rate_c))},
        "null_A_event_position_permutation": {
            "replicates": n_perm,
            "carrier_k": summ(k_a[ci]), "carrier_rate": summ(car_a),
            "p_carrier": p_upper(car_a, obs_carrier), "n_ge_carrier": int(np.sum(car_a >= obs_carrier)),
            "max_scan_rate": summ(max_a), "p_max_scan": p_upper(max_a, obs_max),
            "n_ge_max_scan": int(np.sum(max_a >= obs_max)),
            "draws_any_train_ge_observed": int(np.sum(max_a >= obs_carrier)),
            "mean_trains_per_draw_ge_10": float((rate_a >= 10).sum(axis=0).mean()),
            "p_floor": 1 / (n_perm + 1)},
        "null_B_circular_shift": {
            "delta_range_ticks": [SHIFT_MIN_TICKS, nt - SHIFT_MIN_TICKS], "n_admissible": int(len(adm)),
            "unit_check_delta0": {"k": float(k_b[0]), "h_train_h": float(h_b[0]), "rate": float(rate_b[0])},
            "replicates": n_shift, "sampling": "uniform with replacement from the admissible set",
            "rate_draws": summ(rb_draw), "p_mc": p_upper(rb_draw, obs_carrier),
            "n_ge_mc": int(np.sum(rb_draw >= obs_carrier)), "p_floor_mc": 1 / (n_shift + 1),
            "rate_exact": summ(rb_exact), "p_exact": p_upper(rb_exact, obs_carrier),
            "n_ge_exact": int(np.sum(rb_exact >= obs_carrier)), "p_floor_exact": 1 / (len(adm) + 1),
            "top_exact_shifts": [{"shift_min": float(adm[i] * TICK_S / 60), "k": float(k_b[adm[i]]),
                                  "h": float(h_b[adm[i]]), "rate": float(rb_exact[i])}
                                 for i in np.argsort(-rb_exact)[:5]]},
        "runtime_s": {"load": round(t_load, 2), "observed": round(t_obs - t_load, 2), "null_B": round(t_b, 2),
                      "null_A": round(t_a, 2), "total": round(time.time() - t0, 2)},
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "host": platform.node(),
                        "workers": workers, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
                        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST")},
    }
    (out_dir / "definitive_nulls_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    a, b = summary["null_A_event_position_permutation"], summary["null_B_circular_shift"]
    print(f"Null A carrier: null median {a['carrier_rate']['median']:.2f}, 97.5% {a['carrier_rate']['p97_5']:.2f}, "
          f"max {a['carrier_rate']['max']:.2f}; p={a['p_carrier']:.3g}")
    print(f"Null A max-scan: median {a['max_scan_rate']['median']:.2f}, 97.5% {a['max_scan_rate']['p97_5']:.2f}, "
          f"max {a['max_scan_rate']['max']:.2f}; p={a['p_max_scan']:.3g}")
    print(f"Null B: draws median {b['rate_draws']['median']:.2f}, 97.5% {b['rate_draws']['p97_5']:.2f}, max "
          f"{b['rate_draws']['max']:.2f}; p_mc={b['p_mc']:.3g}; exact max {b['rate_exact']['max']:.2f}, "
          f"p_exact={b['p_exact']:.3g}")
    print(f"total {time.time() - t0:.1f}s -> {out_dir}")


# ================================================================================================
# stage: figure (same layout as src/03_mechanism.py F5)
# ================================================================================================
def stage_figure(out_dir: Path, fig_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = json.loads((out_dir / "definitive_nulls_summary.json").read_text(encoding="utf-8"))
    a, b = s["null_A_event_position_permutation"], s["null_B_circular_shift"]
    max_a = np.load(out_dir / "null_A_draws.npz")["max_rate"]
    ex = np.genfromtxt(out_dir / "null_B_shift_exact.csv", delimiter=",", names=True)
    rate_sh = ex["rate_per_1000h"][ex["admissible"] == 1]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    axes[0].hist(max_a, bins=40, color="0.7")
    axes[0].axvline(s["observed"]["max_scan"], color="C3", lw=2)
    axes[0].set_xlabel("best of 140 trains: breakdowns per 1000 h-near")
    axes[0].set_ylabel("permutations")
    axes[0].set_title(f"event-position permutation (max-scan, R={a['replicates']:,}), p={a['p_max_scan']:.1g}",
                      fontsize=10)
    axes[1].hist(rate_sh, bins=40, color="0.7")
    axes[1].axvline(s["observed"]["rate"], color="C3", lw=2)
    axes[1].set_xlabel(f"breakdowns per 1000 h-near of shifted {CARRIER}")
    axes[1].set_ylabel("shifts")
    axes[1].set_title(f"circular time shift, all {b['n_admissible']:,} shifts, exact p={b['p_exact']:.1g}",
                      fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    print(f"wrote {fig_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", choices=("permute", "figure"), required=True)
    p.add_argument("--export-dir", type=Path, default=ROOT / "data_derived" / "perm_export")
    p.add_argument("--out-dir", type=Path, default=ROOT / "out" / "definitive")
    p.add_argument("--perm", type=int, default=20000)
    p.add_argument("--shifts", type=int, default=4000)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--fig", type=Path, default=ROOT / "figures" / "mechanism" / "fig5_null_tests.png")
    a = p.parse_args()
    if a.stage == "permute":
        stage_permute(a.export_dir, a.out_dir, a.perm, a.shifts, a.seed, a.workers or default_workers())
    else:
        stage_figure(a.out_dir, a.fig)


if __name__ == "__main__":
    main()
