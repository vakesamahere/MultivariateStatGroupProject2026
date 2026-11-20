"""07 Sensitivity analyses for the confirmatory hazard model and the carrier permutation null.

Three reviewer-driven checks of the reported results; nothing here changes a reported number, it only reports
how far the headline quantities move under alternative specifications.

 A. DATE FIXED EFFECT.  Primary Poisson hazard model (train x date x hour x distance-to-carrier cells, log
    train-hour offset, date-clustered SE) refitted with C(date) added. With 105 date dummies and clustering by
    date this is a within-date estimator: only trains running on the same date are compared.
 B. ROUTE FIXED EFFECT.  Same model with C(route_id) (and C(date) + C(route_id) together).
 C. BURST-BLOCK PERMUTATION.  Null A of 03/06 moves every event independently, which destroys the within-day
    burst structure and therefore tests independent placement. Here each of the 23 bursts is moved as ONE unit:
    all of its events are shifted by the same uniformly drawn admissible tick offset on their own date, so the
    internal time offsets, the train composition (and hence the line mix) and the daily event count are preserved.
    The full 140-train max-scan is rescored exactly as in 06_definitive_nulls.py.

Why this file rebuilds 03's objects instead of importing them: src/03_mechanism.py runs its whole pipeline at
import time (~2 min, ~8.5 GB), so it cannot be imported. Everything needed is re-derived from the compact integer
export data_derived/perm_export/perm_export.{npz,json} that src/05_permutation.py writes, plus
data_derived/train_day_cleaning.csv; check_reproduction() asserts that the re-derivation reproduces 03's reported
numbers (140 events, 168,756 train-h, 614,900 cells, d0 RR 1899.724, product switch RR 1.788) and 06's observed
statistic (115 events / 2,545.5 train-h / 45.177 per 1000 h) before any sensitivity analysis is run.

Outputs: out/sensitivity/{sens_summary.json, sens_model_date.csv, sens_model_route.csv, sens_model_compare.csv,
         sens_burst_null.npz, sens_burst_null_summary.csv}. Interpretation: notes/03d_sensitivity.md.

Run: python3 src/07_sensitivity.py                (models + 5,000 burst draws)
     python3 src/07_sensitivity.py --draws 2000
     python3 src/07_sensitivity.py --skip null    (models only)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DERIVED = ROOT / "data_derived"
EXPORT = DERIVED / "perm_export"
OUT = ROOT / "out" / "sensitivity"
CARRIER = "T3420"
TICK_S = 30
GRID_PAD = 60  # export tick 0 = 05:30; 03's tick 0 = 05:00
NT = 2222  # 03's grid length (mechanism_results_perm.txt: "grid (140, 105, 2222) ... from 5.0h")
NEAR_HALFHOPS = 3  # NEAR_D = 1.5 hops, distances stored in half-hops
SHIFT_MIN_TICKS = 60  # 30 min: admissible burst shifts exclude near-identity placements (Null B convention)
SEED = 20260307
EXPECTED = {"k": 115, "h": 2545.5, "rate": 45.177, "events_model": 140, "train_h": 168756, "cells": 614900,
            "rr_d0": 1899.724, "rr_switch": 1.788, "lr_distance": 970.4}
DCAT = np.array(["d0", "d0p5", "d1", "d1p5", "d2", "d2p5", "d3plus", "off"])
OFF = 7
HOUR_TICK = ((18000 + np.arange(NT) * TICK_S) // 3600).astype(np.int64)  # 03: t_start_s = 05:00


def log(*parts) -> None:
    print(" ".join(str(p) for p in parts), flush=True)


# ================================================================================================
# A. grid, exposure and events, re-derived from the compact export
# ================================================================================================
def load_export() -> dict:
    """03's position grid (140 train x 105 date x 2222 tick), hop distances, events and the carrier row."""
    z = np.load(EXPORT / "perm_export.npz", allow_pickle=False)
    meta = json.loads((EXPORT / "perm_export.json").read_text(encoding="utf-8"))
    g_raw = z["grid"]
    n_tr, n_d, nt_raw = g_raw.shape
    nl = int(z["loc_dist_halfhops"].shape[0])
    g = np.full((n_tr, n_d, NT), nl, dtype=np.int16)  # sentinel nl = not in service
    keep = min(nt_raw, NT - GRID_PAD)  # the export carries a few trailing all-off ticks; 03's grid is 1 tick longer
    assert (g_raw[:, :, keep:] < 0).all(), "export grid has in-service ticks past 03's grid end"
    g[:, :, GRID_PAD:GRID_PAD + keep] = np.where(g_raw[:, :, :keep] < 0, nl, g_raw[:, :, :keep])
    ci = int(meta["carrier_code"])
    assert meta["trains"][ci] == CARRIER
    ev = {"train": z["ev_train"].astype(np.int64), "date": z["ev_date"].astype(np.int64),
          "tick": z["ev_tick"].astype(np.int64) + GRID_PAD, "loc": z["ev_loc"].astype(np.int64)}
    assert (g[ev["train"], ev["date"], ev["tick"]] == ev["loc"]).all(), "event position on grid != recorded location"
    ev["tick_hour"] = HOUR_TICK[ev["tick"]]  # 30-s ticks never straddle an hour, so this is the onset hour
    half = z["loc_dist_halfhops"].astype(np.int16)
    near = np.zeros((nl + 1, nl + 1), dtype=bool)
    near[:nl, :nl] = half <= NEAR_HALFHOPS
    return {"g": g, "nl": nl, "ci": ci, "half": half, "near": near, "ev": ev,
            "trains": meta["trains"], "dates": pd.to_datetime(meta["dates"])}


def dcode_grid(D: dict) -> np.ndarray:
    """Distance-to-carrier category (03's d_code: half-hops rounded and capped at 6, 7 = carrier off)."""
    g, nl, ci, half = D["g"], D["nl"], D["ci"], D["half"]
    out = np.full(g.shape, OFF, dtype=np.int8)
    for di in range(g.shape[1]):
        cp = g[ci, di]  # (NT,) carrier position
        pos = g[:, di]  # (140, NT)
        ok = (cp[None, :] < nl) & (pos < nl)
        h = np.minimum(half[np.where(pos < nl, pos, 0), np.where(cp < nl, cp, 0)[None, :]], 6)
        out[:, di] = np.where(ok, h, OFF)
    return out


def exposure_counts(D: dict, dcode: np.ndarray, nlev: int = 8) -> np.ndarray:
    """train-hours by (train, date, hour, distance code); the carrier's own ticks are excluded."""
    g, nl, ci = D["g"], D["nl"], D["ci"]
    out = np.zeros((g.shape[0], g.shape[1], 24, nlev), np.int64)
    for di in range(g.shape[1]):
        svc = g[:, di, :] < nl
        svc[ci] = False
        key = (np.arange(g.shape[0])[:, None] * 24 + HOUR_TICK[None, :]) * nlev + dcode[:, di, :]
        out[:, di] = np.bincount(key[svc], minlength=g.shape[0] * 24 * nlev).reshape(g.shape[0], 24, nlev)
    return out


def event_counts(D: dict, code_ev: np.ndarray, nlev: int = 8) -> np.ndarray:
    ev, g = D["ev"], D["g"]
    key = ((ev["train"] * g.shape[1] + ev["date"]) * 24 + ev["tick_hour"]) * nlev + code_ev
    return np.bincount(key, minlength=g.shape[0] * g.shape[1] * 24 * nlev).reshape(g.shape[0], g.shape[1], 24, nlev)


def load_bursts(D: dict) -> pd.DataFrame:
    """Per-event burst id (from 02's screening table) aligned to the export event order, plus a burst index."""
    scr = pd.read_csv(ROOT / "figures" / "screening" / "tables" / "breakdown_events.csv",
                      encoding="utf-8-sig", parse_dates=["onset"])
    bd = pd.read_csv(DERIVED / "breakdowns.csv", parse_dates=["onset_ts", "date"])
    ev = scr.set_index(["train_id", "onset"])
    out = bd[["train_id", "onset_ts"]].rename(columns={"onset_ts": "onset"})
    out["burst_id"] = out.set_index(["train_id", "onset"]).index.map(ev.burst_id)
    out["in_burst"] = out.set_index(["train_id", "onset"]).index.map(ev.in_burst)
    assert out.burst_id.notna().all() and len(out) == len(D["ev"]["train"])
    # align to the export event order: same (train, date, tick) triple, so verify by position
    tick = np.floor(((out.onset - out.onset.dt.normalize()).dt.total_seconds().values - 18000) / TICK_S).astype(int)
    di = D["dates"].get_indexer(pd.DatetimeIndex(out.onset.dt.normalize()))
    ti = pd.Index(D["trains"]).get_indexer(out.train_id)
    assert (ti == D["ev"]["train"]).all() and (di == D["ev"]["date"]).all() and (tick == D["ev"]["tick"]).all(), \
        "export event order does not match data_derived/breakdowns.csv"
    out["di"], out["ti"], out["tick"] = di, ti, tick
    out["in_burst"] = out.in_burst.fillna(False).astype(bool)
    return out


# ================================================================================================
# B. hazard-model cells: 03's cells_frame, with route/line columns added
# ================================================================================================
TD_COLS = ["clean_product", "prev_product", "clean_team", "product_switch", "is_deep", "hours_since_any_clean",
           "temp_c", "rain_mm", "train_type", "batch_number", "age_years"]
BASE_TERMS = ["product_switch", "C(clean_product)", "C(prev_product)", "C(clean_team)", "is_deep", "age5",
              "C(train_type)", "C(batch_number)", "temp10", "rain10", "C(hour)"]


def load_train_day() -> pd.DataFrame:
    """03's section-5 train-day panel (product switch = clean product differs from the previous service day)."""
    cl = pd.read_csv(DERIVED / "train_day_cleaning.csv", parse_dates=["date"]).sort_values(["train_id", "date"])
    g = cl.groupby("train_id")
    cl["prev_product"] = g.last_clean_product.shift(1)
    cl["prev_team"] = g.last_clean_team.shift(1)
    cl["product_switch"] = (cl.last_clean_product != cl.prev_product).astype(float)
    cl.loc[cl.prev_product.isna(), "product_switch"] = np.nan
    cl["is_deep"] = (cl.last_clean_type == "DEEP").astype(int)
    cl = cl.rename(columns={"last_clean_product": "clean_product", "last_clean_team": "clean_team",
                            "last_clean_type": "clean_type"})
    wx = pd.read_csv(DATA / "temperature.csv", encoding="utf-8-sig", parse_dates=["date"]).merge(
        pd.read_csv(DATA / "rainfall.csv", encoding="utf-8-sig", parse_dates=["date"]), on="date", how="outer")
    wx = wx.rename(columns={"mean_temperature_c": "temp_c", "rainfall_mm": "rain_mm"})
    return cl.merge(wx, on="date", how="left")


def cells_frame(D: dict, exp: np.ndarray, kk: np.ndarray, cats: np.ndarray) -> pd.DataFrame:
    """One row per (train, date, hour, distance) cell with exposure; drops rows without a cleaning record."""
    trains = pd.read_csv(DATA / "train_information.csv", encoding="utf-8-sig")
    trains["service_start_date"] = pd.to_datetime(trains.service_start_date)
    trains["age_years"] = (pd.Timestamp("2026-03-02") - trains.service_start_date).dt.days / 365.25
    td = (load_train_day().set_index(["train_id", "date"])
          .reindex(pd.MultiIndex.from_product([D["trains"], D["dates"]], names=["train_id", "date"])).reset_index()
          .merge(trains, on="train_id", how="left"))
    idx = np.flatnonzero(exp.ravel() > 0)
    ti_, di_, hh_, dc_ = np.unravel_index(idx, exp.shape)
    df = pd.DataFrame({"ti": ti_, "di": di_, "hour": hh_.astype(int), "k": kk.ravel()[idx],
                       "h": exp.ravel()[idx] * TICK_S / 3600})
    df["dcat"] = pd.Categorical(cats[dc_], categories=list(cats))
    pos = df.ti.values * len(D["dates"]) + df.di.values
    for c in TD_COLS + ["route_id", "line_id"]:
        df[c] = td[c].values[pos]
    df["date"] = D["dates"][df.di.values]
    df["age5"] = df.age_years / 5
    df["temp10"] = df.temp_c / 10
    df["rain10"] = df.rain_mm / 10
    return df[df.product_switch.notna() & (df.h > 0)].copy()


# ================================================================================================
# C. carrier score: 03/06 statistic R_c = k_c / h_c x 1000 (30-s grid, NEAR_D = 1.5 hops)
# ================================================================================================
def score_h(D: dict) -> np.ndarray:
    """h_c = train-hours other trains spend within 1.5 hops of candidate c (fixed across draws)."""
    g, near, nl = D["g"], D["near"], D["nl"]
    n_tr, n_d, nt = g.shape
    A = near[:, :nl].astype(np.float64)
    ticks = np.arange(nt)
    h = np.zeros(n_tr)
    for d in range(n_d):
        gd = g[:, d]
        m = gd < nl
        flat = gd[m].astype(np.int64) * nt + np.broadcast_to(ticks, gd.shape)[m]
        occ = np.bincount(flat, minlength=nl * nt).reshape(nl, nt).astype(np.float64)
        nc = A @ occ  # (nl + 1, nt) in-service trains near each position, sentinel row = 0
        h += np.where(m, nc[gd, ticks[None, :]] - 1.0, 0.0).sum(axis=1)  # minus c itself (d = 0 to itself)
    return h * TICK_S / 3600


def score_k(D: dict, ev: dict) -> np.ndarray:
    """k_c for one set of event (train, date, tick) triples; a victim is not counted for its own train."""
    g, near, nl = D["g"], D["near"], D["nl"]
    pos = g[:, ev["date"], ev["tick"]]  # (140, n_ev) candidate position at each onset
    vic = g[ev["train"], ev["date"], ev["tick"]]  # (n_ev,)
    hit = near[pos, vic[None, :]]
    hit[ev["train"], np.arange(len(vic))] = False
    return hit.sum(axis=1).astype(np.int64)


# ================================================================================================
# D. hazard models: 03's primary specification, plus date and route fixed effects
# ================================================================================================
def fit_model(df: pd.DataFrame, extra: tuple[str, ...], label: str) -> dict:
    """Poisson GLM with log train-hour offset and date-clustered SE; proximity block LR test vs the same
    model without the distance terms. `extra` holds the fixed effects to add (e.g. 'C(date)')."""
    prox = "C(dcat, Treatment(reference='d3plus'))"
    terms = BASE_TERMS + list(extra)
    y = df.k.values
    kw = dict(family=sm.families.Poisson(), offset=np.log(df.h))
    t0 = time.time()
    m = smf.glm("k ~ " + " + ".join([prox] + terms), data=df, **kw).fit(
        cov_type="cluster", cov_kwds={"groups": df.date})
    m0 = smf.glm("k ~ " + " + ".join(terms), data=df, **kw).fit(
        cov_type="cluster", cov_kwds={"groups": df.date})
    lr, dfd = 2 * (m.llf - m0.llf), m.df_model - m0.df_model
    ci = m.conf_int()
    prox_terms = [t for t in m.params.index if "dcat" in t]
    rows = []
    for t in prox_terms + ["product_switch"]:
        rows.append({"model": label, "term": t, "label": pretty(t), "RR": float(np.exp(m.params[t])),
                     "lo": float(np.exp(ci.loc[t, 0])), "hi": float(np.exp(ci.loc[t, 1])),
                     "p": float(m.pvalues[t]), "b": float(m.params[t]), "se": float(m.bse[t])})
    out = {"model": label, "n_cells": len(df), "n_events": int(df.k.sum()), "train_h": float(df.h.sum()),
           "n_params": int(m.df_model), "lr_distance": float(lr), "lr_df": float(dfd),
           "lr_p": float(stats.chi2.sf(lr, dfd)), "llf": float(m.llf),
           "n_clusters": int(df.date.nunique()), "fit_s": round(time.time() - t0, 1),
           "n_levels_present": int(df.dcat.nunique()), "terms": rows, "res": m}
    return out


PRETTY = {"product_switch": "Cleaning product switch"}
DIST_NAMES = {"d0": "distance 0 (same segment)", "d0p5": "distance 0.5 hop", "d1": "distance 1 hop",
              "d1p5": "distance 1.5 hops", "d2": "distance 2 hops", "d2p5": "distance 2.5 hops",
              "off": "carrier out of service"}


def pretty(term: str) -> str:
    if "dcat" in term:
        return DIST_NAMES.get(term.split("[T.")[-1].rstrip("]"), term)
    return PRETTY.get(term, term)


def table_of(res: dict) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in r.items() if k != "res"} for r in [res]])


# ================================================================================================
# E. burst-block permutation null
# ================================================================================================
def block_index(bd: pd.DataFrame) -> np.ndarray:
    """Block id per event: the 23 burst ids for burst events, one block per isolated event (no structure to
    preserve, but its own time-of-day is preserved like everything else)."""
    return np.where(bd.in_burst.values, bd.burst_id.values.astype(int), -np.arange(1, len(bd) + 1))


def admissible_shifts(D: dict, bd: pd.DataFrame, block: np.ndarray, min_shift: bool = True) -> dict:
    """Per block: every tick offset delta such that all of its events land on an in-service cell of the same train
    and on the same date. min_shift also drops |delta| < SHIFT_MIN_TICKS (no near-identity placements, the same
    30-min rule as Null B); the robustness variant keeps them."""
    g, nl = D["g"], D["nl"]
    base = np.arange(-(NT - 1), NT)
    out = {}
    for b in np.unique(block):
        idx = np.flatnonzero(block == b)
        d = int(bd.di.values[idx[0]])
        assert (bd.di.values[idx] == d).all(), f"block {b} spans more than one date"
        ok = np.ones(len(base), bool)
        for i in idx:
            t = bd.tick.values[i] + base
            inside = (t >= 0) & (t < NT)
            ok &= inside & (g[bd.ti.values[i], d, np.where(inside, t, 0)] < nl)
        if min_shift:
            ok &= np.abs(base) >= SHIFT_MIN_TICKS
        assert ok.sum() > 0, f"block {b} has no admissible shift"
        out[b] = base[ok]
    return out


def draw_block_ticks(D: dict, bd: pd.DataFrame, block: np.ndarray, adm: dict, reps: int,
                     rng: np.random.Generator) -> np.ndarray:
    """(n_ev, reps) shifted onset ticks: one uniform draw from each block's admissible set, shared by the block."""
    out = np.empty((len(bd), reps), np.int32)
    for b, deltas in adm.items():
        idx = np.flatnonzero(block == b)
        delta = rng.choice(deltas, size=reps)
        out[idx] = bd.tick.values[idx, None] + delta[None, :]
    return out


def null_scan(D: dict, h: np.ndarray, ticks: np.ndarray, reps: int, chunk: int = 500) -> np.ndarray:
    """k_c (n_train, reps) for a set of shifted onset ticks, one chunk of draws at a time."""
    g, near, nl = D["g"], D["near"], D["nl"]
    n_tr = g.shape[0]
    ev = D["ev"]
    k = np.zeros((n_tr, reps), dtype=np.int32)
    for d in np.unique(ev["date"]):
        sel = np.flatnonzero(ev["date"] == d)
        t_d = ticks[sel]  # (n, reps)
        v_tr = ev["train"][sel]
        for a in range(0, reps, chunk):
            b = min(a + chunk, reps)
            t = t_d[:, a:b]
            pos = g[:, d][:, t]  # (n_tr, n, w) candidate position
            vic = g[v_tr[:, None], d, t]  # (n, w) victim position
            hit = near[pos, vic[None, :, :]]
            hit[v_tr[:, None], np.arange(len(sel))[:, None], np.arange(b - a)[None, :]] = False
            k[:, a:b] += hit.sum(axis=1).astype(np.int32)
    return k


def burst_null(D: dict, bd: pd.DataFrame, h: np.ndarray, reps: int, seed: int) -> dict:
    g, ci = D["g"], D["ci"]
    block = block_index(bd)
    adm = admissible_shifts(D, bd, block)
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    ticks = draw_block_ticks(D, bd, block, adm, reps, rng)
    assert (g[bd.ti.values[:, None], bd.di.values[:, None], ticks] < D["nl"]).all(), "shifted event off service"
    k = null_scan(D, h, ticks, reps)
    rate = k / h[:, None] * 1000
    return {"rate": rate, "k": k, "ticks": ticks, "block": block, "adm": adm,
            "n_admissible": {int(b): int(len(v)) for b, v in adm.items()}}


def p_upper(null: np.ndarray, obs: float) -> tuple[float, int]:
    n_ge = int(np.sum(np.asarray(null) >= obs * (1 - 1e-12)))
    return (1 + n_ge) / (1 + len(null)), n_ge


def summ(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {"mean": float(x.mean()), "median": float(np.median(x)), "p97_5": float(np.percentile(x, 97.5)),
            "max": float(x.max()), "min": float(x.min())}


def design_diagnostics(res: dict) -> dict:
    """Rank of the design matrix and, if it is rank-deficient, which columns are aliased.

    statsmodels warns 'rank-deficient' for the primary model as well (03 suppresses the warning), so this is
    reported for every specification rather than treated as new: what matters is whether the distance block and
    the out-of-service level are affected."""
    m = res["res"]
    X = np.asarray(m.model.exog, dtype=np.float64)
    names = list(m.model.exog_names)
    G = X.T @ X
    w, V = np.linalg.eigh(G)
    tol = w.max() * 1e-9
    bad = np.flatnonzero(w <= tol)
    aliased = []
    for j in bad:  # each near-null direction: the columns that share it
        v = np.abs(V[:, j])
        aliased.append(sorted(np.array(names)[v > 0.15].tolist()))
    return {"rank": int((w > tol).sum()), "n_cols": int(len(names)), "n_null": int(len(bad)),
            "aliased_groups": aliased}


def fit_block(df: pd.DataFrame, extra: tuple[str, ...], label: str) -> dict:
    res = fit_model(df, extra, label)
    res["design"] = design_diagnostics(res)
    d = res["design"]
    log(f"[{label}] cells {res['n_cells']:,}, events {res['n_events']}, train-h {res['train_h']:,.0f}, "
        f"params {res['n_params']}, clusters {res['n_clusters']}, distance LR {res['lr_distance']:.1f} on "
        f"{res['lr_df']:.0f} df (p={res['lr_p']:.2g}); design rank {d['rank']}/{d['n_cols']}, "
        f"aliased {d['aliased_groups']}; fit {res['fit_s']}s")
    return res


def model_tables(fits: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for res in fits:
        for r in res["terms"]:
            rows.append({"model": res["model"], **r})
    tab = pd.DataFrame(rows)
    comp = pd.DataFrame([{"model": r["model"], "n_cells": r["n_cells"], "n_events": r["n_events"],
                          "train_h": r["train_h"], "n_params": r["n_params"],
                          "distance_LR": r["lr_distance"], "distance_df": r["lr_df"],
                          "distance_LR_p": r["lr_p"], "design_rank": r["design"]["rank"],
                          "design_cols": r["design"]["n_cols"], "aliased": r["design"]["aliased_groups"]}
                         for r in fits])
    return tab, comp


def stage_models(D: dict, dc: np.ndarray, out_dir: Path) -> list[dict]:
    exp8 = exposure_counts(D, dc, 8)
    k8 = event_counts(D, dc[D["ev"]["train"], D["ev"]["date"], D["ev"]["tick"]], 8)
    df = cells_frame(D, exp8, k8, DCAT)
    log(f"cells frame: {len(df):,} cells, {int(df.k.sum())} events, {df.h.sum():,.0f} train-h, "
        f"{df.date.nunique()} dates, routes {sorted(df.route_id.unique())}")
    fits = [fit_block(df, (), "A0 primary (reproduction)"),
            fit_block(df, ("C(date)",), "A1 + C(date)"),
            fit_block(df, ("C(route_id)",), "B1 + C(route_id)"),
            fit_block(df, ("C(date)", "C(route_id)"), "B2 + C(date) + C(route_id)")]
    tab, comp = model_tables(fits)
    tab.to_csv(out_dir / "sens_model_terms.csv", index=False)
    comp.to_csv(out_dir / "sens_model_compare.csv", index=False)
    log("\nmodel comparison:\n" + comp.drop(columns=["aliased"]).round(2).to_string(index=False))
    for res in fits:
        log(f"\n{res['model']}:\n" + pd.DataFrame(res["terms"])[["label", "RR", "lo", "hi", "p", "se"]]
            .round(3).to_string(index=False))
    return fits


def stage_null(D: dict, bd: pd.DataFrame, out_dir: Path, reps: int, seed: int) -> dict:
    g, ci, trains = D["g"], D["ci"], D["trains"]
    h = score_h(D)
    k_obs = score_k(D, D["ev"])
    rate_obs = k_obs / h * 1000
    obs_carrier, obs_max = float(rate_obs[ci]), float(rate_obs.max())
    assert int(k_obs[ci]) == EXPECTED["k"] and round(float(h[ci]), 1) == EXPECTED["h"] \
        and round(obs_carrier, 3) == EXPECTED["rate"], (k_obs[ci], h[ci], obs_carrier)
    log(f"observed: {CARRIER} {int(k_obs[ci])} events / {h[ci]:,.1f} train-h = {obs_carrier:.3f} per 1000 h; "
        f"max-scan {obs_max:.3f} (best of {len(trains)} trains = {trains[int(rate_obs.argmax())]})")

    block = block_index(bd)
    sizes = pd.Series({b: len(v) for b, v in admissible_shifts(D, bd, block).items()})
    log(f"blocks: {bd[bd.in_burst].burst_id.nunique()} bursts ({int(bd.in_burst.sum())} events) + "
        f"{int((~bd.in_burst).sum())} isolated events = {len(sizes)} units; admissible shifts per unit: "
        f"min {sizes.min()}, median {sizes.median():.0f}, max {sizes.max()}")
    t0 = time.time()
    R = burst_null(D, bd, h, reps, seed)
    rate = R["rate"]
    car, mx = rate[ci], rate.max(axis=0)
    p_carrier, n_carrier = p_upper(car, obs_carrier)
    p_scan, n_scan = p_upper(mx, obs_max)
    log(f"burst-block null ({reps:,} draws, {time.time() - t0:.1f}s): carrier rate mean {car.mean():.2f}, "
        f"median {np.median(car):.2f}, 97.5th pct {np.percentile(car, 97.5):.2f}, max {car.max():.2f}; "
        f"observed {obs_carrier:.3f} -> p={p_carrier:.2g} ({n_carrier} draws >= observed)")
    log(f"burst-block null max-scan: mean {mx.mean():.2f}, median {np.median(mx):.2f}, 97.5th pct "
        f"{np.percentile(mx, 97.5):.2f}, max {mx.max():.2f}; observed {obs_max:.3f} -> p={p_scan:.2g} "
        f"({n_scan} draws >= observed); floor 1/{reps + 1} = {1 / (reps + 1):.2g}")

    # robustness: put the near-identity shifts back in the admissible set
    adm_all = admissible_shifts(D, bd, block, min_shift=False)
    rng2 = np.random.default_rng(np.random.SeedSequence(seed + 1))
    t_all = draw_block_ticks(D, bd, block, adm_all, reps, rng2)
    mx_all = (null_scan(D, h, t_all, reps) / h[:, None] * 1000).max(axis=0)
    p_all, n_all = p_upper(mx_all, obs_max)
    log(f"robustness (near-identity shifts allowed, {reps:,} draws): max-scan mean {mx_all.mean():.2f}, "
        f"max {mx_all.max():.2f}, p={p_all:.2g} ({n_all} draws >= observed)")

    # comparison: the existing independent-event null (out/definitive/null_A_draws.npz, 06_definitive_nulls.py)
    nullA = json.loads((ROOT / "out" / "definitive" / "definitive_nulls_summary.json").read_text(encoding="utf-8"))
    a = nullA["null_A_event_position_permutation"]
    log(f"independent-event null A (existing, {a['replicates']:,} draws): max-scan mean "
        f"{a['max_scan_rate']['mean']:.2f}, max {a['max_scan_rate']['max']:.2f}, p={a['p_max_scan']:.2g}, "
        f"floor {a['p_floor']:.2g}")

    np.savez_compressed(out_dir / "sens_burst_null.npz",
                        max_rate=mx.astype(np.float32), carrier_rate=car.astype(np.float32),
                        carrier_k=R["k"][ci].astype(np.int16), ticks=R["ticks"].astype(np.int16),
                        block=block.astype(np.int32), reps=np.array([reps]))
    rows = [{"test": "burst_block_max_scan", "replicates": reps, "observed": obs_max, "p_value": p_scan,
             "n_ge": n_scan, "p_floor": 1 / (reps + 1), **{f"null_{k}": v for k, v in summ(mx).items()}},
            {"test": "burst_block_carrier", "replicates": reps, "observed": obs_carrier, "p_value": p_carrier,
             "n_ge": n_carrier, "p_floor": 1 / (reps + 1), **{f"null_{k}": v for k, v in summ(car).items()}},
            {"test": "burst_block_max_scan_near_identity_allowed", "replicates": reps, "observed": obs_max,
             "p_value": p_all, "n_ge": n_all, "p_floor": 1 / (reps + 1),
             **{f"null_{k}": v for k, v in summ(mx_all).items()}},
            {"test": "independent_event_max_scan_existing_null_A", "replicates": a["replicates"],
             "observed": obs_max, "p_value": a["p_max_scan"], "n_ge": a["n_ge_max_scan"], "p_floor": a["p_floor"],
             **{f"null_{k}": v for k, v in a["max_scan_rate"].items()}}]
    pd.DataFrame(rows).to_csv(out_dir / "sens_burst_null_summary.csv", index=False)
    return {"reps": reps, "seed": seed, "observed": {"k": float(k_obs[ci]), "h": float(h[ci]),
                                                    "rate": obs_carrier, "max_scan": obs_max},
            "blocks": {"n_bursts": int(bd[bd.in_burst].burst_id.nunique()),
                       "n_burst_events": int(bd.in_burst.sum()),
                       "n_isolated": int((~bd.in_burst).sum()),
                       "admissible_min": int(sizes.min()), "admissible_median": float(sizes.median()),
                       "admissible_max": int(sizes.max())},
            "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--draws", type=int, default=5000, help="burst-block permutation draws (default 5000)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip", choices=("none", "models", "null"), default="none")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    D = load_export()
    dc = dcode_grid(D)
    bd = load_bursts(D)
    log(f"grid {D['g'].shape} (train x date x tick, tick0 05:00), {len(bd)} events, "
        f"{int(D['ev']['date'].max()) + 1} dates")
    summary: dict = {"script": "src/07_sensitivity.py", "seed": args.seed}
    if args.skip != "models":
        fits = stage_models(D, dc, args.out_dir)
        summary["models"] = [{k: v for k, v in r.items() if k not in ("res",)} for r in fits]
    if args.skip != "null":
        summary["burst_null"] = stage_null(D, bd, args.out_dir, args.draws, args.seed)
    summary["runtime_s"] = round(time.time() - t0, 1)
    (args.out_dir / "sens_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    log(f"total {time.time() - t0:.1f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
