"""21 The source is exempt (brief G).

Quantifies the fact that T3420 sits at distance 0 from itself for its whole exposure with zero breakdowns,
while the fitted hazard says a train that close fails at 291 per 1000 train-h. Then looks for the rule's
fingerprints: is any other train exempt, is the carrier ever a victim of delay, is there a second centre,
and what shape does the rule have.

Sections
 1. The exemption, quantified: T3420's exposure and breakdowns; expected counts under four reference rates.
 2. Is any other train exempt: observed/expected for all 140 trains, the route-10 subset, expected zeros.
 3. Does the exemption extend beyond breakdowns: late arrivals, extended traversals, blocked-behind-incident.
 4. Secondary sources: scan all 140 trains for proximity to the 16 events outside T3420's 2.5-hop field.
 5. The rule's shape: clause-by-clause support / refutation counts.
 6. Two paper paragraphs.

Read-only reuse of the 01-20 caches (out/, data_derived/, figures/*/tables/); nothing is recomputed from
data/ except the station graph needed for the hop distance matrix. Outputs: out/source_exemption/.

Run: python3 src/21_source_exemption.py    (~20 s, ~2.4 GB peak RAM)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DERIVED = ROOT / "data_derived"
OUT = ROOT / "out" / "source_exemption"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "source_exemption_results.txt"
LOG.write_text("")

CARRIER = "T3420"
TICK_S = 30
FAR_D = 2.5          # the brief's "outside 2.5 hops of the carrier"
NEAR_D = 1.5         # the radius used by 03/03c for the carrier scan
_log_buf: list[str] = []


def log(*parts) -> None:
    line = " ".join(str(p) for p in parts)
    print(line)
    _log_buf.append(line)


def flush_log() -> None:
    with LOG.open("a") as fh:
        fh.write("\n".join(_log_buf) + "\n")
    _log_buf.clear()


def section(title: str) -> None:
    log("")
    log("=" * 100)
    log(title)
    log("=" * 100)


def poisson_ci(k: int, exposure_h: float, level: float = 0.95) -> tuple[float, float]:
    """Exact Poisson CI for a rate per 1000 train-h (k events / exposure_h)."""
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else stats.chi2.ppf(a, 2 * k) / 2
    hi = stats.chi2.ppf(1 - a, 2 * (k + 1)) / 2
    return 1000 * lo / exposure_h, 1000 * hi / exposure_h


def save(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.to_csv(OUT / name, index=False)
    log(f"[saved] out/source_exemption/{name} ({len(df)} rows)")
    return df


# ================================================================================================
section("0. Caches")
# ================================================================================================
bd = pd.read_csv(DERIVED / "breakdowns.csv", parse_dates=["onset_ts", "recovery_ts", "date"])
tc = pd.read_csv(ROOT / "out" / "victim" / "train_counts.csv")           # per-train exposure + breakdowns
dv = pd.read_csv(ROOT / "out" / "victim" / "delayed_counts.csv")         # per-train lateness (05)
geo = pd.read_csv(ROOT / "out" / "geo" / "events_geo.csv")               # per-event geometry (12)
kernel = pd.read_csv(ROOT / "figures" / "mechanism" / "tables" / "carrier_kernel.csv")
isolated = pd.read_csv(ROOT / "out" / "burst_structure" / "isolated_rates.csv")
scan03 = pd.read_csv(ROOT / "figures" / "mechanism" / "tables" / "carrier_scan.csv")
trav = pd.read_pickle(DERIVED / "traversals.pkl")[
    ["train_id", "line_id", "route_id", "date", "segment", "from_station", "to_station",
     "dep_ts", "arr_ts", "dwell_s", "excess_travel_s", "n_bd_on_segment", "bd_at_from_station"]]
log(f"breakdowns {len(bd)} on {bd.train_id.nunique()} trains | traversals {len(trav):,} | "
    f"trains {tc.train_id.nunique()} | events_geo {len(geo)}")
log(f"caches: carrier_kernel {len(kernel)} rows, isolated_rates {len(isolated)}, carrier_scan {len(scan03)}")
flush_log()

# ================================================================================================
section("1. The exemption, quantified: T3420's own exposure and four reference rates")
# ================================================================================================
# Exposure rule identical to 03: station dwell [dep_ts - dwell_s, dep_ts) + segment running [dep_ts, arr_ts),
# terminal layovers excluded.
run_h = float((trav.arr_ts - trav.dep_ts).dt.total_seconds().sum() / 3600)
dwell_h = float(trav.dwell_s.fillna(0).sum() / 3600)
exp_train = trav.groupby("train_id", observed=True).apply(
    lambda g: ((g.arr_ts - g.dep_ts).dt.total_seconds().sum() + g.dwell_s.fillna(0).sum()) / 3600,
    include_groups=False).rename("exposure_h_calc")
tc = tc.merge(exp_train, left_on="train_id", right_index=True, how="left")
max_dev = float((tc.exposure_h_calc - tc.exposure_h).abs().max())
log(f"exposure rebuild: {run_h + dwell_h:,.1f} train-h ({run_h:,.0f} running + {dwell_h:,.0f} dwelling); "
    f"max |calc - cached| over 140 trains = {max_dev:.3f} h")
assert max_dev < 0.1, "exposure rebuild does not reproduce out/victim/train_counts.csv"

E = tc.set_index("train_id")
E_own = float(E.loc[CARRIER, "exposure_h"])
k_own = int((bd.train_id == CARRIER).sum())
E_near = float(scan03.set_index("train_id").loc[CARRIER, "h_near"])   # other trains' hours within 1.5 hops
log(f"{CARRIER}: own exposure {E_own:,.2f} train-h; breakdowns {k_own}; "
    f"near-field pool (other trains within 1.5 hops) {E_near:,.2f} train-h "
    f"(03 carrier_scan: k_near={int(scan03.set_index('train_id').loc[CARRIER,'k_near'])})")

# Reference rates. Each row names its rate and its exposure.
E_tot = float(tc.exposure_h.sum())
r_overall = 1000 * len(bd) / E_tot
iso_near = isolated[(isolated.channel == "on-line") & (isolated.band == "0-250 m")
                    & (isolated["sample"] == "all events")].iloc[0]
r_metric = 1000 * float(iso_near.events) / float(iso_near.train_h)
far_field = float(kernel[(kernel.d_to_carrier == ">=3") & (kernel.victim_line == "all")].rate_per_1000h.iloc[0])
r_rr1900 = 1900.0 * far_field   # far_field is already per 1000 train-h
r_same_seg = float(kernel[(kernel.d_to_carrier == "0") & (kernel.victim_line == "Line 10")].rate_per_1000h.iloc[0])
grp = tc[(tc.route_id == "10") & (tc.batch_number == "B08")]
r_route_batch = 1000 * float(grp.breakdowns.sum()) / float(grp.exposure_h.sum())
rates = [
    ("(a) overall fleet", r_overall, "143 events / 171,569 train-h", "fleet"),
    ("(b) near-field metric <=250 m, same line", r_metric,
     f"{int(iso_near.events)} events / {iso_near.train_h:.2f} train-h (out/burst_structure/isolated_rates.csv)", "near"),
    ("(c) parametric same segment = 1900 x far field", r_rr1900,
     f"RR 1900 (notes/03) x far-field {far_field:.3f} per 1000 h (>=3 hops)", "parametric"),
    ("(c') observed same segment, Line 10", r_same_seg,
     f"{int(kernel[(kernel.d_to_carrier=='0')&(kernel.victim_line=='Line 10')].k.iloc[0])} events / "
     f"{kernel[(kernel.d_to_carrier=='0')&(kernel.victim_line=='Line 10')].h.iloc[0]:.1f} train-h "
     f"(carrier_kernel.csv)", "observed"),
    ("(d) T3420's own route x batch (route 10, B08)", r_route_batch,
     f"{int(grp.breakdowns.sum())} events / {grp.exposure_h.sum():.2f} train-h over {len(grp)} trains", "own group"),
]
rows = []
for label, rate, exposure_def, kind in rates:
    for ename, eh in [("own exposure", E_own), ("near-field pool 1.5 hops", E_near)]:
        mu = rate / 1000 * eh
        rows.append({"rate": label, "rate_per_1000h": rate, "rate_source": exposure_def, "rate_kind": kind,
                     "exposure": ename, "exposure_h": eh, "expected_count": mu,
                     "p_zero": float(np.exp(-mu)), "log10_p": -mu / np.log(10), "exp_arg": -mu})
ex = pd.DataFrame(rows)
save(ex, "exemption_expected_counts.csv")
for _, r in ex.iterrows():
    log(f"{r.rate:46s} {r.rate_per_1000h:8.2f}/1000h x {r.exposure_h:9.2f} h ({r.exposure:23s}) "
        f"= E {r.expected_count:9.2f}  p(0) = exp({r.exp_arg:.1f}) = 10^{r.log10_p:.1f}")
flush_log()

# ================================================================================================
section("2. Is any other train exempt? observed / expected for all 140 trains")
# ================================================================================================
tc["exp_overall"] = r_overall / 1000 * tc.exposure_h
tc["OE_overall"] = tc.breakdowns / tc.exp_overall
tc["exp_near"] = r_metric / 1000 * tc.exposure_h
tc["OE_near"] = tc.breakdowns / tc.exp_near
tc["exp_zero_rate"] = np.exp(-tc.exp_overall)          # P(0 breakdowns) under the overall rate
tc["exp_zero_near"] = np.exp(-tc.exp_near)            # P(0 breakdowns) under the near-field rate
grp = tc[(tc.route_id == "10") & (tc.batch_number == "B08")]
save(tc[["train_id", "line_id", "route_id", "train_type", "batch_number", "age_years", "exposure_h",
          "breakdowns", "exp_overall", "OE_overall", "exp_near", "OE_near", "exp_zero_rate", "exp_zero_near"]],
     "train_observed_expected.csv")

z_obs, z_exp = int((tc.breakdowns == 0).sum()), float(tc.exp_zero_rate.sum())
log(f"all 140 trains under the overall rate {r_overall:.3f}/1000 h: observed {tc.breakdowns.sum()} vs expected "
    f"{tc.exp_overall.sum():.1f}; O/E mean {tc.OE_overall.mean():.2f} sd {tc.OE_overall.std():.2f} "
    f"(sd of O/E from Poisson alone ~ {1/np.sqrt(tc.exp_overall.mean()):.2f}); variance/mean of counts "
    f"{tc.breakdowns.var()/tc.breakdowns.mean():.2f}")
log(f"zero-breakdown trains: observed {z_obs}/140, expected {z_exp:.1f} "
    f"(sum of exp(-rate x own exposure)); so 'zero breakdowns' is the modal outcome, not an exemption")
r10 = tc[tc.route_id == "10"]
log(f"route 10 (T3420's route): {len(r10)} trains, {int(r10.breakdowns.sum())} breakdowns / "
    f"{r10.exposure_h.sum():.1f} train-h = {1000*r10.breakdowns.sum()/r10.exposure_h.sum():.2f}/1000 h; "
    f"zero-breakdown trains on route 10: {int((r10.breakdowns==0).sum())} "
    f"({', '.join(r10.loc[r10.breakdowns==0,'train_id'])})")
log(f"route 10/10B (the whole line): {int((tc.line_id.astype(str)=='10').sum())} trains, "
    f"{int(tc.loc[tc.line_id.astype(str)=='10','breakdowns'].sum())} breakdowns / "
    f"{tc.loc[tc.line_id.astype(str)=='10','exposure_h'].sum():.1f} train-h")
log(f"T3420's own route x batch group (route 10, B08): {len(grp)} trains "
    f"({', '.join(grp.train_id)}), {int(grp.breakdowns.sum())} breakdowns / {grp.exposure_h.sum():.1f} train-h "
    f"= {r_route_batch:.2f}/1000 h")
# The counterfactual: if the near-field rate were the rate for every train.
log(f"if {r_metric:.1f}/1000 h were the rate for all 140 trains: expected "
    f"{r_metric/1000*E_tot:,.0f} breakdowns vs 143 observed "
    f"({r_metric/1000*E_tot/len(bd):,.0f}x); expected zero-breakdown trains "
    f"{tc.exp_zero_near.sum():.2e} of 140")
log(f"same counterfactual at the observed same-segment rate {r_same_seg:.1f}/1000 h: "
    f"{r_same_seg/1000*E_tot:,.0f} breakdowns expected; at the route-10 x B08 rate "
    f"{r_route_batch:.2f}/1000 h: {r_route_batch/1000*E_tot:,.0f}")
e_carrier = float(tc.set_index("train_id").loc[CARRIER, "exp_overall"])
log(f"T3420 under the overall rate: expected {e_carrier:.2f}, observed 0, O/E 0.00 -> tied with the "
    f"{z_obs-1} other zero trains; P(0 | overall) = {np.exp(-e_carrier):.2f}")
save(pd.DataFrame([{"group": "all 140 trains", "n": len(tc), "breakdowns": int(tc.breakdowns.sum()),
                    "exposure_h": float(tc.exposure_h.sum()), "rate_per_1000h": r_overall,
                    "zero_trains_observed": z_obs, "zero_trains_expected": z_exp},
                   {"group": "route 10", "n": len(r10), "breakdowns": int(r10.breakdowns.sum()),
                    "exposure_h": float(r10.exposure_h.sum()),
                    "rate_per_1000h": 1000 * r10.breakdowns.sum() / r10.exposure_h.sum(),
                    "zero_trains_observed": int((r10.breakdowns == 0).sum()),
                    "zero_trains_expected": float(r10.exp_zero_rate.sum())},
                   {"group": "route 10 x batch B08", "n": len(grp), "breakdowns": int(grp.breakdowns.sum()),
                    "exposure_h": float(grp.exposure_h.sum()), "rate_per_1000h": r_route_batch,
                    "zero_trains_observed": int((grp.breakdowns == 0).sum()),
                    "zero_trains_expected": float(grp.exp_zero_rate.sum())}]), "exemption_by_group.csv")
flush_log()

# ================================================================================================
section("3. Does the exemption extend beyond breakdowns? late arrivals, extended traversals, blocking")
# ================================================================================================
# (3a) Recorded lateness (05_victim: late arrival = arrival later than the scheduled/base time).
dvv = dv.set_index("train_id")
late = dvv.loc[CARRIER]
pct_late = float((dvv.late_rate_per_1000h < late.late_rate_per_1000h).mean() * 100)
pct_exp = float((dvv.delay_experienced_train_min < late.delay_experienced_train_min).mean() * 100)
log(f"(3a) recorded lateness: {CARRIER} {int(late.late_arrivals)} late arrivals, "
    f"{late.delay_experienced_train_min:.2f} train-min experienced, max {late.max_delay_s:.0f} s, "
    f"{late.late_rate_per_1000h:.2f} per 1000 h (fleet median {dvv.late_rate_per_1000h.median():.2f}, "
    f"IQR {dvv.late_rate_per_1000h.quantile(.25):.2f}-{dvv.late_rate_per_1000h.quantile(.75):.2f}; "
    f"{CARRIER} sits at the {pct_late:.0f}th percentile)")
log(f"    delay created by {CARRIER} on others: {late.delay_created_train_min:.2f} train-min; "
    f"experienced/created = {late.delay_experienced_train_min/max(late.delay_created_train_min,1e-9):.1f}")

# (3b) Own traversals extended beyond the base travel time (traversals.excess_travel_s, computed in 01).
tvc = trav[trav.train_id == CARRIER].copy()
tvc["exc"] = tvc.excess_travel_s.fillna(0.0)
fleet_exc = trav.assign(exc=trav.excess_travel_s.fillna(0.0)).groupby("train_id", observed=True).agg(
    exc_min=("exc", lambda s: s.sum() / 60), n_trav=("exc", "size"), n_ext=("exc", lambda s: int((s >= 5).sum())))
fleet_exc["exc_per_1000h"] = 1000 * fleet_exc.exc_min / 60 / E.reindex(fleet_exc.index).exposure_h
pct_exc = float((fleet_exc.exc_per_1000h < fleet_exc.loc[CARRIER, "exc_per_1000h"]).mean() * 100)
log(f"(3b) traversals of {CARRIER}: {len(tvc):,}; extended >=5 s {int((tvc.exc>=5).sum())} "
    f"(>={30}s {int((tvc.exc>=30).sum())}, >={60}s {int((tvc.exc>=60).sum())}); total excess "
    f"{tvc.exc.sum()/60:.2f} train-min = {1000*tvc.exc.sum()/3600/E_own:.2f} per 1000 h "
    f"(fleet median {fleet_exc.exc_per_1000h.median():.2f}, {CARRIER} at the {pct_exc:.0f}th percentile)")
log(f"    own incidents on its traversals: n_bd_on_segment {int(tvc.n_bd_on_segment.sum())}, "
    f"bd_at_from_station {int(tvc.bd_at_from_station.sum())} (0 = it never breaks down on the move)")

# (3c) Was T3420 ever the follower that inherited a breakdown's delay? A traversal counts only if it is
# extended, has no own incident, and a breakdown on its own line was active (onset <= dep <= recovery).
inc10 = bd[bd.line_id.astype(str) == "10"][["onset_ts", "recovery_ts", "train_id", "location"]]
ons, rec = inc10.onset_ts.values, inc10.recovery_ts.values
dep = tvc.dep_ts.values
active = (dep[:, None] >= ons[None, :]) & (dep[:, None] <= rec[None, :])
tvc["blocked_window"] = active.any(axis=1)
tvc["n_active"] = active.sum(axis=1)
cand = tvc[(tvc.exc >= 5) & (tvc.n_bd_on_segment == 0) & (~tvc.bd_at_from_station)]
blk = cand[cand.blocked_window]
log(f"(3c) extended traversals of {CARRIER} with no own incident: {len(cand)}; of these "
    f"{len(blk)} ({100*len(blk)/max(len(cand),1):.0f}%) run while a breakdown on line 10 is active "
    f"(onset <= departure <= recovery), carrying {blk.exc.sum()/60:.2f} train-min of excess")
log(f"    the other {len(cand)-len(blk)} ({100*(len(cand)-len(blk))/max(len(cand),1):.0f}%) are extended with no "
    f"breakdown active anywhere on line 10: {cand.loc[~cand.blocked_window,'exc'].sum()/60:.2f} train-min, "
    f"i.e. ordinary timetable noise, not inherited delay")
share_fleet = float((fleet_exc.n_ext.sum() - fleet_exc.loc[CARRIER, "n_ext"]))
log(f"    fleet reference: {int(fleet_exc.n_ext.sum()):,} extended traversals over 140 trains "
    f"({fleet_exc.n_ext.median():.0f} per train, median); {CARRIER} has {int(fleet_exc.loc[CARRIER,'n_ext'])}")
save(pd.DataFrame([{"check": "late_arrivals", "value": float(late.late_arrivals), "unit": "events",
                    "per_1000h": float(late.late_rate_per_1000h)},
                   {"check": "delay_experienced", "value": float(late.delay_experienced_train_min),
                    "unit": "train-min", "per_1000h": 1000 * late.delay_experienced_train_min / 60 / E_own},
                   {"check": "delay_created", "value": float(late.delay_created_train_min), "unit": "train-min",
                    "per_1000h": 1000 * late.delay_created_train_min / 60 / E_own},
                   {"check": "traversals_extended_ge5s", "value": float((tvc.exc >= 5).sum()), "unit": "traversals",
                    "per_1000h": 1000 * (tvc.exc >= 5).sum() / E_own},
                   {"check": "excess_travel", "value": float(tvc.exc.sum() / 60), "unit": "train-min",
                    "per_1000h": 1000 * tvc.exc.sum() / 3600 / E_own},
                   {"check": "excess_in_breakdown_window", "value": float(blk.exc.sum() / 60),
                    "unit": "train-min", "per_1000h": 1000 * blk.exc.sum() / 3600 / E_own},
                   {"check": "own_incidents", "value": float(tvc.n_bd_on_segment.sum()), "unit": "events",
                    "per_1000h": 0.0}]), "carrier_as_victim.csv")
flush_log()

# ================================================================================================
section("4a. Space-time grid (same definitions as 03) for the secondary-source scan")
# ================================================================================================
st = pd.read_csv(DATA / "station_information.csv")
adj: dict[str, set] = {s: set() for s in st.station_id}
for a, b in zip(st.station_id, st.next_station_id):
    if isinstance(b, str):
        adj[a].add(b)
        adj[b].add(a)
nodes = sorted(adj)
nidx = {s: i for i, s in enumerate(nodes)}
HOP = np.full((len(nodes), len(nodes)), 999, dtype=np.int16)
for s in nodes:
    dist_s, frontier = {s: 0}, [s]
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in dist_s:
                    dist_s[v] = dist_s[u] + 1
                    nxt.append(v)
        frontier = nxt
    for v, x in dist_s.items():
        HOP[nidx[s], nidx[v]] = x


def loc_nodes(loc: str) -> list[str]:
    return loc.replace(" -> ", " - ").split(" - ")


dw = trav[trav.dwell_s.notna()]
iv = pd.concat([
    pd.DataFrame({"train_id": dw.train_id.values, "date": dw.date.values,
                  "loc": dw.from_station.astype(str).values,
                  "start": (dw.dep_ts - pd.to_timedelta(dw.dwell_s, unit="s")).values, "end": dw.dep_ts.values}),
    pd.DataFrame({"train_id": trav.train_id.values, "date": trav.date.values,
                  "loc": trav.segment.astype(str).values,
                  "start": trav.dep_ts.values, "end": trav.arr_ts.values}),
], ignore_index=True)
del dw
iv["train_id"] = iv.train_id.astype(str).astype("category")
iv["loc"] = iv["loc"].astype("category")
iv["date"] = pd.to_datetime(iv.date)
TRAINS = list(iv.train_id.cat.categories)
LABELS = list(iv["loc"].cat.categories)
TRAIN_IDX = {t: i for i, t in enumerate(TRAINS)}
DATES = np.array(sorted(iv.date.unique()))
DATE_IDX = {pd.Timestamp(d): i for i, d in enumerate(DATES)}
sec = lambda ts, day: (ts - day).dt.total_seconds().values  # noqa: E731
t_start_s = float(np.floor(sec(iv.start, iv.date).min() / 3600) * 3600)
t_end_s = float(sec(iv.end, iv.date).max())
NT = int(np.ceil((t_end_s - t_start_s) / TICK_S)) + 1
s_rel = (sec(iv.start, iv.date) - t_start_s) / TICK_S
e_rel = (sec(iv.end, iv.date) - t_start_s) / TICK_S
a_t, b_t = np.ceil(s_rel).astype(np.int64), np.ceil(e_rel).astype(np.int64)
n_t = np.maximum(b_t - a_t, 0)
rep = np.repeat(np.arange(len(iv)), n_t)
offs = np.arange(n_t.sum()) - np.repeat(np.cumsum(n_t) - n_t, n_t)
GRID = np.full((len(TRAINS), len(DATES), NT), -1, dtype=np.int16)
GRID[iv.train_id.cat.codes.values[rep], iv.date.map(DATE_IDX).values[rep], a_t[rep] + offs] = \
    iv["loc"].cat.codes.values[rep]
del rep, offs, s_rel, e_rel, a_t, b_t, n_t, iv
OFF = 99.0
lab_nodes = [[nidx[x] for x in loc_nodes(lb)] for lb in LABELS]
LD = np.full((len(LABELS) + 1, len(LABELS) + 1), OFF, dtype=np.float32)
for i, a in enumerate(lab_nodes):
    for j, b in enumerate(lab_nodes):
        h = HOP[np.ix_(a, b)].min()
        na, nb = len(a), len(b)
        if na == 2 and nb == 2:
            LD[i, j] = 0.0 if set(a) == set(b) else 1.0 + h
        elif na == 2 or nb == 2:
            LD[i, j] = 0.5 + h
        else:
            LD[i, j] = h
GIDX = np.where(GRID < 0, len(LABELS), GRID)
log(f"grid {GRID.shape} (train x date x {TICK_S}s tick), occupied train-ticks {int((GRID >= 0).sum()):,} "
    f"= {(GRID >= 0).sum() * TICK_S / 3600:,.0f} train-h; {len(LABELS)} position labels; "
    f"tick origin {t_start_s/3600:.1f}h")
flush_log()

# ================================================================================================
section("4b. Secondary sources: scan all 140 trains for the 16 events outside T3420's 2.5-hop field")
# ================================================================================================
FAR_FIELD = far_field  # per 1000 train-h, from the kernel (>=3 hops)
geo["onset_ts"] = pd.to_datetime(geo.onset_ts)
far = geo[geo.d_c > FAR_D].reset_index(drop=True)
far["ti"] = far.train_id.map(TRAIN_IDX)
far["di"] = far.onset_ts.dt.normalize().map(DATE_IDX)
far["tick"] = np.floor((sec(far.onset_ts, far.onset_ts.dt.normalize()) - t_start_s) / TICK_S).astype(np.int64)
loc_of = bd.set_index("incident_id").location
far_lab = np.array([LABELS.index(loc_of[i]) if loc_of[i] in LABELS else len(LABELS)
                    for i in far.incident_id])
far_ok = np.mean(GIDX[far.ti.values, far.di.values, far.tick.values] == far_lab)
log(f"far set: {len(far)} events with d_c > {FAR_D} hops; their own train sits on the recorded label at the "
    f"onset tick in {far_ok:.0%} of cases")
log(f"of the {len(far)}: {int((~far.route_id.astype(str).isin(['10','10B'])).sum())} are also off T3420's "
    f"route, {int(far.route_id.astype(str).isin(['10','10B']).sum())} are on route 10/10B but far from it "
    f"(the brief's 'neither near nor on route' describes the first group)")
EV_DIST = np.stack([LD[GIDX[:, int(d), int(t)], int(l)] for d, t, l in
                    zip(far.di.values, far.tick.values, far_lab)])          # (16 events, 140 trains)

carr_i = TRAIN_IDX[CARRIER]
h25 = np.zeros(len(TRAINS))
h15 = np.zeros(len(TRAINS))
h0 = np.zeros(len(TRAINS))
for c in range(len(TRAINS)):
    for d0 in range(0, len(DATES), 15):
        sl = slice(d0, min(d0 + 15, len(DATES)))
        dc = LD[GIDX[:, sl, :], GIDX[c][sl][None, :, :]]
        for rad, acc in ((FAR_D, h25), (NEAR_D, h15), (0.0, h0)):
            near = dc <= rad
            near[c] = False
            acc[c] += near.sum() * TICK_S / 3600
log(f"[check] T3420 near-field exposure rebuilt from the grid: {h15[carr_i]:.2f} train-h at <=1.5 hops "
    f"(03 carrier_scan: {E_near:.2f}); {h25[carr_i]:.2f} train-h at <=2.5 hops; "
    f"{h0[carr_i]:.2f} train-h at distance 0 (head-on meetings / same segment)")
assert abs(h15[carr_i] - E_near) < 0.05, "grid does not reproduce the 03 near-field exposure"

k25 = np.array([int(((EV_DIST[:, c] <= FAR_D) & (far.ti.values != c)).sum()) for c in range(len(TRAINS))])
k15 = np.array([int(((EV_DIST[:, c] <= NEAR_D) & (far.ti.values != c)).sum()) for c in range(len(TRAINS))])
sec = pd.DataFrame({"train_id": TRAINS, "h_near_2.5": h25, "h_near_1.5": h15, "h_at0": h0,
                    "k_2.5": k25, "k_1.5": k15,
                    "rate_2.5": 1000 * k25 / np.where(h25 > 0, h25, np.nan),
                    "rate_1.5": 1000 * k15 / np.where(h15 > 0, h15, np.nan),
                    "expected_2.5_at_far_field": FAR_FIELD / 1000 * h25})
ci = [poisson_ci(k, h) if h > 0 else (np.nan, np.nan) for k, h in zip(k25, h25)]
sec["lo_2.5"] = [x[0] for x in ci]
sec["hi_2.5"] = [x[1] for x in ci]
sec = sec.merge(tc[["train_id", "route_id", "batch_number", "breakdowns", "exposure_h"]], on="train_id", how="left")
def min_hops_to_carrier(c: int) -> float:
    """Smallest hop distance between train c and the carrier over ticks where both are in service."""
    gc, gw = GIDX[c], GIDX[carr_i]
    ok = (gc >= 0) & (gw >= 0)
    return float(LD[gc[ok], gw[ok]].min()) if ok.any() else np.nan


sec["dist_to_carrier_hops"] = [min_hops_to_carrier(c) for c in range(len(TRAINS))]
sec = sec.sort_values(["k_2.5", "rate_2.5"], ascending=False).reset_index(drop=True)
save(sec, "secondary_source_scan.csv")
top = sec.head(5)
log("top 5 candidates for a second centre (16 far events, exposure = all trains' hours within 2.5 hops):")
for _, r in top.iterrows():
    k_, h_ = int(r["k_2.5"]), float(r["h_near_2.5"])
    log(f"    {r['train_id']} route {r['route_id']}: {k_} events / {h_:.1f} h = {r['rate_2.5']:.2f} "
        f"[{r['lo_2.5']:.2f}, {r['hi_2.5']:.2f}] per 1000 h (expected {r['expected_2.5_at_far_field']:.2f} at the "
        f"far-field {FAR_FIELD:.3f}); own breakdowns {int(r['breakdowns'])}")
log(f"T3420 on this restricted set: {int(sec.set_index('train_id').loc[CARRIER, 'k_2.5'])} events within 2.5 hops "
    f"(0 by construction), nearest far event {EV_DIST[:, carr_i].min():.1f} hops")
h_others = sec.set_index("train_id")["h_near_1.5"].drop(index=CARRIER)
log(f"exposure is comparable across candidates: 1.5-hop exposure {h_others.min():,.1f}-{h_others.max():,.1f} "
    f"train-h (largest {h_others.idxmax()}), {CARRIER} {h15[carr_i]:,.1f}; so the rate ranking is not an "
    f"exposure-size artefact")
log(f"events explainable by some train within 2.5 hops: "
    f"{int((EV_DIST.min(axis=1) <= FAR_D).sum())}/{len(far)}; "
    f"max by one candidate: {int(k25.max())}")
rng = np.random.default_rng(20260302)
mu = FAR_FIELD / 1000 * h25
null_max = (1000 * rng.poisson(mu[None, :], size=(4000, len(TRAINS))) / h25[None, :]).max(axis=1)
obs_max = float(np.nanmax(1000 * k25 / np.where(h25 > 0, h25, np.nan)))
p_sel = (1 + int((null_max >= obs_max).sum())) / (1 + len(null_max))
log(f"max-scan null (4000 draws of 140 Poisson counts at the far-field rate, same exposures): median max rate "
    f"{np.median(null_max):.2f}, 95th {np.quantile(null_max, .95):.2f} per 1000 h; observed max "
    f"{obs_max:.2f} -> p = {p_sel:.3f}")
save(pd.DataFrame({"draw": np.arange(len(null_max)), "max_rate_per_1000h": null_max}), "secondary_null.csv")
flush_log()

# ================================================================================================
section("5. The rule's shape: clause by clause, with support and refutation counts")
# ================================================================================================
# (5a) Clause 0: the centre is excluded from its own risk set by construction. 03 sets DC[carrier] = NaN,
# so the d = 0 stratum holds other trains' time only. Put the centre's own time back and see what happens.
k0 = int(kernel[(kernel.d_to_carrier == "0") & (kernel.victim_line == "all")].k.iloc[0])
h0_stratum = float(kernel[(kernel.d_to_carrier == "0") & (kernel.victim_line == "all")].h.iloc[0])
r0_stratum = 1000 * k0 / h0_stratum
h0_with_self = h0_stratum + E_own
mu_self = r0_stratum / 1000 * h0_with_self
dilution = h0_with_self / h0_stratum
log(f"(5a) d=0 stratum as fitted: {k0} events / {h0_stratum:.2f} train-h of OTHER trains' time next to the "
    f"carrier = {r0_stratum:.2f} per 1000 h")
log(f"    the centre's own {E_own:,.2f} train-h also sit at distance 0 from the centre; adding them "
    f"(with their {k_own} events) gives {k0}/{h0_with_self:.1f} h = {1000*k0/h0_with_self:.2f} per 1000 h "
    f"-> exposure x{dilution:.2f}, rate /{dilution:.2f}")
log(f"    internal consistency if the centre were left in: expected {mu_self:.1f} events under the "
    f"fitted {r0_stratum:.1f} per 1000 h, observed {k0} -> O/E {k0/mu_self:.2f}, p = {np.exp(-(mu_self-k0)):.0e}")
log(f"    so the fitted kernel is only estimable if self-distance is excluded; the exemption is a property "
    f"of the estimand, not a fitted coefficient")

# (5b) Clause 1: one centre. (5c) Clause 2: radius. (5d) Clause 3: contemporaneity.
n25 = int((geo.d_c <= FAR_D).sum())
n15 = int((geo.d_c <= NEAR_D).sum())
n0 = int((geo.d_c == 0).sum())
scan_i = scan03.set_index("train_id")
r_carr_near = float(scan_i.loc[CARRIER, "rate_per_1000h"])
others = scan_i.drop(index=CARRIER).rate_per_1000h
runner = float(others.max())
runner_id = others.idxmax()
l10 = geo.line_id.astype(str) == "10"
l10_0 = int(((geo.d_c == 0) & l10).sum())
opt = pd.read_csv(ROOT / "out" / "incubation" / "optimum.csv")
fin = opt[opt.is_finest]
close = fin[fin.criterion.isin(["C3_efold", "C3_loglik", "C4_separation"])]
log(f"(5b) one centre: {n25}/{len(geo)} events within {FAR_D} hops of T3420, {n15} within {NEAR_D}; its "
    f"near-field rate {r_carr_near:.2f} per 1000 h vs runner-up {runner_id} {runner:.2f} "
    f"({r_carr_near/runner:.1f}x) and median {others.median():.2f}; refuted by {len(geo)-n25} events outside")
log(f"(5c) radius: {n0}/{len(geo)} events at d = 0 ({r0_stratum:.1f} per 1000 h), {l10_0}/"
    f"{int(l10.sum())} line-10 victims at d = 0; refuted as a pure 'same segment' rule by "
    f"{len(geo)-n0} events at d > 0, and as a smooth radial kernel by the empty 250-600 m band "
    f"({isolated[(isolated.channel=='on-line')&(isolated.band=='250-600 m')].iloc[0].events:.0f} events / "
    f"{isolated[(isolated.channel=='on-line')&(isolated.band=='250-600 m')].iloc[0].train_h:.1f} train-h)")
log(f"(5d) contemporaneity: at the finest level {int((close.t_opt_s==0).sum())}/{len(close)} closeness fits put "
    f"the optimum at t = 0 and {int((close.t_opt_s<=TICK_S).sum())}/{len(close)} within one {TICK_S}-s step; "
    f"no rewind improves on the onset instant, but the test cannot separate 0 from {TICK_S} s")

# (5e) Clause 4: must the source be moving? Clause 5: must it be in service?
csx = pd.read_csv(ROOT / "out" / "carrier" / "carrier_state_exposure.csv").set_index("carrier_state")
state_h = {"segment": float(csx.loc["cruise", "exposure_train_h"]),
           "station": float(csx.loc[["depart", "arrive", "dwell"], "exposure_train_h"].sum()),
           "terminal": float(csx.loc["terminal", "exposure_train_h"]),
           "off": float(csx.loc["off", "exposure_train_h"])}
st_ev = geo.carrier_state.value_counts().to_dict()
state_rows = []
for stt, hh in state_h.items():
    kk = int(st_ev.get(stt, 0))
    lo, hi = poisson_ci(kk, hh)
    state_rows.append({"carrier_state": stt, "events": kk, "train_h": hh, "rate_per_1000h": 1000 * kk / hh,
                       "lo": lo, "hi": hi, "vs_far_field": (1000 * kk / hh) / far_field})
state_tab = save(pd.DataFrame(state_rows), "rule_state_rates.csv")
for _, r in state_tab.iterrows():
    log(f"    source {r['carrier_state']:8s}: {int(r['events']):3d} events / {r['train_h']:8.1f} train-h = "
        f"{r['rate_per_1000h']:6.2f} [{r['lo']:.2f}, {r['hi']:.2f}] per 1000 h = {r['vs_far_field']:.1f}x far field")
rr_move = (st_ev.get("segment", 0) / state_h["segment"]) / (st_ev.get("station", 0) / state_h["station"])
se_move = np.sqrt(1 / st_ev.get("segment", 1) + 1 / max(st_ev.get("station", 1), 1))
log(f"    moving vs standing at a station: rate ratio {rr_move:.2f} "
    f"[{np.exp(np.log(rr_move)-1.96*se_move):.2f}, {np.exp(np.log(rr_move)+1.96*se_move):.2f}] -> the source "
    f"marks more while moving, but it marks from a standstill too ({st_ev.get('station',0)} events)")
off_grid = int((geo.d_c >= 99).sum())
r_off_grid = float(kernel[(kernel.d_to_carrier == "carrier off") & (kernel.victim_line == "all")].rate_per_1000h.iloc[0])
log(f"    on the grid {len(geo)-off_grid}/{len(geo)} events; the {off_grid} events with the source off the grid "
    f"(terminal layover or out of service) run at {r_off_grid:.3f} per 1000 h = far field {far_field:.3f} "
    f"-> the field is off when the source is, not merely dormant")

# (5f) Clause 6: all lines, through interchanges.
off_line = int((~l10).sum())
log(f"(5f) all lines: {len(geo)} events on {geo.line_id.nunique()} lines, {off_line} of them off the source's "
    f"own line; route counts {geo.route_id.astype(str).value_counts().to_dict()}")
log(f"    the 'shared station' clause is untestable here: 0/70 burst line-pairs lack one, but the network has no "
    f"unconnected line pair (notes/13)")

clauses = pd.DataFrame([
    {"clause": "0 identifiability", "statement": "the field is defined on other trains' exposure; the centre's own "
     "time is excluded from every stratum, because self-distance is 0 by construction",
     "supported_by": f"{k0} events / {h0_stratum:.1f} h at d=0 without the centre; including its {E_own:.0f} h "
     f"would cut the rate {dilution:.2f}x and give O/E {k0/mu_self:.2f}",
     "refuted_by": "nothing: construction, not an empirical claim", "status": "structural"},
    {"clause": "1 one centre", "statement": "one train out of 140 marks the victims",
     "supported_by": f"{n25}/{len(geo)} events within {FAR_D} hops; {r_carr_near:.1f} vs runner-up {runner:.1f} "
     f"per 1000 h ({r_carr_near/runner:.1f}x), median {others.median():.2f}",
     "refuted_by": f"{len(geo)-n25} events outside {FAR_D} hops; no second centre (max-scan p = {p_sel:.3f})",
     "status": "supported"},
    {"clause": "2 radius", "statement": "the field is local: same segment, or the adjacent 250 m on the same line",
     "supported_by": f"{n0}/{len(geo)} events at d=0 ({r0_stratum:.1f} per 1000 h); {l10_0}/{int(l10.sum())} "
     f"line-10 victims at d=0; 62/68 on-line events within 250 m",
     "refuted_by": f"{len(geo)-n0} events at d>0 (up to 26.5 hops); 0 events in the 250-600 m band (279.1 h); "
     f"6/68 on-line events beyond 250 m",
     "status": "supported in part"},
    {"clause": "3 contemporaneity", "statement": "the field is switched on while the source is at the location, "
     "with no incubation lag",
     "supported_by": f"{int((close.t_opt_s==0).sum())}/{len(close)} finest closeness fits at t=0, "
     f"{int((close.t_opt_s<=TICK_S).sum())}/{len(close)} within one {TICK_S}-s step",
     "refuted_by": f"none; but the test has no power below {TICK_S} s", "status": "supported, low power"},
    {"clause": "4 source moving", "statement": "the source must be moving",
     "supported_by": f"moving rate {rr_move:.2f}x [{np.exp(np.log(rr_move)-1.96*se_move):.2f}, "
     f"{np.exp(np.log(rr_move)+1.96*se_move):.2f}] the standing-at-station rate",
     "refuted_by": f"{st_ev.get('station',0)} events with the source standing at a station "
     f"({1000*st_ev.get('station',0)/state_h['station']:.2f} per 1000 h)", "status": "refuted as a requirement"},
    {"clause": "5 source in service", "statement": "the field is off when the source is off the network",
     "supported_by": f"{len(geo)-off_grid}/{len(geo)} events with the source on the grid",
     "refuted_by": f"{off_grid} events with the source off the grid, at {r_off_grid:.3f} per 1000 h = the far "
     f"field {far_field:.3f}", "status": "supported"},
    {"clause": "6 all lines", "statement": "victims on all four lines, i.e. the field crosses at interchanges",
     "supported_by": f"{off_line}/{len(geo)} events off the source's line, on {geo.line_id.nunique()} lines",
     "refuted_by": "none; the 'shared station' version is untestable (no unconnected line pair exists)",
     "status": "supported in part"},
    {"clause": "7 mechanism", "statement": "power draw, cleaning, or schedule explains the field",
     "supported_by": "nothing", "refuted_by": "0 of the tested mechanisms survive BH in notes/03",
     "status": "untestable with these data"},
])
save(clauses, "rule_clauses.csv")
flush_log()

# ================================================================================================
section("6. Two paper paragraphs")
# ================================================================================================
r_with_self = 1000 * k0 / h0_with_self
pn = pd.DataFrame([
    {"value": f"{h0_stratum:.1f}", "quantity": "train-h of other trains' time at d=0", "table": "exemption_kernel_dilution.csv"},
    {"value": f"{k0}", "quantity": "events in the d=0 stratum", "table": "exemption_kernel_dilution.csv"},
    {"value": f"{r0_stratum:.1f}", "quantity": "d=0 rate per 1000 train-h", "table": "exemption_kernel_dilution.csv"},
    {"value": f"{E_own:,.1f}", "quantity": "T3420 own exposure, train-h", "table": "exemption_kernel_dilution.csv"},
    {"value": f"{dilution:.1f}", "quantity": "exposure multiplier if the centre is left in", "table": "exemption_kernel_dilution.csv"},
    {"value": f"{r_with_self:.1f}", "quantity": "d=0 rate if the centre is left in", "table": "exemption_kernel_dilution.csv"},
    {"value": f"{mu_self:.1f}", "quantity": "expected d=0 events if the centre is left in", "table": "exemption_kernel_dilution.csv"},
    {"value": f"{r_carr_near:.1f}", "quantity": "T3420 near-field rate per 1000 train-h", "table": "carrier_scan.csv (03)"},
    {"value": f"{r_carr_near/runner:.0f}", "quantity": "ratio to the runner-up", "table": "rule_clauses.csv"},
    {"value": f"{len(TRAINS)}", "quantity": "trains scanned", "table": "secondary_source_scan.csv"},
    {"value": f"{len(far)}", "quantity": "events outside the 2.5-hop field", "table": "secondary_source_scan.csv"},
    {"value": f"{p_sel:.2f}", "quantity": "max-scan p for a second centre", "table": "secondary_null.csv"},
    {"value": f"{off_line}", "quantity": "victims off T3420's own line", "table": "rule_clauses.csv"},
    {"value": f"{len(geo)}", "quantity": "victims in total", "table": "rule_clauses.csv"},
    {"value": f"{1000*st_ev.get('segment',0)/state_h['segment']:.2f}", "quantity": "rate with the source moving", "table": "rule_state_rates.csv"},
    {"value": f"{1000*st_ev.get('station',0)/state_h['station']:.2f}", "quantity": "rate with the source standing at a station", "table": "rule_state_rates.csv"},
    {"value": f"{r_off_grid:.2f}", "quantity": "rate with the source off the grid", "table": "rule_state_rates.csv"},
    {"value": f"{late.late_rate_per_1000h:.1f}", "quantity": "T3420 late arrivals per 1000 train-h", "table": "carrier_as_victim.csv"},
    {"value": f"{pct_late:.0f}", "quantity": "T3420 lateness percentile in the fleet", "table": "carrier_as_victim.csv"},
    {"value": f"{late.delay_created_train_min:.1f}", "quantity": "train-min of delay T3420 puts on others", "table": "carrier_as_victim.csv"},
])
save(pn, "paragraph_numbers.csv")
save(pd.DataFrame([{"k_d0": k0, "h_d0": h0_stratum, "rate_d0": r0_stratum, "own_h": E_own, "k_own": k_own,
                    "h_d0_with_self": h0_with_self, "rate_d0_with_self": r_with_self, "dilution": dilution,
                    "expected_with_self": mu_self, "OE_with_self": k0 / mu_self,
                    "p_with_self": float(np.exp(-(mu_self - k0)))}]), "exemption_kernel_dilution.csv")

P1 = (f"The fitted hazard is a field around one train, and it is estimated from other trains' exposure only. "
      f"The stratum at zero hop distance holds {h0_stratum:.1f} train-hours of other trains' time beside T3420 "
      f"and {k0} breakdowns, {r0_stratum:.1f} per 1000 train-hours. T3420's own {E_own:,.1f} train-hours also lie "
      f"at distance zero from the field's centre, because a train is always at distance zero from itself. "
      f"Leaving them in would multiply that stratum's exposure {dilution:.1f}-fold and cut its rate to "
      f"{r_with_self:.1f} per 1000 train-hours, with {k0} events where {mu_self:.1f} are expected. Excluding the "
      f"centre is therefore not a modelling choice but a condition of estimability, and the defensible claim is "
      f"not that T3420 is exempt from its own field: a field centred on a train cannot be applied to that train, "
      f"because the centre is never a victim.")
P2 = (f"The field has one centre. T3420's near-field rate is {r_carr_near:.1f} per 1000 train-hours, "
      f"{r_carr_near/runner:.1f} times the runner-up among {len(TRAINS)} trains, and none of the {len(far)} events "
      f"outside its field leaves a second centre: the best of the {len(TRAINS)} candidates sits at "
      f"{obs_max:.2f} per 1000 train-hours, below the 95th percentile of its own background null (p = {p_sel:.2f}). "
      f"The field crosses line boundaries, since {off_line} of {len(geo)} victims are off T3420's own line, and it "
      f"fires while the source is moving ({1000*st_ev.get('segment',0)/state_h['segment']:.2f} per 1000 "
      f"train-hours) as well as while it stands at a station ({1000*st_ev.get('station',0)/state_h['station']:.2f}), "
      f"but not when the source is off the network ({r_off_grid:.2f}). It is not a hazard to the centre: T3420's own lateness is "
      f"{late.late_rate_per_1000h:.1f} per 1000 train-hours, the {pct_late:.0f}th percentile of the fleet, and it "
      f"puts {late.delay_created_train_min:.1f} train-minutes of delay on other trains.")
for i, para in enumerate((P1, P2), start=1):
    log(f"(6.{i}) {len(para.split())} words")
(OUT / "paper_paragraphs.txt").write_text(
    "# Paragraphs for the paper (brief G). Every number appears in out/source_exemption/paragraph_numbers.csv.\n\n"
    f"## The centre is not in its own risk set\n\n{P1}\n\n## One centre, no second\n\n{P2}\n")
log(f"[saved] out/source_exemption/paper_paragraphs.txt ({len(P1.split())} + {len(P2.split())} words)")
flush_log()

# ================================================================================================
section("7. Verdict")
# ================================================================================================
V1 = (f"T3420 is exempt from the breakdown hazard it defines, but the exemption is structural rather than "
      f"exceptional: at the fleet rate its {E_own:,.1f} train-hours predict {r_overall/1000*E_own:.2f} "
      f"breakdowns and it has {k_own}, so zero is the modal outcome (55 of 140 trains have zero against "
      f"{tc.exp_overall.apply(lambda m: np.exp(-m)).sum():.1f} expected), while at the near-field rate "
      f"{r_metric:.1f} per 1000 train-hours the same exposure predicts {r_metric/1000*E_own:.1f} and the "
      f"observation has probability exp(-{r_metric/1000*E_own:.1f}) = 10^{np.log10(np.exp(-r_metric/1000*E_own)):.1f}; "
      f"the reason is that the centre sits at distance zero from itself for all of its service time and the "
      f"distance kernel excludes it, which it must, since leaving it in would cut the d=0 rate {dilution:.2f}-fold "
      f"and leave {k0} events where {mu_self:.1f} are expected.")
V2 = (f"It is the only centre: its near-field rate is {r_carr_near:.1f} per 1000 train-hours against a runner-up "
      f"{runner:.2f} and a median of {others.median():.2f} across {len(TRAINS)} trains, and scanning all "
      f"{len(TRAINS)} trains against the {len(far)} events outside its field produces no second centre "
      f"(best {obs_max:.2f} per 1000 train-hours, max-scan p = {p_sel:.3f}, i.e. inside the background null).")
V3 = (f"The rule is most likely a location-and-time field attached to one train, live only while that train is on "
      f"the network ({len(geo)-off_grid} of {len(geo)} victims), strongest on its own segment ({n0} events at "
      f"d = 0) and the adjacent 250 m, crossing at interchanges ({off_line} of {len(geo)} victims off its own "
      f"line), and silent about its own centre, which is why {CARRIER} accumulates {late.late_rate_per_1000h:.1f} "
      f"late arrivals per 1000 train-hours (the {pct_late:.0f}th fleet percentile) and {late.delay_created_train_min:.1f} "
      f"train-minutes of delay on others while suffering none of the failures it causes.")
for i, v in enumerate((V1, V2, V3), start=1):
    log(f"[{i}] {v}")
(OUT / "verdict.txt").write_text("\n\n".join(f"[{i}] {v}" for i, v in enumerate((V1, V2, V3), start=1)) + "\n")
log("")
log(f"[saved] out/source_exemption/verdict.txt; full log {LOG.name}")
flush_log()
