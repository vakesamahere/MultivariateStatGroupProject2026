"""03 Breakdown mechanism (Shanghai Metro semi-synthetic data).

Explains WHY breakdowns happen, using exact time-at-location exposure from data_derived/traversals.pkl.

Sections
 1. Exact exposure: every train's time dwelling at each station and travelling each segment
    (cells: location x kind x line x hour x date).
 2. Exposure-adjusted rates per station / segment and competing location hypotheses
    (Line-10 hop distance, interchange, Line-10 interchange, 10/10B branch vs shared trunk).
 3. Space-time structure: 30-s tick grid of every train's position; scan for a "carrier" train whose
    proximity raises other trains' hazard; distance kernel; mediation of the location pattern; burst
    structure (first event, spread), contact-vs-field check, recent-station-visit case-control scan.
 4. Permutation / negative-control tests of the carrier effect (event-position permutation with the full 140-train
    max-scan null; circular whole-day time shift of the carrier's trajectory with exact exposure). Run here only as
    a PILOT (--perm / --shifts); the definitive large-scale run is deferred to the cluster via --only perm, with
    the full specification in notes/03_mechanism.md ("Deferred to the cluster").
 5. What switches the carrier on: trip-level panel (28 active vs 887 inactive carrier trips) against cleaning,
    product switch, team, deep clean, direction, trip number, hour, weather, and per-trip encounter exposure.
 6. Confirmatory discrete-time hazard model (Poisson, log-exposure offset on train x date x hour x distance cells,
    date-clustered SE). PRE-SPECIFIED PRIMARY = proximity to the carrier and cleaning-product switch; secondary
    attributes with Holm; sensitivity to the distance definition and to the largest burst.
 7. Figures: distance kernel, carrier scan, burst timelines, forest plot, null distributions, location mediation.

Outputs: figures/mechanism/*.png, figures/mechanism/tables/*.csv(.gz),
figures/mechanism/mechanism_results.txt (numeric log; interpretation in notes/03_mechanism.md).

Run: python3 src/03_mechanism.py            (~2 min, ~8.5 GB peak RAM; seeds fixed; pilot permutation counts)
     python3 src/03_mechanism.py --only perm --perm 20000 --shifts 4000   (cluster mode: sections 1-4 only)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from statsmodels.discrete.conditional_models import ConditionalLogit  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402
from statsmodels.stats.contingency_tables import StratifiedTable  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="The design matrix is rank-deficient")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DERIVED = ROOT / "data_derived"
FIG = ROOT / "figures" / "mechanism"
TAB = FIG / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
LOG_PATH = FIG / "mechanism_results.txt"  # full run; --only perm writes mechanism_results_perm.txt instead
SCREEN_EVENTS = ROOT / "figures" / "screening" / "tables" / "breakdown_events.csv"  # burst ids from 02

SEED = 20260302
N_SIM = 2000  # Monte-Carlo replicates of the section-3 goodness-of-fit test (cheap; stays local)
# Section 4 replicates are PILOT defaults here; the definitive run is specified in notes/03_mechanism.md and
# executed as: python3 src/03_mechanism.py --only perm --perm 20000 --shifts 4000
N_PERM_PILOT = 500  # event-position permutation draws (null floor 1/501)
N_SHIFT_PILOT = 100  # circular whole-day time shifts (null floor 1/101)
STUDY_START = pd.Timestamp("2026-03-02")
TICK_S = 30  # space-time grid resolution (dwell is 30 s, so every station stop occupies one tick)
NEAR_D = 1.5  # carrier scan: "near" = within 1.5 hops (see distance definition in label_distance)
ZONE_D = 2.5  # exposure zone around the carrier used for the hazard panel (no event beyond 2.5)
RECENT_TICKS = 60  # 30 min look-back for the recent-visit case-control scan
OFF = 99.0  # distance code: one of the two trains not in service


def parse_args() -> argparse.Namespace:
    """Replicate counts are CLI-driven so the same script runs as a local pilot and as the cluster job."""
    p = argparse.ArgumentParser(description="03 breakdown mechanism")
    p.add_argument("--perm", type=int, default=N_PERM_PILOT,
                   help="event-position permutation draws (default %(default)s = pilot)")
    p.add_argument("--shifts", type=int, default=N_SHIFT_PILOT,
                   help="circular time-shift replicates (default %(default)s = pilot)")
    p.add_argument("--only", choices=("all", "perm"), default="all",
                   help="'perm' = sections 1-4 only (grid + permutation tests), then exit: cluster mode")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


ARGS = parse_args()
if ARGS.only == "perm":  # cluster mode must not clobber the full local log
    LOG_PATH = FIG / "mechanism_results_perm.txt"
N_PERM = ARGS.perm
N_SHIFT = ARGS.shifts
rng = np.random.default_rng(ARGS.seed)
sns.set_theme(style="whitegrid", context="paper")
_log_lines: list[str] = []


def log(*parts) -> None:
    text = " ".join(str(p) for p in parts)
    shown = text.splitlines()  # console gets a short version; the results file gets everything
    print("\n".join(shown[:4]) + (f"\n   ... (+{len(shown) - 4} lines in results file)" if len(shown) > 4 else ""),
          flush=True)
    _log_lines.append(text)


def section(title: str) -> None:
    log("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def flush_log() -> None:
    LOG_PATH.write_text("\n".join(_log_lines) + "\n", encoding="utf-8")


def read_csv(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(DATA / name, encoding="utf-8-sig", dtype=str, **kw)


def savefig(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def poisson_ci(k, exposure, level=0.95):
    """Exact (Garwood) CI for a Poisson rate k/exposure."""
    k = np.asarray(k, dtype=float)
    a = (1 - level) / 2
    lo = np.where(k > 0, stats.chi2.ppf(a, 2 * k) / 2, 0.0)
    hi = stats.chi2.ppf(1 - a, 2 * k + 2) / 2
    return lo / exposure, hi / exposure


def add_rate(df: pd.DataFrame, k="k", h="h", per=1000.0) -> pd.DataFrame:
    df = df.copy()
    df["rate_per_1000h"] = df[k] / df[h] * per
    lo, hi = poisson_ci(df[k].values, df[h].values)
    df["lo"], df["hi"] = lo * per, hi * per
    return df


def or_row(res, term: str) -> str:
    b, (lo, hi), p = res.params[term], res.conf_int().loc[term], res.pvalues[term]
    return f"OR={np.exp(b):.2f} [{np.exp(lo):.2f}, {np.exp(hi):.2f}], p={p:.4f}"


def undirected(loc: str) -> str:
    """'S005 -> S004' and 'S004 -> S005' are the same piece of track; stations stay as they are."""
    if " -> " in loc:
        a, b = loc.split(" -> ")
        return " - ".join(sorted([a, b]))
    return loc


def loc_nodes(loc: str) -> list[str]:
    return loc.replace(" -> ", " - ").split(" - ")


# ================================================================================================
section("1. Exact exposure from traversals (time at each station / on each segment)")
# ================================================================================================
trav = pd.read_pickle(DERIVED / "traversals.pkl")
bd = pd.read_csv(DERIVED / "breakdowns.csv", parse_dates=["onset_ts", "recovery_ts", "date"])
st = read_csv("station_information.csv")
trains = read_csv("train_information.csv")
trains["service_start_date"] = pd.to_datetime(trains.service_start_date)
trains["age_years"] = (STUDY_START - trains.service_start_date).dt.days / 365.25

# Station exposure: the dwell before each departure, [dep_ts - dwell_s, dep_ts), at from_station (dwell_s is
# missing at trip origins). Segment exposure: [dep_ts, arr_ts) (includes any breakdown delay on the segment).
# Terminal layovers are excluded (no breakdown ever happens at a terminal).
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
iv["h"] = (iv.end - iv.start).dt.total_seconds() / 3600
iv["hour"] = iv.start.dt.hour
loc_u_map = {c: undirected(c) for c in iv["loc"].cat.categories}
iv["loc_u"] = iv["loc"].map(loc_u_map).astype("category")
log(f"exposure intervals: {len(iv):,}; train-h by kind:",
    iv.groupby("kind", observed=True).h.sum().round(0).to_dict(), f"total {iv.h.sum():,.0f} train-h")

bd["loc_u"] = bd.location.map(undirected)
bd["kind"] = np.where(bd.location.str.contains("->"), "segment", "station")
bd["hour"] = bd.onset_ts.dt.hour
cells = iv.groupby(["loc_u", "kind", "line_id", "hour", "date"], observed=True).h.sum().reset_index()
cells = cells[cells.h > 0]
k_cells = bd.assign(line_id=bd.line_id.astype(str)).groupby(["loc_u", "kind", "line_id", "hour", "date"]).size()
k_cells = k_cells.rename("k").reset_index()
cells["loc_u"], cells["line_id"] = cells.loc_u.astype(str), cells.line_id.astype(str)
cells = cells.merge(k_cells, on=["loc_u", "kind", "line_id", "hour", "date"], how="left").fillna({"k": 0})
log(f"exposure cells (location x kind x line x hour x date): {len(cells):,}; events matched to a cell: "
    f"{int(cells.k.sum())}/{len(bd)}")
cells.to_csv(TAB / "exposure_cells.csv.gz", index=False, compression="gzip")
lam0 = len(bd) / iv.h.sum()
log(f"overall breakdown rate {lam0 * 1000:.3f} per 1000 train-h")

# ================================================================================================
section("2. Exposure-adjusted rates per station / segment; competing location hypotheses")
# ================================================================================================
adj: dict[str, set] = {s: set() for s in st.station_id}
for a, b in zip(st.station_id, st.next_station_id):
    if isinstance(b, str):
        adj[a].add(b)
        adj[b].add(a)
nodes = sorted(adj)
nidx = {s: i for i, s in enumerate(nodes)}
HOP = np.full((len(nodes), len(nodes)), 999, dtype=np.int16)
for s in nodes:  # BFS from every station over the undirected track graph
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


def hop_to_set(loc: str, targets: set) -> int:
    return int(min(HOP[nidx[a], nidx[b]] for a in loc_nodes(loc) for b in targets)) if targets else 999


lines_at = st.groupby("station_id").line_id.apply(set)
routes_at = st.groupby("station_id").route_id.apply(set)
L10 = set(st.loc[st.line_id == "10", "station_id"])
INTER = set(lines_at[lines_at.map(len) > 1].index)
INTER10 = INTER & L10
BR_10B = {s for s in L10 if routes_at[s] & {"10", "10B"} == {"10B"}}
BR_10 = {s for s in L10 if routes_at[s] & {"10", "10B"} == {"10"}}
log("interchanges:", sorted(INTER), "| Line-10 interchanges:", sorted(INTER10))
log("10B-only branch:", sorted(BR_10B), "| route-10-only branch:", sorted(BR_10))

locs = iv.groupby(["loc_u", "kind"], observed=True).h.sum().rename("h").reset_index()
locs["loc_u"], locs["kind"] = locs.loc_u.astype(str), locs.kind.astype(str)
locs["k"] = locs.loc_u.map(bd.loc_u.value_counts()).fillna(0).astype(int)
lines_by_loc = iv.groupby("loc_u", observed=True).line_id.agg(lambda s: ",".join(sorted(set(s.astype(str)))))
locs["lines"] = locs.loc_u.map(lines_by_loc)
locs["hop_L10"] = [hop_to_set(u, L10) for u in locs.loc_u]
locs["served_L10"] = locs.lines.str.split(",").map(lambda s: "10" in s)
locs["touch_interchange"] = [any(n in INTER for n in loc_nodes(u)) for u in locs.loc_u]
locs["hop_inter10"] = [hop_to_set(u, INTER10) for u in locs.loc_u]


def zone_of(u: str, served: bool) -> str:
    ns = loc_nodes(u)
    if not served:
        return "off L10"
    if any(n in BR_10B for n in ns):
        return "10B-only branch"
    if any(n in BR_10 for n in ns):
        return "10-only branch"
    return "shared 10/10B trunk"


locs["zone10"] = [zone_of(u, s) for u, s in zip(locs.loc_u, locs.served_L10)]
locs["hop_class"] = np.where(locs.served_L10, "served by L10",
                             np.where(locs.hop_L10 == 0, "off-L10, touches L10 station",
                                      np.where(locs.hop_L10 == 1, "off-L10, 1 hop", "off-L10, >=2 hops")))
locs = add_rate(locs)
locs["p_excess"] = stats.poisson.sf(locs.k - 1, locs.h * lam0)
locs["q_bh"] = multipletests(locs.p_excess, method="fdr_bh")[1]
locs = locs.sort_values("rate_per_1000h", ascending=False)
locs.to_csv(TAB / "location_rates.csv", index=False)
log(f"{len(locs)} locations ({(locs.kind == 'station').sum()} stations, {(locs.kind == 'segment').sum()} "
    f"undirected segments); {int((locs.k > 0).sum())} with >=1 event")
log("top locations by exposure-adjusted rate (k>=3):")
log(locs[locs.k >= 3][["loc_u", "kind", "lines", "k", "h", "rate_per_1000h", "lo", "hi", "p_excess", "q_bh",
                       "hop_class"]].round(4).to_string(index=False))
log(f"locations with BH q<0.05 vs uniform rate: {int((locs.q_bh < 0.05).sum())}")

class_rows = []
for col in ["hop_class", "kind", "zone10", "touch_interchange", "hop_inter10"]:
    g = locs.groupby([col, "kind"] if col != "kind" else [col]).agg(k=("k", "sum"), h=("h", "sum"),
                                                                       n_loc=("k", "size"))
    g = add_rate(g).reset_index()
    g.insert(0, "classifier", col)
    g = g.rename(columns={col: "level"})
    class_rows.append(g)
    log(f"\nrates by {col} x kind:\n" + g.round(3).to_string(index=False))
class_tab = pd.concat(class_rows, ignore_index=True)
class_tab.to_csv(TAB / "location_class_rates.csv", index=False)

locs["log_h"] = np.log(locs.h)
glm_loc = {}
for name, f in {"null": "k ~ 1", "kind": "k ~ C(kind)", "hop_L10 class x kind": "k ~ C(hop_class) * C(kind)",
                "served_L10 x kind": "k ~ served_L10 * C(kind)", "interchange x kind": "k ~ touch_interchange * C(kind)",
                "hop to L10 interchange x kind": "k ~ C(np.minimum(hop_inter10, 3)) * C(kind)",
                "zone10 x kind": "k ~ C(zone10) * C(kind)"}.items():
    m = smf.glm(f, data=locs, family=sm.families.Poisson(), offset=locs.log_h).fit()
    glm_loc[name] = m
    log(f"location Poisson GLM {name:32s} AIC={m.aic:8.1f} deviance={m.deviance:7.1f} df={m.df_resid:.0f}")
flush_log()

# ================================================================================================
section("3. Space-time grid: is there a train whose proximity makes other trains break down?")
# ================================================================================================
LABELS = list(iv["loc"].cat.categories)  # directed segments 'A -> B' and stations 'S...'
TRAINS = list(iv.train_id.cat.categories)
DATES = np.array(sorted(iv.date.unique()))
DATE_IDX = {pd.Timestamp(d): i for i, d in enumerate(DATES)}
TRAIN_IDX = {t: i for i, t in enumerate(TRAINS)}
sec_of_day = lambda ts, day: (ts - day).dt.total_seconds().values  # noqa: E731
t_start_s = float(np.floor(sec_of_day(iv.start, iv.date).min() / 3600) * 3600)
t_end_s = float(sec_of_day(iv.end, iv.date).max())
NT = int(np.ceil((t_end_s - t_start_s) / TICK_S)) + 1


def to_tick(ts: pd.Series, day: pd.Series) -> np.ndarray:
    return np.floor((sec_of_day(ts, day) - t_start_s) / TICK_S).astype(np.int64)


# A tick is the instant t_start + i*30 s; interval [s, e) covers ticks ceil(s)..ceil(e)-1 (dwell of 30 s = 1 tick).
s_rel = (sec_of_day(iv.start, iv.date) - t_start_s) / TICK_S
e_rel = (sec_of_day(iv.end, iv.date) - t_start_s) / TICK_S
a_t, b_t = np.ceil(s_rel).astype(np.int64), np.ceil(e_rel).astype(np.int64)
n_t = np.maximum(b_t - a_t, 0)
rep = np.repeat(np.arange(len(iv)), n_t)
offs = np.arange(n_t.sum()) - np.repeat(np.cumsum(n_t) - n_t, n_t)
GRID = np.full((len(TRAINS), len(DATES), NT), -1, dtype=np.int16)  # label code or -1 (not in service)
GRID[iv.train_id.cat.codes.values[rep], iv.date.map(DATE_IDX).values[rep], a_t[rep] + offs] = \
    iv["loc"].cat.codes.values[rep]
del rep, offs, s_rel, e_rel, a_t, b_t, n_t
log(f"grid {GRID.shape} (train x date x {TICK_S}s tick from {t_start_s / 3600:.1f}h), occupied train-ticks "
    f"{int((GRID >= 0).sum()):,} = {(GRID >= 0).sum() * TICK_S / 3600:,.0f} train-h")

# Distance between two positions (network hops on the undirected track graph): station-station = hops;
# station-segment = 0.5 + hops to nearer end; segment-segment = 0 if same undirected piece of track (trains meeting
# head-on or following), else 1 + hops between nearest ends. Position code -1 (not in service) -> OFF.
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
GIDX = np.where(GRID < 0, len(LABELS), GRID)  # -1 -> sentinel row/col of LD

bd["ti"] = bd.train_id.map(TRAIN_IDX)
bd["di"] = bd.date.map(DATE_IDX)
bd["tick"] = to_tick(bd.onset_ts, bd.date)
label_ok = np.mean([LABELS[c] == l if c >= 0 else False for c, l in zip(GRID[bd.ti, bd.di, bd.tick], bd.location)])
log(f"breakdown position on grid at onset tick matches recorded location: {label_ok:.1%}")
if SCREEN_EVENTS.exists():
    ev = pd.read_csv(SCREEN_EVENTS, parse_dates=["onset"])[["train_id", "onset", "burst_id", "in_burst"]]
    bd = bd.merge(ev.rename(columns={"onset": "onset_ts"}), on=["train_id", "onset_ts"], how="left")
else:
    bd["burst_id"], bd["in_burst"] = np.nan, False
bd["in_burst"] = bd.in_burst.fillna(False).astype(bool)

# Carrier scan: for every train c, rate of OTHER trains' breakdowns while within NEAR_D of c, per 1000 train-h
# spent within NEAR_D of c (exposure counted on the grid, events at their onset tick).
scan_rows = []
for c, cname in enumerate(TRAINS):
    dc = LD[GIDX, GIDX[c][None, :, :]]  # (train, date, tick)
    near = dc <= NEAR_D
    near[c] = False
    h_near = near.sum() * TICK_S / 3600
    ev_d = dc[bd.ti.values, bd.di.values, bd.tick.values]
    k_near = int(((ev_d <= NEAR_D) & (bd.ti.values != c)).sum())
    scan_rows.append({"train_id": cname, "k_near": k_near, "h_near": h_near,
                      "own_breakdowns": int((bd.ti == c).sum())})
scan = add_rate(pd.DataFrame(scan_rows), k="k_near", h="h_near")
scan = scan.merge(trains[["train_id", "route_id", "train_type", "batch_number", "age_years"]], on="train_id")
scan = scan.sort_values("rate_per_1000h", ascending=False)
scan.to_csv(TAB / "carrier_scan.csv", index=False)
log("carrier scan (top 8):\n" + scan.head(8).round(3).to_string(index=False))
CARRIER = scan.iloc[0].train_id
ci_ = TRAIN_IDX[CARRIER]
r1, r2 = scan.iloc[0], scan.iloc[1]
log(f"carrier = {CARRIER}: {r1.rate_per_1000h:.1f} [{r1.lo:.1f}, {r1.hi:.1f}] vs runner-up {r2.train_id} "
    f"{r2.rate_per_1000h:.2f}; median over all trains {scan.rate_per_1000h.median():.2f}")
log("carrier attributes:", trains.set_index("train_id").loc[CARRIER, ["route_id", "train_type", "batch_number",
                                                                       "service_start_date"]].to_dict(),
    "| own breakdowns:", int((bd.train_id == CARRIER).sum()))
same_route = trains[trains.route_id == trains.set_index("train_id").loc[CARRIER, "route_id"]]
log(f"carrier age rank within its route: oldest of {len(same_route)}"
    if same_route.service_start_date.idxmin() == trains.index[trains.train_id == CARRIER][0] else "not oldest")

DC = LD[GIDX, GIDX[ci_][None, :, :]]  # distance of every train to the carrier
DC[ci_] = np.nan  # carrier itself excluded from all risk sets
carrier_on = GRID[ci_] >= 0  # (date, tick)
bd["d_c"] = DC[bd.ti, bd.di, bd.tick]
bd["carrier_on"] = carrier_on[bd.di, bd.tick]
flush_log()

# ---- distance kernel over all carrier service time --------------------------------------------------------------
D_LABELS = ["0", "0.5", "1", "1.5", "2", "2.5", ">=3", "carrier off"]
train_line = trains.set_index("train_id").line_id.reindex(TRAINS).values
is_l10 = (train_line == "10").astype(np.int64)
in_svc = GRID >= 0
in_svc[ci_] = False


def d_code(d: np.ndarray) -> np.ndarray:
    d = np.nan_to_num(np.asarray(d, dtype=np.float64), nan=OFF)
    return np.where(d >= OFF, 7, np.minimum(np.round(d * 2), 6)).astype(np.int64)


DCODE = d_code(DC).astype(np.int8)
l10_b = np.broadcast_to(is_l10[:, None, None], GRID.shape)
cnt = np.bincount((DCODE * 2 + l10_b)[in_svc], minlength=16).reshape(8, 2)
bd_dcode = d_code(bd.d_c.values)
bd_l10 = (bd.line_id.astype(str) == "10").astype(int).values
k_kern = np.bincount(bd_dcode * 2 + bd_l10, minlength=16).reshape(8, 2)
kern = []
for j, lab in enumerate(D_LABELS):
    for l10 in (0, 1):
        kern.append({"d_to_carrier": lab, "victim_line": "Line 10" if l10 else "Lines 2/8/12",
                     "k": int(k_kern[j, l10]), "h": cnt[j, l10] * TICK_S / 3600})
    kern.append({"d_to_carrier": lab, "victim_line": "all", "k": int(k_kern[j].sum()),
                 "h": cnt[j].sum() * TICK_S / 3600})
kern = add_rate(pd.DataFrame(kern))
kern.to_csv(TAB / "carrier_kernel.csv", index=False)
log("\nbreakdown rate by distance to carrier (all carrier service time; other trains):\n"
    + kern.round(3).to_string(index=False))
far = kern[(kern.victim_line == "all") & kern.d_to_carrier.isin([">=3", "carrier off"])]
near_all = kern[(kern.victim_line == "all") & kern.d_to_carrier.isin(["0", "0.5", "1", "1.5", "2", "2.5"])]
k_far, h_far = far.k.sum(), far.h.sum()
k_nr, h_nr = near_all.k.sum(), near_all.h.sum()
rr_zone = (k_nr / h_nr) / (k_far / h_far)
log(f"within d<={ZONE_D} of carrier: {k_nr} events / {h_nr:,.0f} train-h ({k_nr / h_nr * 1000:.1f} per 1000 h); "
    f"elsewhere/carrier off: {k_far} / {h_far:,.0f} ({k_far / h_far * 1000:.3f}); RR={rr_zone:.0f}; share of all "
    f"events in the zone {k_nr / len(bd):.1%} while zone holds {h_nr / (h_nr + h_far):.1%} of exposure")
log(f"burst events within d<={ZONE_D} of carrier: {int((bd.in_burst & (bd.d_c <= ZONE_D)).sum())}/"
    f"{int(bd.in_burst.sum())}; non-burst events: {int((~bd.in_burst & (bd.d_c <= ZONE_D)).sum())}/"
    f"{int((~bd.in_burst).sum())}")

# ---- mediation: does carrier proximity explain the location pattern? -------------------------------------------
loc_class_u = dict(zip(locs.loc_u, locs.hop_class + " / " + locs.kind))
LCLASS = sorted(set(loc_class_u.values()))
lab_class = np.array([LCLASS.index(loc_class_u[undirected(lb)]) for lb in LABELS] + [-1])
lc = lab_class[GIDX]
med_cnt = np.bincount((lc * 8 + DCODE)[in_svc], minlength=len(LCLASS) * 8).reshape(len(LCLASS), 8)
bd_lc = lab_class[GRID[bd.ti, bd.di, bd.tick]]
med_k = np.bincount(bd_lc * 8 + bd_dcode, minlength=len(LCLASS) * 8).reshape(len(LCLASS), 8)
med = pd.DataFrame([{"loc_class": LCLASS[i], "dcode": j, "zone": "near carrier" if j <= 5 else "away/off",
                     "k": int(med_k[i, j]), "h": med_cnt[i, j] * TICK_S / 3600}
                    for i in range(len(LCLASS)) for j in range(8) if med_cnt[i, j] > 0])
med["log_h"] = np.log(med.h)
ref = "off-L10, >=2 hops / segment"
fml = f"C(loc_class, Treatment('{ref}'))"
m_loc = smf.glm(f"k ~ {fml}", data=med, family=sm.families.Poisson(), offset=med.log_h).fit()
m_loc_d = smf.glm(f"k ~ {fml} + C(dcode)", data=med, family=sm.families.Poisson(), offset=med.log_h).fit()
m_d = smf.glm("k ~ C(dcode)", data=med, family=sm.families.Poisson(), offset=med.log_h).fit()
lr_loc_given_d = 2 * (m_loc_d.llf - m_d.llf)
df_ld = m_loc_d.df_model - m_d.df_model
lr_d_given_loc = 2 * (m_loc_d.llf - m_loc.llf)
df_dl = m_loc_d.df_model - m_loc.df_model
log(f"\nmediation (Poisson on location class x carrier-distance cells): LR location | distance = "
    f"{lr_loc_given_d:.1f} on {df_ld:.0f} df (p={stats.chi2.sf(lr_loc_given_d, df_ld):.2g}); LR distance | location = "
    f"{lr_d_given_loc:.1f} on {df_dl:.0f} df (p={stats.chi2.sf(lr_d_given_loc, df_dl):.2g})")
med_rr = []
for lv in LCLASS:
    if lv == ref:
        continue
    t = f"{fml}[T.{lv}]"
    row = {"loc_class": lv}
    for nm, m in [("crude", m_loc), ("adj_carrier_distance", m_loc_d)]:
        row[f"RR_{nm}"] = np.exp(m.params[t])
        row[f"lo_{nm}"], row[f"hi_{nm}"] = np.exp(m.conf_int().loc[t])
    med_rr.append(row)
med_rr = pd.DataFrame(med_rr)
log("location-class RR vs off-L10 >=2 hops segment, crude vs adjusted for carrier distance:\n"
    + med_rr.round(2).to_string(index=False))
zone_tab = add_rate(med.groupby(["loc_class", "zone"]).agg(k=("k", "sum"), h=("h", "sum"))).reset_index()
log("location class x (near carrier d<=2.5 / away or carrier off):\n" + zone_tab.round(3).to_string(index=False))
med_rr.to_csv(TAB / "location_mediation.csv", index=False)
zone_tab.to_csv(TAB / "location_by_carrier_zone.csv", index=False)

bg = bd[~(bd.d_c <= ZONE_D)].copy()
bg["loc_class"] = bg.loc_u.map(loc_class_u)
log(f"\nbackground events (carrier off or d>{ZONE_D}): {len(bg)}; carrier off service at onset: "
    f"{int((~bg.carrier_on).sum())}; in a screening burst: {int(bg.in_burst.sum())}")
log(bg[["incident_id", "train_id", "line_id", "date", "onset_ts", "location", "d_c", "loc_class", "in_burst"]]
    .to_string(index=False))
bg_cnt = med[med.zone == "away/off"].groupby("loc_class").agg(k=("k", "sum"), h=("h", "sum"))
chi = stats.chisquare(bg_cnt.k, bg_cnt.k.sum() * bg_cnt.h / bg_cnt.h.sum())
sim = rng.multinomial(int(bg_cnt.k.sum()), bg_cnt.h / bg_cnt.h.sum(), size=N_SIM)
e_ = bg_cnt.k.sum() * bg_cnt.h.values / bg_cnt.h.sum()
p_sim = (1 + np.sum(((sim - e_) ** 2 / e_).sum(1) >= chi.statistic)) / (N_SIM + 1)
log(f"background events vs exposure across location classes: chi2={chi.statistic:.1f}, Monte-Carlo p={p_sim:.3f} "
    f"(background rate {bg_cnt.k.sum() / bg_cnt.h.sum() * 1000:.3f} per 1000 train-h)")
flush_log()

# ---- kernel geometry: victim position kind x carrier position kind x node hops ---------------------------------
lab_kind = np.array([1 if " -> " in lb else 0 for lb in LABELS] + [2])  # 0 station, 1 segment, 2 off
HOPLAB = np.full((len(LABELS) + 1, len(LABELS) + 1), 9, dtype=np.int64)
for i, a in enumerate(lab_nodes):
    for j, b in enumerate(lab_nodes):
        HOPLAB[i, j] = min(int(HOP[np.ix_(a, b)].min()), 9)
SAME_SEG = np.zeros_like(HOPLAB, dtype=bool)
for i, a in enumerate(lab_nodes):
    for j, b in enumerate(lab_nodes):
        SAME_SEG[i, j] = len(a) == 2 and set(a) == set(b)
cg = GIDX[ci_][None, :, :]
zone_mask = in_svc & (DCODE <= 5)
vk, ck = lab_kind[GIDX][zone_mask], np.broadcast_to(lab_kind[cg], GRID.shape)[zone_mask]
hh = HOPLAB[GIDX, cg][zone_mask]
ss = SAME_SEG[GIDX, cg][zone_mask].astype(np.int64)
ll = l10_b[zone_mask]
key = (((vk * 3 + ck) * 10 + hh) * 2 + ss) * 2 + ll
cnt_g = np.bincount(key, minlength=3 * 3 * 10 * 2 * 2)
bz = bd[bd.d_c <= ZONE_D]
bgi, cgi = GRID[bz.ti, bz.di, bz.tick], GRID[ci_][bz.di, bz.tick]
bkey = (((lab_kind[bgi] * 3 + lab_kind[cgi]) * 10 + HOPLAB[bgi, cgi]) * 2 + SAME_SEG[bgi, cgi]) * 2 + \
    (bz.line_id.astype(str) == "10").values
k_g = np.bincount(bkey, minlength=len(cnt_g))
KN = ["station", "segment"]
geo = pd.DataFrame([{"victim_line": "Line 10" if l else "Lines 2/8/12", "victim_at": KN[v], "carrier_at": KN[c],
                     "node_hops": h, "same_segment": bool(s), "k": int(k_g[kk]), "h": cnt_g[kk] * TICK_S / 3600}
                    for v in (0, 1) for c in (0, 1) for h in range(10) for s in (0, 1) for l in (0, 1)
                    for kk in [(((v * 3 + c) * 10 + h) * 2 + s) * 2 + l] if cnt_g[kk] > 0])
geo = add_rate(geo).sort_values(["victim_line", "victim_at", "carrier_at", "node_hops", "same_segment"])
geo.to_csv(TAB / "carrier_kernel_geometry.csv", index=False)
log("\nkernel geometry inside the zone (d<=2.5): victim kind x carrier kind x min node hops:\n"
    + geo.round(2).to_string(index=False))
flush_log()

# ---- lag scan: compare each train's position now with the carrier's position L ticks later -----------------
lag_rows = []
bd_l10_mask = bd.line_id.astype(str) == "10"
for lag in [-6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 10]:
    cpos = np.roll(GIDX[ci_], -lag, axis=1)
    if lag > 0:
        cpos[:, -lag:] = len(LABELS)
    elif lag < 0:
        cpos[:, :-lag] = len(LABELS)
    dl = LD[GIDX, cpos[None, :, :]]
    dl_ev = dl[bd.ti, bd.di, bd.tick]
    for grp, lmask, emask in [("Line 10", is_l10 == 1, bd_l10_mask.values), ("Lines 2/8/12", is_l10 == 0,
                                                                               ~bd_l10_mask.values)]:
        risk = in_svc & lmask[:, None, None]
        for th in (0.0, 0.5, 1.5):
            lag_rows.append({"lag_min": lag * TICK_S / 60, "victim_line": grp, "d_max": th,
                             "k": int(((dl_ev <= th) & emask & (bd.train_id != CARRIER).values).sum()),
                             "h": float((dl[risk] <= th).sum() * TICK_S / 3600)})
    del dl
lagtab = add_rate(pd.DataFrame(lag_rows))
lagtab.to_csv(TAB / "carrier_lag_scan.csv", index=False)
log("\nlag scan (distance between a train now and the carrier lag_min later; rate per 1000 train-h):\n"
    + lagtab.pivot_table(index="lag_min", columns=["victim_line", "d_max"], values="rate_per_1000h")
    .round(1).to_string())

# ---- carrier trips and activity -------------------------------------------------------------------------------
ctr = trav[trav.train_id == CARRIER].groupby("trip_key").agg(
    date=("date", "first"), trip_no=("trip_no", "first"), direction=("direction", "first"),
    start=("dep_ts", "min"), end=("arr_ts", "max")).reset_index()
ctr["date"] = pd.to_datetime(ctr.date)
ctr = ctr.sort_values("start").reset_index(drop=True)
bz = bd[bd.d_c <= ZONE_D].copy()
bz["carrier_trip"] = [ctr.trip_key[(ctr.start <= t) & (ctr.end > t)].iloc[0]
                      if ((ctr.start <= t) & (ctr.end > t)).any() else np.nan for t in bz.onset_ts]
ctr["k_zone"] = ctr.trip_key.map(bz.carrier_trip.value_counts()).fillna(0).astype(int)
ctr["active"] = ctr.k_zone > 0
ctr["dur_min"] = (ctr.end - ctr.start).dt.total_seconds() / 60
log(f"\ncarrier trips: {len(ctr)} (duration {ctr.dur_min.min():.0f}-{ctr.dur_min.max():.0f} min); zone events inside a "
    f"carrier trip: {int(bz.carrier_trip.notna().sum())}/{len(bz)}; active trips (>=1 zone event): "
    f"{int(ctr.active.sum())} on {ctr[ctr.active].date.nunique()} dates")
log("events per active trip:", ctr[ctr.active].k_zone.value_counts().sort_index().to_dict())
bz = bz.merge(ctr[["trip_key", "start"]].rename(columns={"trip_key": "carrier_trip", "start": "trip_start"}),
              on="carrier_trip", how="left")
bz["min_into_trip"] = (bz.onset_ts - bz.trip_start).dt.total_seconds() / 60

# ---- burst structure ------------------------------------------------------------------------------------------
burst_rows, rad_rows = [], []
for bid, b in bd[bd.in_burst].groupby("burst_id"):
    b = b.sort_values("onset_ts")
    pos_e = GIDX[b.ti.values, b.di.values, b.tick.values]
    pos_c = GIDX[:, b.di.values, b.tick.values]  # every train at each onset
    dmat = np.minimum(LD[pos_c, pos_e[None, :]], 9.0)
    dmat[b.ti.values, np.arange(len(b))] = np.nan
    mean_d = np.nanmean(dmat, axis=1)
    order = np.argsort(mean_d)
    first = b.iloc[0]
    trips_b = sorted(set(bz.loc[bz.burst_id == bid, "carrier_trip"].dropna().astype(int)))
    burst_rows.append({"burst_id": bid, "date": first.date.date(), "n": len(b), "start": first.onset_ts,
                       "span_min": (b.onset_ts.max() - first.onset_ts).total_seconds() / 60,
                       "lines": ",".join(sorted(set(b.line_id.astype(str)))),
                       "nearest_train": TRAINS[order[0]], "nearest_mean_d": mean_d[order[0]],
                       "runner_up_mean_d": mean_d[order[1]], "carrier_rank": int(np.where(order == ci_)[0][0]) + 1,
                       "median_d_to_carrier": float(np.median(b.d_c)), "carrier_trips": ",".join(map(str, trips_b)),
                       "first_location": first.location,
                       "carrier_loc_at_first": LABELS[GRID[ci_, first.di, first.tick]]
                       if GRID[ci_, first.di, first.tick] >= 0 else "off"})
    for _, e in b.iloc[1:].iterrows():
        rad_rows.append({"burst_id": bid, "min_since_first": (e.onset_ts - first.onset_ts).total_seconds() / 60,
                         "hops_from_first": int(min(HOP[nidx[x], nidx[y]] for x in loc_nodes(e.location)
                                                    for y in loc_nodes(first.location))),
                         "d_to_carrier": e.d_c})
bursts = pd.DataFrame(burst_rows)
bursts.to_csv(TAB / "burst_structure.csv", index=False)
rad = pd.DataFrame(rad_rows)
log("\nburst structure:\n" + bursts.drop(columns=["start"]).round(2).to_string(index=False))
log(f"carrier is the nearest train in {int((bursts.carrier_rank == 1).sum())}/{len(bursts)} bursts; median mean distance "
    f"carrier {bursts.nearest_mean_d.median():.2f} vs runner-up {bursts.runner_up_mean_d.median():.2f}")
rho_r = stats.spearmanr(rad.min_since_first, rad.hops_from_first)
log(f"spread within bursts: hops from first event location vs minutes since first event Spearman rho="
    f"{rho_r.statistic:.2f} (p={rho_r.pvalue:.2g}, n={len(rad)}); distance to carrier stays flat: median "
    f"{rad.d_to_carrier.median():.1f}, rho with minutes = {stats.spearmanr(rad.min_since_first, rad.d_to_carrier).statistic:.2f}")
log("-> subsequent breakdowns move away from the first location at the carrier's speed, i.e. they follow a moving"
    " source rather than radiating from a fixed point" if rho_r.statistic > 0.5 else "-> no clear spread pattern")
log(f"bursts spanning two carrier trips: {int(bursts.carrier_trips.str.contains(',').sum())}; bursts with no "
    f"carrier trip: {int((bursts.carrier_trips == '').sum())}")

# ---- contact vs field: was the victim ever closer to the carrier in the 10 min before onset? -----------------
W = 20
past_min = np.array([np.nanmin(np.r_[DC[r.ti, r.di, max(r.tick - W, 0):r.tick], OFF]) for r in bz.itertuples()])
bz["past10_min_d"] = past_min
log("\nzone events: current distance to carrier x minimum distance in the previous 10 min:\n"
    + pd.crosstab(bz.d_c, bz.past10_min_d).to_string())
nonl10 = bz.line_id.astype(str) != "10"
log(f"Lines 2/8/12 victims that were closer to the carrier before onset than at onset: "
    f"{int((nonl10 & (bz.past10_min_d < bz.d_c)).sum())}/{int(nonl10.sum())} -> hazard acts at a distance "
    "(no physical contact needed)")
flush_log()

# ---- hypothesis-free trigger scan: stations visited in the 30 min before onset, cases vs matched controls ------
# Controls for each of the 143 events: trains in service at the onset tick within d<=2 of the victim's position
# (same place, same time), not themselves broken down then. Exposure = visited station s (dwelt at it or travelled
# a segment ending at it) in the previous RECENT_TICKS ticks; also "within 1.5 of the carrier". Mantel-Haenszel OR.
LAB_ST = np.zeros((len(LABELS) + 1, len(nodes)), dtype=bool)
for i, a in enumerate(lab_nodes):
    LAB_ST[i, a] = True
strata = []
for r in bd.itertuples():
    here = GIDX[:, r.di, r.tick]
    dvic = LD[here, GIDX[r.ti, r.di, r.tick]]
    ctrl = np.where((dvic <= 2.0) & (GRID[:, r.di, r.tick] >= 0))[0]
    ctrl = ctrl[(ctrl != r.ti) & (ctrl != ci_)]
    if len(ctrl) == 0:
        continue
    members = np.r_[r.ti, ctrl]
    a0 = max(r.tick - RECENT_TICKS, 0)
    win = GIDX[members, r.di, a0:r.tick]
    visited = LAB_ST[win].any(axis=1)  # (members, station)
    near_c = (DC[members, r.di, a0:r.tick] <= NEAR_D).any(axis=1)
    strata.append((visited, near_c))
log(f"\nrecent-visit case-control scan: {len(strata)} events with >=1 control; median controls per event "
    f"{np.median([len(v) - 1 for v, _ in strata]):.0f}")


def mh_or(exposed_by_stratum: list[np.ndarray]) -> tuple[float, float, float, int]:
    """Mantel-Haenszel OR (1 case : m controls per stratum) with Robins-Breslow-Greenland CI and CMH p."""
    tabs = []
    for x in exposed_by_stratum:
        a, b = int(x[0]), int(1 - x[0])
        c, d = int(x[1:].sum()), int((~x[1:]).sum())
        tabs.append(np.array([[a, b], [c, d]]))
    t = StratifiedTable(tabs)
    with np.errstate(all="ignore"):
        try:
            orr = t.oddsratio_pooled
            lo, hi = t.oddsratio_pooled_confint()
            p = t.test_null_odds().pvalue
        except (ZeroDivisionError, ValueError):
            orr, lo, hi, p = np.nan, np.nan, np.nan, np.nan
    return orr, lo, hi, p, sum(int(x[0]) for x in exposed_by_stratum)


scan_st = []
for j, s_id in enumerate(nodes):
    xs = [v[:, j] for v, _ in strata]
    n_case = sum(int(x[0]) for x in xs)
    if n_case < 5:
        continue
    orr, lo, hi, p, _ = mh_or(xs)
    scan_st.append({"exposure": f"visited {s_id}", "cases_exposed": n_case,
                    "controls_exposed_share": np.mean(np.concatenate([x[1:] for x in xs])),
                    "MH_OR": orr, "lo": lo, "hi": hi, "p": p})
orr, lo, hi, p, n_case = mh_or([c for _, c in strata])
scan_st.append({"exposure": f"within {NEAR_D} of carrier {CARRIER}", "cases_exposed": n_case,
                "controls_exposed_share": np.mean(np.concatenate([c[1:] for _, c in strata])),
                "MH_OR": orr, "lo": lo, "hi": hi, "p": p})
scan_st = pd.DataFrame(scan_st)
scan_st["q_holm"] = multipletests(scan_st.p.fillna(1), method="holm")[1]
scan_st = scan_st.sort_values("p")
scan_st.to_csv(TAB / "recent_visit_scan.csv", index=False)
log(f"top exposures in the previous {RECENT_TICKS * TICK_S // 60} min (Holm over {len(scan_st)} tests):\n"
    + scan_st.head(12).round(4).to_string(index=False))
flush_log()

# ================================================================================================
section("4. Permutation / negative-control tests of the carrier proximity effect (PILOT replicates)")
# ================================================================================================
# Statistic under test: breakdowns of OTHER trains within NEAR_D network hops of the carrier, per 1000 train-h that
# those trains spend within NEAR_D of it. Observed value 45.2 per 1000 h for T3420.
h_vec = scan.set_index("train_id").h_near.reindex(TRAINS).values.astype(float)
k_vec = scan.set_index("train_id").k_near.reindex(TRAINS).values.astype(float)
rate_vec = k_vec / h_vec * 1000
obs_carrier, obs_max = float(rate_vec[ci_]), float(rate_vec.max())
in_service = GRID >= 0
log(f"observed: carrier {CARRIER} {obs_carrier:.1f} per 1000 h-near ({int(k_vec[ci_])} events / "
    f"{h_vec[ci_]:,.0f} train-h); best of {len(TRAINS)} trains {obs_max:.1f}")
log(f"PILOT replicate counts: {N_PERM} event-position permutations, {N_SHIFT} circular shifts. The definitive run is\n"
    f"  python3 src/03_mechanism.py --only perm --perm <n_perm> --shifts <n_shift>\n"
    f"(specification, inputs and cost estimate: notes/03_mechanism.md, 'Deferred to the cluster').")

# ---- 4a permutation test: move every breakdown to a uniformly random in-service train-tick of its own date ------
# Daily event counts and the whole exposure grid are held fixed, so the test asks: is the space-time alignment of
# the events with this particular train's path more than chance? The whole 140-train scan is recomputed per draw, so
# the null also covers the search over trains (max-scan statistic).
ev_by_date = {int(d): int((bd.di == d).sum()) for d in np.unique(bd.di.values)}


def draw_cells(di: int, n: int, reps: int) -> tuple[np.ndarray, np.ndarray]:
    """reps draws of n uniformly random in-service (train, tick) cells on date di."""
    ti_out = np.empty((reps, n), np.int64)
    tk_out = np.empty((reps, n), np.int64)
    have = 0
    while have < reps * n:
        ti_try = rng.integers(0, len(TRAINS), reps * n)
        tk_try = rng.integers(0, NT, reps * n)
        sel = np.flatnonzero(in_service[ti_try, di, tk_try])[: reps * n - have]
        ti_out.ravel()[have:have + len(sel)] = ti_try[sel]
        tk_out.ravel()[have:have + len(sel)] = tk_try[sel]
        have += len(sel)
    return ti_out, tk_out


null_k = np.zeros((len(TRAINS), N_PERM))
for di, n_ev in ev_by_date.items():
    ti_all, tk_all = draw_cells(di, n_ev, N_PERM)
    null_k += (DC[:, di, tk_all] <= NEAR_D).sum(axis=2)
null_rate = null_k / h_vec[:, None] * 1000
p_zone = (1 + int((null_rate[ci_] >= obs_carrier).sum())) / (N_PERM + 1)
p_scan = (1 + int((null_rate.max(axis=0) >= obs_max).sum())) / (N_PERM + 1)
log(f"\n4a PILOT permutation of event positions ({N_PERM} draws, daily counts fixed): carrier events under the null mean "
    f"{null_k[ci_].mean():.1f} (97.5th pct {np.percentile(null_k[ci_], 97.5):.0f}) vs observed "
    f"{int(k_vec[ci_])}; p={p_zone:.4f} (1/{N_PERM + 1} floor)")
log(f"max-scan null (best of 140 trains): mean {null_rate.max(axis=0).mean():.2f}, 97.5th pct "
    f"{np.percentile(null_rate.max(axis=0), 97.5):.2f}, max {null_rate.max():.2f}; observed {obs_max:.1f} -> "
    f"p={p_scan:.4f}")
log(f"draws in which ANY train reached {obs_carrier:.1f}: {int((null_rate.max(axis=0) >= obs_carrier).sum())}/"
    f"{N_PERM}; mean number of trains per draw at >=10 per 1000 h: {float((null_rate >= 10).sum(0).mean()):.2f}")

# ---- 4b negative control: circular whole-day time shift of the carrier's own trajectory -------------------------
# Keeps the carrier's route, the trains' schedules and the clock times of the breakdowns; only the alignment between
# the carrier's path and the events is destroyed. Exposure (trains near the shifted carrier) is recomputed exactly.
near_lab = LD <= NEAR_D
car_lab = GIDX[ci_]  # (nd, NT) carrier position per tick
car_uniq = np.unique(car_lab[car_lab >= 0])
sub_map = np.full(len(LABELS) + 1, -1, np.int32)
sub_map[car_uniq] = np.arange(len(car_uniq), dtype=np.int32)
A_sub = near_lab[car_uniq].astype(np.float32)  # (n_uniq, n_labels + 1)
ticks = np.arange(NT)
N_sub = np.empty((len(DATES), len(car_uniq), NT), np.float32)
for di in range(len(DATES)):
    L = GIDX[:, di, :]
    m = L >= 0
    flat = L[m].astype(np.int64) * NT + np.broadcast_to(ticks, L.shape)[m]
    oh = np.bincount(flat, minlength=(len(LABELS) + 1) * NT).reshape(len(LABELS) + 1, NT).astype(np.float32)
    N_sub[di] = A_sub @ oh
deltas = np.r_[0, np.sort(rng.choice(np.arange(60, NT - 60), size=N_SHIFT, replace=False))]
h_sh = np.zeros(len(deltas))
ev_lab = GIDX[bd.ti.values, bd.di.values, bd.tick.values]
Cs_ev = car_lab[bd.di.values[:, None], (bd.tick.values[:, None] + deltas[None, :]) % NT]  # (n_ev, n_shift)
ok_ev = Cs_ev >= 0
k_sh = near_lab[ev_lab[:, None], np.where(ok_ev, Cs_ev, 0)].sum(axis=0).astype(float)
for di in range(len(DATES)):
    Cp = car_lab[di]
    Cs = Cp[(ticks[None, :] + deltas[:, None]) % NT]  # (n_shift, NT) carrier position after the shift
    ok = Cs >= 0
    sidx = np.where(ok, sub_map[Cs], 0)
    v = N_sub[di][sidx, ticks[None, :]]
    self_near = ok & near_lab[Cp[None, :], np.where(ok, Cs, 0)]
    h_sh += np.where(ok, v, 0.0).sum(axis=1) - self_near.sum(axis=1)
h_sh *= TICK_S / 3600  # train-ticks -> train-hours
rate_sh = k_sh / np.maximum(h_sh, 1e-9) * 1000
sh_null = deltas != 0
p_shift = (1 + int((rate_sh[sh_null] >= obs_carrier).sum())) / (int(sh_null.sum()) + 1)
log(f"\n4b PILOT circular time shift of the carrier trajectory (check at shift 0: {k_sh[0]:.0f} events / {h_sh[0]:,.0f} "
    f"h = {rate_sh[0]:.1f} per 1000 h, i.e. reproduces the observed value)")
log(f"{int(sh_null.sum())} random shifts >=30 min: rate median {np.median(rate_sh[sh_null]):.2f}, 95th pct "
    f"{np.percentile(rate_sh[sh_null], 95):.2f}, max {rate_sh[sh_null].max():.2f} per 1000 h; observed "
    f"{obs_carrier:.1f} -> p={p_shift:.4f}; shifts reaching the observed rate: "
    f"{int((rate_sh[sh_null] >= obs_carrier).sum())}/{int(sh_null.sum())}")
flush_log()
pd.DataFrame({"perm": np.arange(N_PERM), "k_carrier": null_k[ci_], "rate_carrier": null_rate[ci_],
              "max_rate": null_rate.max(axis=0)}).to_csv(TAB / "permutation_null.csv", index=False)
pd.DataFrame({"shift_ticks": deltas, "shift_min": deltas * TICK_S / 60, "k": k_sh, "h": h_sh,
              "rate_per_1000h": rate_sh}).to_csv(TAB / "carrier_time_shift_null.csv", index=False)
pd.DataFrame([{"test": "event_position_permutation", "replicates": N_PERM, "statistic": "carrier rate per 1000 h-near",
               "observed": obs_carrier, "p_value": p_zone, "null_floor": 1 / (N_PERM + 1), "role": "pilot"},
              {"test": "event_position_permutation_max_scan", "replicates": N_PERM,
               "statistic": "best-of-140 rate per 1000 h-near", "observed": obs_max, "p_value": p_scan,
               "null_floor": 1 / (N_PERM + 1), "role": "pilot"},
              {"test": "circular_time_shift", "replicates": int(sh_null.sum()), "statistic": "carrier rate per 1000 h-near",
               "observed": obs_carrier, "p_value": p_shift, "null_floor": 1 / (int(sh_null.sum()) + 1),
               "role": "pilot"}]).to_csv(TAB / "permutation_pilot_summary.csv", index=False)
if ARGS.only == "perm":  # cluster mode: the grid build is the expensive prerequisite, sections 5-7 are not needed
    flush_log()
    print(f"\n[--only perm] sections 1-4 complete -> {LOG_PATH.relative_to(ROOT)}", flush=True)
    raise SystemExit(0)

# ================================================================================================
section("5. What switches the carrier on? Trip-level panel (915 carrier trips)")
# ================================================================================================
# The carrier itself never breaks down; only 28 of its 915 trips coincide with any nearby breakdown. Are those 28
# trips special (cleaning, product, team, weather, hour, direction, trip number) or just the trips that meet more
# trains? Exposure per trip = train-h that other trains spend within NEAR_D of the carrier during the trip.

def load_train_day() -> pd.DataFrame:
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


tdp = load_train_day()
near_pt = (DC <= NEAR_D).sum(axis=0)  # (nd, NT) number of other trains within NEAR_D of the carrier per tick
tr = ctr.copy()
tr["di"] = tr.date.map(DATE_IDX).astype(int)
tr["t0"] = to_tick(tr.start, tr.date)
tr["t1"] = np.minimum(to_tick(tr.end, tr.date), NT)
tr["h_near"] = [near_pt[di, a:b].sum() * TICK_S / 3600 for di, a, b in zip(tr.di, tr.t0, tr.t1)]
tr["hour"] = tr.start.dt.hour
tr["weekday"] = tr.date.dt.dayofweek
cd = tdp[tdp.train_id == CARRIER].set_index("date")
tr = tr.join(cd[["clean_type", "clean_team", "clean_product", "prev_product", "product_switch", "is_deep",
                 "hours_since_any_clean", "svc_days_since_deep_clean", "cleaned_same_morning", "temp_c",
                 "rain_mm"]], on="date")
tr["hot"] = (tr.temp_c > 30.0).astype(int)
tr["heavy_rain"] = (tr.rain_mm > 10.0).astype(int)
tr["active"] = tr.active.astype(int)
tr.to_csv(TAB / "carrier_trip_panel.csv", index=False)
log(f"trips {len(tr)}; active {int(tr.active.sum())} on {tr[tr.active == 1].date.nunique()} dates; events "
    f"{int(tr.k_zone.sum())}; exposure near carrier: active mean {tr[tr.active == 1].h_near.mean():.1f} train-h vs "
    f"inactive {tr[tr.active == 0].h_near.mean():.1f} (Mann-Whitney p="
    f"{stats.mannwhitneyu(tr[tr.active == 1].h_near, tr[tr.active == 0].h_near).pvalue:.3g})")


def bin_test(col: str) -> dict:
    t = pd.crosstab(tr[col].astype(str), tr.active)
    if t.shape != (2, 2):
        return {"covariate": col, "test": "n/a", "stat": np.nan, "p": np.nan, "detail": ""}
    orr, p = stats.fisher_exact(t.values)
    n = t.sum(axis=1).values
    return {"covariate": col, "test": "Fisher OR", "stat": orr, "p": p,
            "detail": f"{int(t.values[1, 1])}/{int(n[1])} vs {int(t.values[0, 1])}/{int(n[0])}"}


def iqr(x: pd.Series) -> str:
    x = x.dropna()
    return f"{x.median():.2f} [{x.quantile(0.25):.2f}, {x.quantile(0.75):.2f}]"


trip_tests = []
for col in ["product_switch", "is_deep", "cleaned_same_morning", "hot", "heavy_rain"]:
    trip_tests.append(bin_test(col))
for col in ["clean_type", "clean_team", "clean_product", "prev_product", "direction"]:
    t = pd.crosstab(tr[col].astype(str), tr.active)
    if min(t.shape) < 2:
        continue
    chi2, p, dof, _ = stats.chi2_contingency(t)
    trip_tests.append({"covariate": col, "test": f"chi2 ({t.shape[0]}x2)", "stat": chi2, "p": p,
                       "detail": "levels " + ",".join(t.index)})
for col in ["h_near", "hours_since_any_clean", "svc_days_since_deep_clean", "temp_c", "rain_mm", "hour",
            "trip_no", "weekday"]:
    a, b = tr.loc[tr.active == 1, col], tr.loc[tr.active == 0, col]
    trip_tests.append({"covariate": col, "test": "Mann-Whitney", "stat": np.nan,
                       "p": stats.mannwhitneyu(a.dropna(), b.dropna()).pvalue,
                       "detail": f"active {iqr(a)} vs inactive {iqr(b)}"})
trip_tests = pd.DataFrame(trip_tests)
trip_tests["q_bh"] = multipletests(trip_tests.p.fillna(1), method="fdr_bh")[1]
trip_tests = trip_tests[["covariate", "test", "stat", "p", "q_bh", "detail"]]
trip_tests.to_csv(TAB / "carrier_trip_tests.csv", index=False)
log("\nactivation vs covariates (28 active trips):\n" + trip_tests.round(3).to_string(index=False))

act_cols = ["product_switch", "clean_product", "prev_product", "clean_team", "is_deep", "direction",
            "trip_no", "hour", "temp_c", "rain_mm", "h_near"]
tm = tr.dropna(subset=act_cols).query("h_near > 0").reset_index(drop=True)
log(f"activation models: {len(tm)} of {len(tr)} trips usable ({len(tr) - len(tm)} dropped for zero encounter "
    f"exposure or a missing cleaning record); active among them {int(tm.active.sum())}")
hq = tr.loc[tr.h_near > 0].copy()
hq["h_q"] = pd.qcut(hq.h_near, 4, labels=["Q1", "Q2", "Q3", "Q4"])
exq = hq.groupby("h_q", observed=True).agg(trips=("trip_key", "size"), active=("active", "sum"),
                                           events=("k_zone", "sum"), h=("h_near", "sum"))
exq["rate_per_1000h"] = exq.events / exq.h * 1000
exq.to_csv(TAB / "carrier_trip_exposure_quartiles.csv")
log("\nencounter exposure vs events (quartiles of train-h within NEAR_D per trip):\n"
    + exq.assign(h=lambda d: d.h.round(0)).round(2).to_string())
m_exp = smf.glm("k_zone ~ C(h_q)", data=hq, family=sm.families.Poisson(),
                offset=np.log(hq.h_near)).fit(cov_type="cluster", cov_kwds={"groups": hq.date})
log(f"rate by exposure quartile, joint test of Q1=Q2=Q3=Q4: chi2={m_exp.wald_test_terms().table.loc['C(h_q)', 'statistic']:.1f} "
    f"p={m_exp.wald_test_terms().table.loc['C(h_q)', 'pvalue']:.3g}")
m_act = smf.glm("k_zone ~ product_switch + C(clean_product) + C(prev_product) + C(clean_team) + is_deep + "
                "C(direction) + trip_no + hour + temp_c + rain_mm", data=tm, family=sm.families.Poisson(),
                offset=np.log(tm.h_near)).fit(cov_type="cluster", cov_kwds={"groups": tm.date})
act_tab = pd.DataFrame({"term": m_act.params.index, "RR": np.exp(m_act.params.values),
                        "lo": np.exp(m_act.conf_int()[0].values), "hi": np.exp(m_act.conf_int()[1].values),
                        "p": m_act.pvalues.values})
act_tab.to_csv(TAB / "carrier_trip_model.csv", index=False)
log("multivariable Poisson with log exposure offset, clustered by date (activation model):\n"
    + act_tab.head(10).round(3).to_string(index=False))
log(f"activation model: product_switch RR={act_tab.set_index('term').loc['product_switch', 'RR']:.2f} "
    f"(p={act_tab.set_index('term').loc['product_switch', 'p']:.3f}); terms with p<0.05: "
    f"{', '.join(act_tab.loc[(act_tab.p < 0.05) & (act_tab.term != 'Intercept'), 'term']) or 'none'}")
flush_log()

# ================================================================================================
section("6. Confirmatory discrete-time hazard model (Poisson, log-exposure offset, date-clustered SE)")
# ================================================================================================
# Unit of analysis: (train, date, hour, distance-to-carrier category) cells built on the 30-s position grid, with
# log(train-hours in the cell) as offset. Two PRE-SPECIFIED PRIMARY hypotheses: (i) proximity to T3420, (ii) cleaning
# product switch vs the previous service day. Everything else is secondary and Holm-adjusted within that family.
HOUR_TICK = ((t_start_s + np.arange(NT) * TICK_S) // 3600).astype(np.int64)
ND = len(DATES)
DCAT = np.array(["d0", "d0p5", "d1", "d1p5", "d2", "d2p5", "d3plus", "off"])
TD_COLS = ["clean_product", "prev_product", "clean_team", "product_switch", "is_deep", "hours_since_any_clean",
           "temp_c", "rain_mm", "train_type", "batch_number", "age_years"]
BASE_TERMS = ["product_switch", "C(clean_product)", "C(prev_product)", "C(clean_team)", "is_deep", "age5",
              "C(train_type)", "C(batch_number)", "temp10", "rain10", "C(hour)"]


def exposure_counts(code_of_date, nlev: int) -> np.ndarray:
    """train-hours by (train, date, hour, distance code) from the position grid; the carrier's own ticks excluded."""
    out = np.zeros((len(TRAINS), ND, 24, nlev), np.int64)
    for di in range(ND):
        svc = GRID[:, di, :] >= 0
        svc[ci_] = False
        key = (np.arange(len(TRAINS))[:, None] * 24 + HOUR_TICK[None, :]) * nlev + code_of_date(di)
        out[:, di] = np.bincount(key[svc], minlength=len(TRAINS) * 24 * nlev).reshape(len(TRAINS), 24, nlev)
    return out


def event_counts(code_ev: np.ndarray, nlev: int, mask: np.ndarray | None = None) -> np.ndarray:
    m = np.ones(len(bd), bool) if mask is None else mask
    key = ((bd.ti.values * ND + bd.di.values) * 24 + bd.hour.values) * nlev + code_ev
    return np.bincount(key[m], minlength=len(TRAINS) * ND * 24 * nlev).reshape(len(TRAINS), ND, 24, nlev)


td = (tdp.set_index(["train_id", "date"])
      .reindex(pd.MultiIndex.from_product([TRAINS, DATES], names=["train_id", "date"])).reset_index()
      .merge(trains[["train_id", "train_type", "batch_number", "age_years"]], on="train_id", how="left"))


def cells_frame(exp: np.ndarray, kk: np.ndarray, cats: np.ndarray) -> pd.DataFrame:
    idx = np.flatnonzero(exp.ravel() > 0)
    ti_, di_, hh_, dc_ = np.unravel_index(idx, exp.shape)
    df = pd.DataFrame({"ti": ti_, "di": di_, "hour": hh_.astype(int), "k": kk.ravel()[idx],
                       "h": exp.ravel()[idx] * TICK_S / 3600})
    df["dcat"] = pd.Categorical(cats[dc_], categories=list(cats))
    pos = df.ti.values * ND + df.di.values
    for c in TD_COLS:
        df[c] = td[c].values[pos]
    df["date"] = DATES[df.di.values]
    df["age5"] = df.age_years / 5
    df["temp10"] = df.temp_c / 10
    df["rain10"] = df.rain_mm / 10
    return df[df.product_switch.notna() & (df.h > 0)].copy()


def fit_hazard(exp: np.ndarray, kk: np.ndarray, cats: np.ndarray, ref: str, label: str,
               drop: tuple[str, ...] = ()) -> tuple[object, pd.DataFrame]:
    df = cells_frame(exp, kk, cats)
    prox = f"C(dcat, Treatment(reference='{ref}'))"
    terms = [t for t in BASE_TERMS if t not in drop]
    log(f"fitting {label} ({len(df):,} cells, {df.dcat.nunique()} distance levels: "
        f"{dict(df.dcat.value_counts().sort_index())})")
    m = smf.glm("k ~ " + " + ".join([prox] + terms), data=df, family=sm.families.Poisson(),
                offset=np.log(df.h)).fit(cov_type="cluster", cov_kwds={"groups": df.date})
    m0 = smf.glm("k ~ " + " + ".join(terms), data=df, family=sm.families.Poisson(),
                 offset=np.log(df.h)).fit(cov_type="cluster", cov_kwds={"groups": df.date})
    lr, dfd = 2 * (m.llf - m0.llf), m.df_model - m0.df_model
    log(f"\n{label}: {len(df):,} cells, {int(df.k.sum())} events, {df.h.sum():,.0f} train-h; proximity LR "
        f"{lr:.1f} on {dfd:.0f} df, p={stats.chi2.sf(lr, dfd):.2g}")
    return m, df


exp8 = exposure_counts(lambda di: DCODE[:, di, :].astype(np.int64), 8)
k8 = event_counts(bd_dcode, 8)
m_main, df_main = fit_hazard(exp8, k8, DCAT, "d3plus", "PRIMARY: instantaneous network-hop distance")
PRETTY = {"product_switch": "Cleaning product switch", "is_deep": "Deep clean that morning",
          "age5": "Age (per 5 years)", "temp10": "Temperature (per 10 C)", "rain10": "Rainfall (per 10 mm)"}
VAR_NAMES = {"clean_product": "Cleaning product", "prev_product": "Previous product", "clean_team": "Cleaning team",
             "train_type": "Train type", "batch_number": "Batch", "hour": "Hour"}
DIST_NAMES = {"d0": "distance 0 (same segment)", "d0p5": "distance 0.5 hop", "d1": "distance 1 hop",
              "d1p5": "distance 1.5 hops", "d2": "distance 2 hops", "d2p5": "distance 2.5 hops",
              "off": "carrier out of service", "near<=1.5": "within 1.5 hops",
              "h0": "node hops 0", "h1": "node hops 1", "h2": "node hops 2", "h3plus": "node hops 3+",
              "h4": "carrier out of service"}


def pretty(term: str) -> str:
    if "dcat" in term:
        return DIST_NAMES.get(term.split("[T.")[-1].rstrip("]"), term)
    if term in PRETTY:
        return PRETTY[term]
    m = re.match(r"C\((\w+)\)\[T\.(.*)\]$", term)
    if m:
        return f"{VAR_NAMES.get(m.group(1), m.group(1))} {m.group(2)}"
    return term


ci_main = m_main.conf_int()
main_tab = pd.DataFrame([{"term": t, "RR": float(np.exp(m_main.params[t])),
                          "lo": float(np.exp(min(ci_main.loc[t, 0], 700))),
                          "hi": float(np.exp(min(ci_main.loc[t, 1], 700))), "p": float(m_main.pvalues[t])}
                         for t in m_main.params.index if t != "Intercept"])
main_tab["label"] = main_tab.term.map(pretty)
main_tab["family"] = np.where(main_tab.term.str.contains("dcat"), "primary: T3420 proximity",
                               np.where(main_tab.term == "product_switch", "primary: product switch", "secondary"))
sec = main_tab.family == "secondary"
main_tab["q_holm_secondary"] = np.nan
main_tab.loc[sec, "q_holm_secondary"] = multipletests(main_tab.loc[sec, "p"], method="holm")[1]
main_tab.to_csv(TAB / "hazard_model_main.csv", index=False)
log("primary hypotheses:\n" + main_tab[main_tab.family.str.startswith("primary")]
    [["label", "RR", "lo", "hi", "p"]].round(3).to_string(index=False))
sw = main_tab.set_index("term").loc["product_switch"]
m_nosw, _ = fit_hazard(exp8, k8, DCAT, "d3plus", "product switch LR test", drop=("product_switch",))
lr_sw = 2 * (m_main.llf - m_nosw.llf)
log(f"product switch: RR={sw.RR:.2f} [{sw.lo:.2f}, {sw.hi:.2f}], Wald p={sw.p:.4f}, LR p="
    f"{stats.chi2.sf(lr_sw, 1):.4g}")
log("\nsecondary terms (Holm within the secondary family):\n"
    + main_tab[sec].sort_values("p").head(8)[["label", "RR", "lo", "hi", "p", "q_holm_secondary"]]
    .round(3).to_string(index=False))
log(f"secondary terms with Holm q<0.05: "
    f"{', '.join(main_tab.loc[sec & (main_tab.q_holm_secondary < 0.05), 'label']) or 'none'}")

# ---- sensitivity analyses --------------------------------------------------------------------------------------
SENT = len(LABELS)  # GIDX sentinel code for "no service at this tick" (see GIDX definition)


def hop_code_of_date(di: int) -> np.ndarray:
    """Alternative distance: integer node hops only (a head-on meeting on the same segment counts as 0 hops)."""
    cpos = GIDX[ci_, di, :]
    code = np.minimum(HOPLAB[GIDX[:, di, :], cpos[None, :]], 3)
    code[:, cpos == SENT] = 4
    return code


ev_pos = GIDX[bd.ti.values, bd.di.values, bd.tick.values]
cpos_ev = GIDX[ci_, bd.di.values, bd.tick.values]
hop_ev = np.minimum(HOPLAB[ev_pos, cpos_ev], 3)
hop_ev[cpos_ev == SENT] = 4
near_ev = (bd_dcode > 3).astype(np.int64)  # 0 = within 1.5 hops, 1 = farther / carrier off
big_bid = float(bursts.sort_values("n").iloc[-1].burst_id)
keep_ev = bd.burst_id.fillna(-1).values != big_bid
sens = []
m_s1, _ = fit_hazard(exposure_counts(lambda di: (DCODE[:, di, :] > 3).astype(np.int64), 2),
                     event_counts(near_ev, 2), np.array(["near<=1.5", "far"]), "far",
                     "S1 distance as a binary indicator (<=1.5 hops)")
m_s2, _ = fit_hazard(exposure_counts(hop_code_of_date, 5), event_counts(hop_ev, 5),
                     np.array(["h0", "h1", "h2", "h3plus", "h4"]), "h3plus",
                     "S2 alternative distance (integer node hops)")
ci2 = m_s2.conf_int()
log("S2 proximity (vs 3+ node hops):\n" + pd.DataFrame(
    [{"label": pretty(t), "RR": float(np.exp(m_s2.params[t])), "lo": float(np.exp(min(ci2.loc[t, 0], 700))),
      "hi": float(np.exp(min(ci2.loc[t, 1], 700))), "p": float(m_s2.pvalues[t])}
     for t in m_s2.params.index if "dcat" in t]).round(3).to_string(index=False))
m_s3, _ = fit_hazard(exp8, event_counts(bd_dcode, 8, keep_ev), DCAT, "d3plus",
                     f"S3 excluding the largest burst ({int(big_bid)}, {int((~keep_ev).sum())} events)")
for nm, m, prox_terms in [("main", m_main, [t for t in m_main.params.index if "dcat" in t]),
                          ("S1 binary <=1.5", m_s1, [t for t in m_s1.params.index if "dcat" in t]),
                          ("S2 node hops", m_s2, [t for t in m_s2.params.index if "dcat" in t]),
                          ("S3 drop largest burst", m_s3, [t for t in m_s3.params.index if "dcat" in t])]:
    ci = m.conf_int()
    row = {"model": nm, "product_switch_RR": np.exp(m.params["product_switch"]),
           "product_switch_lo": np.exp(ci.loc["product_switch", 0]),
           "product_switch_hi": np.exp(ci.loc["product_switch", 1]),
           "product_switch_p": m.pvalues["product_switch"],
           "nearest_level_RR": np.exp(m.params[prox_terms[0]]),
           "nearest_level_lo": np.exp(ci.loc[prox_terms[0], 0]),
           "nearest_level_hi": np.exp(ci.loc[prox_terms[0], 1]),
           "nearest_level": pretty(prox_terms[0])}
    sens.append(row)
sens = pd.DataFrame(sens)
sens.to_csv(TAB / "hazard_sensitivity.csv", index=False)
log("\nsensitivity analyses:\n" + sens.round(3).to_string(index=False))
flush_log()

# ================================================================================================
section("7. Figures")
# ================================================================================================
# F1 distance kernel
fig, ax = plt.subplots(figsize=(6.6, 4.2))
for nm, sty, col in [("all", "o-", "C0"), ("Line 10", "s--", "C1"), ("Lines 2/8/12", "^:", "C2")]:
    s = kern[kern.victim_line == nm]
    y = s.rate_per_1000h.values
    ok = y > 0
    ax.errorbar(np.arange(len(s))[ok], y[ok], yerr=[y[ok] - s.lo.values[ok], s.hi.values[ok] - y[ok]], fmt=sty,
                color=col, label=nm, capsize=3, ms=5)
ax.set_yscale("log")
ax.set_xticks(np.arange(len(D_LABELS)))
ax.set_xticklabels(D_LABELS, rotation=25)
ax.set_xlabel(f"network-hop distance to {CARRIER}")
ax.set_ylabel("breakdowns per 1000 train-h")
ax.legend(title="victim")
ax.set_title("Hazard of other trains vs distance to the carrier")
savefig(fig, "fig1_distance_kernel.png")

# F2 carrier scan across all 140 trains
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
s = scan.reset_index(drop=True)
axes[0].plot(np.arange(len(s)), s.rate_per_1000h, "o", ms=3, color="C0")
axes[0].set_yscale("log")
axes[0].annotate(f"{CARRIER}", (0, s.rate_per_1000h.iloc[0]), xytext=(12, -4), textcoords="offset points",
                 color="C3", fontweight="bold")
axes[0].set_xlabel("train rank (140 trains, same statistic for each)")
axes[0].set_ylabel("breakdowns of other trains\nper 1000 h within 1.5 hops")
axes[0].set_title("Carrier scan")
routes = sorted(s.route_id.unique())
for i, r in enumerate(routes):
    sub = s[s.route_id == r]
    axes[1].scatter(sub.age_years, sub.rate_per_1000h, s=16, color=f"C{i}", label=f"route {r}")
axes[1].set_yscale("log")
axes[1].set_xlabel("train age at study start (years)")
axes[1].set_ylabel("proximity-hazard score (per 1000 h)")
axes[1].legend(fontsize=7, ncol=2)
axes[1].set_title("Score vs age (the carrier is the oldest route-10 train)")
savefig(fig, "fig2_carrier_scan.png")

# F3 burst timelines: breakdowns and the carrier's own path
nonl10_bursts = bd[bd.in_burst & (bd.line_id.astype(str) != "10")].groupby("burst_id").size()
bid2 = int(nonl10_bursts.idxmax()) if len(nonl10_bursts) else int(big_bid)
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
for ax, bid in zip(axes, [int(big_bid), bid2]):
    bb = bd[bd.burst_id == bid].sort_values("onset_ts")
    first = bb.iloc[0]
    di_, t0 = int(first.di), int(first.tick)
    lab0 = int(GIDX[first.ti, di_, t0])
    tt = np.arange(max(t0 - 40, 0), min(t0 + 130, NT))
    cpos = GIDX[ci_, di_, tt]
    dcar = np.where(cpos == SENT, np.nan, LD[lab0, cpos])
    ax.plot((tt - t0) * TICK_S / 60, dcar, "-", color="C3", lw=1.8, label=f"{CARRIER} (distance from first location)")
    ax.plot((bb.tick.values - t0) * TICK_S / 60,
            LD[lab0, GIDX[bb.ti.values, di_, bb.tick.values]], "o", color="C0", ms=7, label="breakdown")
    ax.axvline(0, color="0.7", lw=0.8)
    ax.set_ylim(-0.5, max(6.0, float(np.nanmax(dcar)) + 2.0))
    ax.set_xlabel("minutes from the first breakdown")
    ax.set_ylabel("network hops from the first breakdown location")
    ax.set_title(f"burst {bid}: {len(bb)} breakdowns, lines "
                 f"{','.join(sorted(set(bb.line_id.astype(str))))}", fontsize=10)
axes[0].legend(fontsize=7, loc="upper left")
savefig(fig, "fig3_burst_timelines.png")

# F4 forest plot of the confirmatory model
order = pd.concat([main_tab[main_tab.family == "primary: T3420 proximity"],
                   main_tab[main_tab.family == "primary: product switch"],
                   main_tab[main_tab.family == "secondary"].sort_values("p")]).head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.2, 6.4))
y = np.arange(len(order))
ax.errorbar(order.RR, y, xerr=[order.RR - order.lo, order.hi - order.RR], fmt="o", ms=4, color="C0", capsize=3)
prim = order.family.str.startswith("primary").values
ax.scatter(order.RR[prim], y[prim], color="C3", zorder=3, s=26)
ax.axvline(1, color="0.5", lw=0.8)
ax.set_xscale("log")
ax.set_yticks(y)
ax.set_yticklabels(order.label, fontsize=7.5)
ax.set_xlabel("rate ratio (Poisson, log-exposure offset, 95% CI clustered by date)")
ax.set_title("Confirmatory hazard model: red = pre-specified primary", fontsize=10)
savefig(fig, "fig4_hazard_forest.png")

# F5 permutation and time-shift nulls
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
axes[0].hist(null_rate.max(axis=0), bins=40, color="0.7")
axes[0].axvline(obs_max, color="C3", lw=2)
axes[0].set_xlabel("best of 140 trains: breakdowns per 1000 h-near")
axes[0].set_ylabel("permutations")
axes[0].set_title(f"event-position permutation (max-scan), p={p_scan:.3f}", fontsize=10)
axes[1].hist(rate_sh[sh_null], bins=40, color="0.7")
axes[1].axvline(obs_carrier, color="C3", lw=2)
axes[1].set_xlabel(f"breakdowns per 1000 h-near of shifted {CARRIER}")
axes[1].set_title(f"circular time shift of the carrier path, p={p_shift:.3f}", fontsize=10)
savefig(fig, "fig5_null_tests.png")

# F6 mediation: location-class rate ratios before and after adjusting for distance to the carrier
fig, ax = plt.subplots(figsize=(6.8, 4.0))
mr = med_rr.reset_index(drop=True)
y = np.arange(len(mr))
for i, r in mr.iterrows():
    ax.plot([r.RR_crude, r.RR_adj_carrier_distance], [i, i], "-", color="0.75", lw=1)
ax.plot(mr.RR_crude, y, "o", color="C0", label="crude")
ax.plot(mr.RR_adj_carrier_distance, y, "s", color="C1", label="adjusted for distance to carrier")
ax.axvline(1, color="0.5", lw=0.8)
ax.set_xscale("log")
ax.set_yticks(y)
ax.set_yticklabels(mr.loc_class, fontsize=7.5)
ax.set_xlabel("rate ratio vs off-Line-10 segments >=2 hops from Line 10")
ax.set_title("Location effect is mediated by proximity to the carrier", fontsize=10)
ax.legend(fontsize=8)
savefig(fig, "fig6_location_mediation.png")

log("figures written:")
for f in sorted(FIG.glob("*.png")):
    log(f"  {f.name} ({f.stat().st_size // 1024} kB)")
flush_log()
print(f"\nresults file: {LOG_PATH.relative_to(ROOT)}", flush=True)
