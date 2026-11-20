"""15 Distance distributions at a few rewind instants (figure only, no new statistics).

Reads the same event table and the same interpolator as src/14_incubation.py, recomputes d_both(t) at
t = 0, 5, 10, 15, 30, 60, 120, 300, 600 s, and draws
  figures/incubation/fig_dist_by_t.png   9 histograms of log10 distance, shared axes
  figures/incubation/fig_ecdf_by_t.png   the 9 empirical CDFs on one panel
  out/incubation/dist_by_t.csv           the per-event distances behind both figures

Run: python3 src/15_inc_dist.py   (~10 s)
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
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
GEO = ROOT / "out" / "geo"
OUT = ROOT / "out" / "incubation"
FIG = ROOT / "figures" / "incubation"
CARRIER = "T3420"
T_SHOW = [0, 5, 10, 15, 30, 60, 120, 300, 600]  # seconds


def load_locator():
    """Exec the part of 12_geo.py above section 5 in a scratch root, so out/geo/ is never rewritten."""
    src_path = SRC / "12_geo.py"
    text = src_path.read_text(encoding="utf-8")
    cut = text.index('section("5. Exposure on the 30-s grid')
    scratch = Path(tempfile.mkdtemp(prefix="incdist_geo_"))
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
    ns = {"__file__": str(scratch / "src" / "12_geo.py"), "__name__": "geo12_for_incdist"}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(text[:cut], str(src_path), "exec"), ns)
    shutil.rmtree(scratch)
    return ns


NS = load_locator()
locate, haversine = NS["locate"], NS["haversine"]

ev = pd.read_csv(GEO / "events_geo.csv")
rows = []
for r in ev.itertuples():
    onset = pd.Timestamp(r.onset_ts)
    day = onset.normalize()
    d = {}
    for t in T_SHOW:
        at = onset - pd.Timedelta(seconds=t)
        v, c = locate(r.train_id, day, at), locate(CARRIER, day, at)
        d[t] = np.nan if (v is None or c is None) else haversine(v["lat"], v["lon"], c["lat"], c["lon"])
    if all(np.isfinite(list(d.values()))):
        rows.append({"incident_id": r.incident_id, "group": "same_line" if str(r.line_id) == "10" else "cross_line",
                     **{f"t{t}": d[t] for t in T_SHOW}})
df = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / "dist_by_t.csv", index=False)
print(f"events kept: {len(df)} / {len(ev)}  (same_line {int((df.group == 'same_line').sum())}, "
      f"cross_line {int((df.group == 'cross_line').sum())})")

# ---- figure 1: nine histograms of log10 distance, shared axes -------------------------------------------------
bins = np.linspace(0.0, 4.6, 31)  # 1 m .. 40 km
fig, axes = plt.subplots(3, 3, figsize=(11.0, 8.4), sharex=True, sharey=True)
for ax, t in zip(axes.ravel(), T_SHOW):
    v = df[f"t{t}"].to_numpy()
    ax.hist(np.log10(np.clip(v, 1, None)), bins=bins, color="#3b6ea5", edgecolor="white", linewidth=0.5)
    med = np.median(v)
    ax.axvline(np.log10(med), color="#c0392b", lw=1.6)
    ax.set_title(f"$t$ = {t} s   median {med:,.0f} m", fontsize=10)
    ax.set_xlim(0, 4.6)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(["1 m", "10 m", "100 m", "1 km", "10 km"])
    ax.grid(axis="y", alpha=0.25)
for ax in axes[:, 0]:
    ax.set_ylabel("events")
for ax in axes[-1, :]:
    ax.set_xlabel("carrier--victim distance (log scale)")
fig.suptitle("Carrier--victim distance as both trains are rewound before onset (140 events)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(FIG / "fig_dist_by_t.png", dpi=160)
print("wrote", FIG / "fig_dist_by_t.png")

# ---- figure 2: the nine empirical CDFs on one panel ----------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.8))
cmap = plt.get_cmap("viridis")
for i, t in enumerate(T_SHOW):
    v = np.sort(df[f"t{t}"].to_numpy())
    y = np.arange(1, v.size + 1) / v.size
    a1.step(np.clip(v, 1, None), y, where="post", color=cmap(i / (len(T_SHOW) - 1)), lw=1.8, label=f"{t} s")
    a2.step(np.clip(v, 1, None), y, where="post", color=cmap(i / (len(T_SHOW) - 1)), lw=1.8)
for ax in (a1, a2):
    ax.set_xscale("log")
    ax.set_xlim(50, 4e4)
    ax.set_xlabel("carrier--victim distance (m)")
    ax.grid(alpha=0.3)
a1.set_ylabel("share of events closer than x")
a1.legend(title="rewind $t$", fontsize=8, ncol=1, frameon=False)
a2.set_xlim(50, 1200)
a2.set_title("zoom: the first kilometre", fontsize=10)
a1.set_title("all nine instants", fontsize=10)
fig.suptitle("Rewinding never moves the distribution left: t = 0 is already the tightest", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(FIG / "fig_ecdf_by_t.png", dpi=160)
print("wrote", FIG / "fig_ecdf_by_t.png")

print("\nshare of events within 1 km / median m:")
for t in T_SHOW:
    v = df[f"t{t}"].to_numpy()
    print(f"  t={t:>4} s  {np.mean(v < 1000):5.1%}  median {np.median(v):8,.0f} m  q25 {np.quantile(v, .25):8,.0f} m")
