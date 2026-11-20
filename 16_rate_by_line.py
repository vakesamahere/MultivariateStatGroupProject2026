"""Per-line breakdown rate as a function of straight-line distance to the carrier.

Purpose
-------
The metre kernel of `12_geo.py` pools all victims.  Splitting the raw event
counts by victim line gives a different apparent "peak" for each line
(Line 8 near 1 km, Line 12 and Line 2 near 1.5--2 km), which invites the
reading "each line has its own resonance".  This script asks whether those
peaks survive normalisation by exposure: it divides each line's events by the
train-hours that line actually spends in each distance shell.

Exposure source
---------------
`out/rewind_dist/bg_positions.npz`, the background sample written by
`14_rewind_dist.py`: every in-service train's position (and distance to the
carrier) every 5 min.  One row = 5 min = 1/12 h; a bin's exposure is
rows / 12 train-hours.  The sample is sparse, so the distance shell just
inside the carrier's own line (100--200 m) contains no background rows at all;
bins are therefore kept coarse (>= 500 m wide near the carrier).

Line label
----------
The background file carries no line identity, so each row is assigned the line
shared by its two nearest stations.  A train between two stations has exactly
those two as nearest, and their shared line is unique; rows where the two
nearest stations share no line (points off-network, or the Line 10/10B branch
where both labels are "10") fall back to the nearest station's unique line and
are otherwise dropped.

Outputs
-------
out/geo/rate_by_line.csv     line x distance bin: events, exposure, rate, CI, RR vs far background
out/geo/rate_by_line.txt     the same as a readable table plus the headline numbers
figures/geo/fig_rate_by_line.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTD = ROOT / "out" / "geo"
FIGD = ROOT / "figures" / "geo"
FIGD.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUTD / "rate_by_line.txt"
_LOG: list[str] = []


def log(*parts) -> None:
    text = " ".join(str(p) for p in parts)
    lines = text.splitlines()
    print("\n".join(lines[:15]) + (f"\n   ... (+{len(lines) - 15} more lines in {LOG_PATH.name})"
                                   if len(lines) > 15 else ""), flush=True)
    _LOG.append(text)


def section(title: str) -> None:
    log("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def flush_log() -> None:
    LOG_PATH.write_text("\n".join(_LOG) + "\n", encoding="utf-8")
BINS = [0, 500, 1000, 1500, 2000, 3000, 5000]
LABELS = ["0-500", "500-1k", "1-1.5k", "1.5-2k", "2-3k", "3-5k"]
CENTRES = np.array([250, 750, 1250, 1750, 2500, 4000], dtype=float)
COLOUR = {"10": "#1f77b4", "2": "#d62728", "8": "#2ca02c", "12": "#ff7f0e"}
HOURS_PER_ROW = 1.0 / 12.0  # the background grid is one sample every 5 minutes


def garwood(k: np.ndarray, hours: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact Poisson (Garwood) interval for a rate per 1000 train-hours."""
    k = np.asarray(k, dtype=float)
    hours = np.maximum(np.asarray(hours, dtype=float), 1e-12)
    lo = np.where(k > 0, chi2.ppf(0.025, 2 * np.maximum(k, 1)) / 2.0, 0.0)
    hi = chi2.ppf(0.975, 2 * k + 2) / 2.0
    return lo / hours * 1000.0, hi / hours * 1000.0


def assign_lines(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Line label per point, from the line shared by its two nearest stations."""
    sc = pd.read_csv(OUTD / "station_coords.csv")
    si = pd.read_csv(ROOT / "data" / "station_information.csv", encoding="utf-8-sig")
    lines = {s: set(g.line_id.astype(str)) for s, g in si.groupby("station_id")}

    slat = sc.lat.to_numpy(dtype=float)
    slon = sc.lon.to_numpy(dtype=float)
    sid = sc.station_id.to_numpy()
    # haversine, chunked to keep the (points x stations) matrix small
    out = np.empty(len(lat), dtype=object)
    out[:] = None
    p = np.pi / 180.0
    for start in range(0, len(lat), 200_000):
        stop = min(start + 200_000, len(lat))
        a = lat[start:stop][:, None]
        b = lon[start:stop][:, None]
        h = (np.sin((slat[None, :] - a) * p / 2) ** 2
             + np.cos(a * p) * np.cos(slat[None, :] * p)
             * np.sin((slon[None, :] - b) * p / 2) ** 2)
        near = np.argsort(h, axis=1)[:, :2]
        for i in range(near.shape[0]):
            s1, s2 = sid[near[i, 0]], sid[near[i, 1]]
            shared = lines[s1] & lines[s2]
            if len(shared) == 1:
                out[start + i] = next(iter(shared))
            elif len(lines[s1]) == 1:
                out[start + i] = next(iter(lines[s1]))
    return out.astype(str)


def main() -> None:
    section("1. Background rows within 5 km, labelled by line")
    bg = np.load(ROOT / "out" / "rewind_dist" / "bg_positions.npz", allow_pickle=True)
    dist = bg["bg_dist"]
    keep = dist <= BINS[-1]
    log(f"background rows: {dist.size:,}; within {BINS[-1]:,} m: {int(keep.sum()):,} "
        f"= {keep.sum() * HOURS_PER_ROW:,.1f} train-h")
    lat = bg["bg_lat"][keep]
    lon = bg["bg_lon"][keep]
    d = dist[keep]
    lab = assign_lines(lat, lon)
    log(f"line label resolved for {int((lab != 'None').sum()):,} rows; "
        f"unresolved (off-network or ambiguous) {int((lab == 'None').sum()):,}")
    log("rows per line: " + str(pd.Series(lab).value_counts().to_dict()))
    flush_log()

    section("2. Events with a locatable position")
    ev = pd.read_csv(OUTD / "events_geo.csv")
    n_all = len(ev)
    ev = ev[ev.d_c != 99].copy()  # carrier out of service: no distance exists
    ev["line"] = ev.line_id.astype(str)
    log(f"events: {n_all}; carrier in service (distance defined): {len(ev)}; "
        f"per line: {ev.line.value_counts().to_dict()}")
    flush_log()

    section("3. Rate by line and distance")
    rows = []
    for ln in ["10", "2", "8", "12"]:
        hb, _ = np.histogram(d[lab == ln], bins=BINS)
        he, _ = np.histogram(ev.loc[ev.line == ln, "straight_m"], bins=BINS)
        hours = hb * HOURS_PER_ROW
        lo, hi = garwood(he, hours)
        for i, name in enumerate(LABELS):
            rows.append(dict(
                line=ln, bin=name, bin_lo=BINS[i], bin_hi=BINS[i + 1],
                events=int(he[i]), exposure_h=round(float(hours[i]), 2),
                rate_per_1000h=round(float(he[i] / hours[i] * 1000), 2) if hours[i] > 0 else np.nan,
                ci_low=round(float(lo[i]), 2) if hours[i] > 0 else np.nan,
                ci_high=round(float(hi[i]), 2) if hours[i] > 0 else np.nan,
            ))
    tab = pd.DataFrame(rows)

    # far background per line: the 3-5 km shell, which the kernel fits as flat
    far = tab[tab.bin == "3-5k"].set_index("line").rate_per_1000h
    tab["rr_vs_own_far"] = tab.apply(
        lambda r: round(r.rate_per_1000h / far[r.line], 1)
        if pd.notna(r.rate_per_1000h) and far.get(r.line, 0) > 0 else np.nan, axis=1)
    pooled = (tab.groupby("bin", sort=False)[["events", "exposure_h"]].sum()
              .reindex(LABELS))
    pooled["rate_per_1000h"] = (pooled.events / pooled.exposure_h * 1000).round(1)
    tab.to_csv(OUTD / "rate_by_line.csv", index=False)
    pooled.reset_index().to_csv(OUTD / "rate_pooled.csv", index=False)

    lines: list[str] = []
    lines.append("exposure (train-h) per line and distance shell")
    lines.append(tab.pivot(index="bin", columns="line", values="exposure_h")
                 .reindex(LABELS).to_string())
    lines.append("")
    lines.append("events per line and distance shell")
    lines.append(tab.pivot(index="bin", columns="line", values="events")
                 .reindex(LABELS).to_string())
    lines.append("")
    lines.append("rate per 1000 train-h, with exact Poisson 95% CI")
    for ln in ["10", "2", "8", "12"]:
        s = tab[tab.line == ln]
        lines.append(f"  line {ln:>2}: " + " | ".join(
            f"{r.bin} {r.events}/{r.exposure_h}h={r.rate_per_1000h}"
            f" ({r.ci_low}-{r.ci_high})" for r in s.itertuples()))
    lines.append("")
    lines.append("pooled over the four lines")
    lines.append(pooled.to_string())
    log("\n".join(lines))

    section("4. Headline numbers")
    same = tab[(tab.line == "10")]
    near = same[same.bin == "0-500"].iloc[0]
    just_out = same[same.bin == "500-1k"].iloc[0]
    cross = tab[tab.line.isin(["2", "8", "12"]) & tab.bin.isin(["500-1k", "1-1.5k", "1.5-2k"])]
    cross_ev = int(cross.events.sum())
    cross_h = float(cross.exposure_h.sum())
    cross_rate = cross_ev / cross_h * 1000
    pooled_far = float(pooled.loc["3-5k", "rate_per_1000h"])
    log(f"same line, 0-500 m: {near.events} events / {near.exposure_h} h = "
        f"{near.rate_per_1000h} per 1000 h; 500 m-1 km: {just_out.events} / "
        f"{just_out.exposure_h} h = {just_out.rate_per_1000h} "
        f"-> drop of {near.rate_per_1000h / just_out.rate_per_1000h:.0f}x at 500 m")
    log(f"cross-line, 500 m-2 km: {cross_ev} events / {cross_h:.1f} h = "
        f"{cross_rate:.1f} per 1000 h; per-line range "
        f"{cross.rate_per_1000h.min():.1f}-{cross.rate_per_1000h.max():.1f}")
    log(f"pooled far background (3-5 km): {pooled_far} per 1000 h "
        f"-> cross-line plateau is {cross_rate / pooled_far:.0f}x it")
    log(f"pooled rate by shell: " + ", ".join(
        f"{b} {pooled.loc[b, 'rate_per_1000h']}" for b in LABELS))
    flush_log()

    section("5. Figure")
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.4))
    ax = axs[0]
    for ln in ["10", "2", "8", "12"]:
        s = tab[tab.line == ln]
        ax.plot(CENTRES, s.events.to_numpy(), "o-", color=COLOUR[ln],
                label=f"line {ln} (n={int(s.events.sum())})", lw=1.8)
    ax.set_xscale("log")
    ax.set_xlabel("straight-line distance to the carrier (m)")
    ax.set_ylabel("number of breakdowns")
    ax.set_title("A. Raw counts: each line appears to peak at its own distance")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    ax = axs[1]
    for ln in ["10", "2", "8", "12"]:
        s = tab[(tab.line == ln) & (tab.events > 0) & (tab.exposure_h > 0)]
        mid = (s.bin_lo.to_numpy() + s.bin_hi.to_numpy()) / 2.0
        ax.errorbar(mid, s.rate_per_1000h.to_numpy(),
                    yerr=[s.rate_per_1000h.to_numpy() - s.ci_low.to_numpy(),
                          s.ci_high.to_numpy() - s.rate_per_1000h.to_numpy()],
                    fmt="o-", color=COLOUR[ln], capsize=3, lw=1.8,
                    label=f"line {ln} (n={int(tab[tab.line == ln].events.sum())})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("straight-line distance to the carrier (m)")
    ax.set_ylabel("breakdowns per 1000 train-hours")
    ax.set_title("B. Divided by exposure: the peaks mostly dissolve")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGD / "fig_rate_by_line.png", dpi=160)
    log(f"wrote {FIGD / 'fig_rate_by_line.png'}")
    flush_log()


if __name__ == "__main__":
    main()
