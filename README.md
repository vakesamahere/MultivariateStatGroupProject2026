# Shanghai Metro breakdown analysis

Analysis code for the Multivariate Analysis group project 2026, which studies train breakdowns on the Shanghai Metro using the semi-synthetic course data.

## Layout

This repository holds only the `src/` directory. The scripts locate the project root as the parent of `src/` and expect:

```
<project>/
  src/            this repository
  data/           course CSVs (breakdown_recovery, cleaning_records, normal_operations,
                  rainfall, station_information, temperature, train_information), not included
  data_derived/   caches written by 01_prepare.py
  out/            numeric results, one subfolder per script
  figures/        figures, one subfolder per script
```

The data are not included.

## Dependencies

Python 3.11.9, pandas 2.3.3, numpy 2.4.2, scipy 1.17.0, statsmodels 0.15.0, matplotlib 3.10.8, seaborn 0.13.2.

## Run order

Run `01_prepare.py` first; it builds the caches in `data_derived/`. Then run the other scripts in numeric order, e.g. `python3 src/02_screening.py` from the project root. `05_permutation.py` and `06_definitive_nulls.py` have `--stage export|permute|figure` steps; see their docstrings.

| Script | What it does |
|---|---|
| `01_prepare.py` | Data audit and derived tables |
| `02_screening.py` | Exploratory screening of breakdown drivers |
| `03_mechanism.py` | Breakdown mechanism models |
| `04_impact.py` | Operational impact of breakdowns and delay propagation |
| `05_permutation.py` | Permutation / bootstrap validation of the T3420 "carrier" proximity effect |
| `06_definitive_nulls.py` | Definitive-scale negative controls (Null A, Null B) of the carrier effect |
| `07_sensitivity.py` | Sensitivity analyses for the confirmatory hazard model and the carrier permutation null |
| `08_signed_lag.py` | Signed lag from each zone breakdown to the carrier's occupancy of the track |
| `09_interchange.py` | Decomposes the 2.5-hop carrier zone into an interchange mechanism |
| `10_victim.py` | Tests whether victims are a random draw from the exposure at risk or a fixed set of trains |
| `11_recovery.py` | How long a line stays off schedule after a breakdown, and what sets it |
| `12_geo.py` | Breakdown distance to the carrier in metres, from real station coordinates |
| `13_carrier_state.py` | Carrier state at the moment of each breakdown |
| `14_incubation.py` | Incubation scan: rewinds both trains t minutes before onset |
| `14_rewind_dist.py` | Carrier-victim distance at 9 rewind instants against the in-service background |
| `15_inc_dist.py` | Distance distributions at a few rewind instants (figure only) |
| `16_rate_by_line.py` | Per-line breakdown rate as a function of straight-line distance to the carrier |
| `17_distance_groups.py` | Tests for several distance groups versus one kernel seen through two geometries |
| `18_leadtime.py` | Lead-time geometry of the off-line channel |
| `19_support_gap.py` | Whether the empty 250-600 m band is a risk fact or a timetable fact |
| `20_burst_structure.py` | Internal structure of breakdown bursts |
| `21_source_exemption.py` | Tests whether the source (carrier) itself is exempt |
| `22_peak_anova.py` | Variance decomposition (ANOVA) of the victim-to-carrier distance |
| `23_audit_fixes.py` | Intervals for the D-claims raised in the inference audit |
