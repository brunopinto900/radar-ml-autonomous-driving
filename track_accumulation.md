# Track accumulation

`track-accumulation` branch. Whiteboard and EDA findings for multi-scan accumulation, before any model changes.

## Coordinate frame

- `x_cc`/`y_cc` (car-coordinate, origin at ego rear axle) is unusable raw across scans: it shifts with the ego vehicle every measurement, so concatenating it mixes points expressed in different, shifted frames.
- `x_seq`/`y_seq` (dataset's own fixed global-sequence frame, odometry-corrected, per RadarScenes Readme) is the required minimum, not a whiteboard optimization, points from different scans aren't spatially comparable without it.
- Global-fixed and object-centered are two different, orthogonal fixes. `x_seq`/`y_seq` only removes ego motion. It does not stop the *tracked object's own* translation from smearing into the accumulated shape.
- Confirmed on a worked example (`sequence_1` car, `notebooks/track_accumulation_eda.ipynb`), position relative to the object's own centroid *at that same scan* stays bounded; relative to the object's first detection, it drifts monotonically:

| # | rel. to first detection (x, y) | rel. to that scan's own centroid (x, y) |
|---|---|---|
| 0 | 0.00, 0.00 | 0.87, 0.35 |
| 2 | -0.67, -0.43 | 0.77, 0.38 |
| 3 | -2.21, -1.18 | -0.77, -0.38 |
| 5 | -1.94, -1.23 | 1.03, -0.11 |
| 6 | -3.99, -1.01 | -1.03, 0.11 |
| 7 | -2.62, -1.56 | 0.89, 0.21 |
| 8 | -4.40, -1.98 | -0.89, -0.21 |

- Recipe: `x_seq`/`y_seq` as the base frame, then recenter every scan on *that scan's own* object centroid before pooling (per-scan version of today's single-scan `x_rel`/`y_rel`, just re-run per scan instead of once). Rotation compensation (object's own heading change) is a further, harder step, needs a heading estimate not available for free, genuinely whiteboard territory.

## range_sc

- Don't drop it. The drift in raw `range_sc` across scans is redundant with integrating `vr_compensated`, but that's not what made it valuable, last week's validated effect was the *absolute* distance at detection (a positional/class-correlated signal), not how it changes over the window.
- Naive-consistent fix: pick one well-defined value per accumulated instance (e.g. range at first detection), don't derive a new quantity, don't delete the feature.

## Motion smearing, quantified

Full lifetime of the same worked-example car, every sensor-2 detection, `notebooks/track_accumulation_eda.ipynb`:

| metric | value |
|---|---|
| scenes | 830 |
| duration | 61.74 s |
| `x_seq`/`y_seq` displacement, first to last | 501.63 m |
| implied average speed | 8.13 m/s |
| `vr_compensated` mean over track | 7.32 m/s (agrees) |

- A single accumulated point set from this track's whole lifetime, with no per-scan recentering, would span ~500 m for one object, several orders of magnitude past its actual physical size. Confirms the smearing risk isn't hypothetical.
- If a naive (no recentering) accumulation test underperforms, it won't be possible to tell "accumulation doesn't help" from "smearing wiped it out" without splitting results by track speed/duration.

## Aspect angle diversity: does accumulation expose new shape?

Method: `results/data/points_table.parquet` (sensor 2, all sequences, 1,443,816 detections). Grouped by `(sequence_name, track_id)`, kept groups with >=10 detections (4,906 of 5,930 tracks). First 10 detections per track in timestamp order, `azimuth_sc` range (max minus min), degrees.

| percentile | azimuth range (deg) |
|---|---|
| 10% | 0.93 |
| 25% | 1.36 |
| 50% (median) | 2.32 |
| 75% | 4.90 |
| 90% | 11.25 |
| 95% | 17.21 |
| max | 62.90 |

| class | median azimuth range (deg) |
|---|---|
| pedestrian | 1.83 |
| animal | 1.87 |
| other_dynamic | 1.87 |
| pedestrian_group | 2.19 |
| truck | 2.25 |
| large_vehicle | 2.37 |
| bus | 2.38 |
| train | 2.49 |
| car | 2.51 |
| bicycle | 4.14 |
| motorized_two_wheeler | 4.70 |

- Over half of all tracks sweep under ~2.3° across their first 10 detections, too little to expose new facets.
- Worked example car: ~2.0° sweep, right at the median, not a cherry-picked case.
- `two_wheeler` (bicycle and motorized) is the one clear outlier, roughly double every other class's median, the one class where genuine new-facet exposure is plausible, and also this project's most stubbornly hard class.
- `pedestrian` sweeps *less* than `car`, not more. Gait-related variation shows up in velocity/RCS, not heading (not yet quantified across more than one worked example).
- Real tail exists regardless of class: 10% of tracks exceed 11°, 5% exceed 17°.
- **Confound, unresolved as of this section**: `azimuth_sc` is sensor-coordinate (per the Readme), so it moves both when the object rotates and when the ego vehicle steers. A stationary object with the ego yawing underneath it would show a large range here with zero real rotation on the object's part. See correction below.

## Correction: removing ego-yaw contamination from azimuth range

Method: same 4,906 tracks and first-10-detections window as above. Pulled each sequence's `odometry` table (`radar_data.h5`, columns `timestamp`/`x_seq`/`y_seq`/`yaw_seq`/`vx`/`yaw_rate`, not used elsewhere in this project so far), nearest-neighbor matched `yaw_seq` to each detection's timestamp, computed `azimuth_sc + yaw_seq` (cancels ego rotation, since `azimuth_sc` is sensor-relative and `yaw_seq` is the ego's heading in the same fixed frame `x_seq`/`y_seq` uses), `np.unwrap`ped in time order before taking the range (14 of 4,906 tracks otherwise hit a spurious ~360° wraparound where the nearest-neighbor-sampled `yaw_seq` crossed a wrap boundary, a genuine artifact, not a real ego rotation, fixed by unwrapping).

| percentile | raw (deg) | corrected (deg) |
|---|---|---|
| 10% | 0.93 | 0.91 |
| 25% | 1.36 | 1.29 |
| 50% (median) | 2.32 | 2.12 |
| 75% | 4.90 | 4.28 |
| 90% | 11.25 | 9.82 |
| 95% | 17.21 | 16.04 |
| max | 62.90 | 75.75 |

| class | raw median (deg) | corrected median (deg) | delta |
|---|---|---|---|
| animal | 1.87 | 1.72 | -0.14 |
| pedestrian | 1.83 | 1.79 | -0.04 |
| other_dynamic | 1.87 | 1.89 | +0.01 |
| large_vehicle | 2.37 | 1.89 | -0.48 |
| pedestrian_group | 2.19 | 1.98 | -0.21 |
| train | 2.49 | 2.16 | -0.33 |
| truck | 2.25 | 2.20 | -0.05 |
| car | 2.51 | 2.27 | -0.24 |
| bus | 2.38 | 2.30 | -0.08 |
| bicycle | 4.14 | 3.49 | -0.65 |
| motorized_two_wheeler | 4.70 | 5.40 | +0.71 |

- The confound was real: most classes drop after correction, so some of the raw diversity was ego-steering noise, not object rotation, `large_vehicle` and `bicycle` shrink the most (-0.48°, -0.65°).
- The headline finding survives, and for `motorized_two_wheeler` specifically it gets stronger, not weaker (+0.71°, now the single highest median of any class). `two_wheeler` being the outlier class isn't an artifact of where the ego vehicle happens to be steering when it encounters one.
- Worked-example car (`sequence_1`): 2.0° raw, 1.92° corrected, barely moved, ego yaw wasn't a meaningful factor for this specific track.

## Why point-cloud density likely doesn't help (most tracks)

- Radar sparsity is a scattering-physics problem, not a sampling-density one: smooth metal panels don't return radar energy to the sensor at all, only a few dominant specular points do (corners, wheel wells). More looks at a target with 2 real reflectors won't reveal a 3rd, there's no return there to find.
- Automotive radar's actual angular/range precision is decimeter-scale (beamwidth/bandwidth limited), not millimeter. The ~0.1 to 0.25 m scan-to-scan jitter observed in the worked example's own reflector positions sits inside that noise band, consistent with measurement jitter (glint), not new structure.
- LIDAR-style accumulation relies on scan-to-scan registration (ICP or similar) before merging, exactly because coarse odometry alone isn't precise enough. Not buildable here: registration needs multiple stable point correspondences per scan, and most scans have 1 to 2 points.
- Consistent with the aspect-angle finding above: this mechanism only has room to work where aspect angle genuinely changes, a minority of tracks, concentrated somewhat in `two_wheeler`.

## Temporal signal candidate: micro-Doppler and RCS stability

One car (`sequence_1`) vs. the longest pedestrian track found (`sequence_23`, 1122 detections), first 10 detections each:

| | `vr_compensated` range | relative spread | `rcs` pattern |
|---|---|---|---|
| car | 9.9686 to 9.9892 | ~0.2% of mean | bimodal, stable: one point 17-26, one point -3 to -9, same two reflectors every scan |
| pedestrian | 0.6510 to 0.9581 | ~37% of mean | -7 to -26, no repeatable structure |

- Micro-Doppler is definitionally temporal, a limb's oscillating velocity over a gait cycle cannot be seen in any single scan regardless of point count, only across scans.
- RCS *temporal stability* (same reflector, same magnitude, scan after scan) is likewise something a single scan structurally cannot measure, "is this consistent when I look again" needs more than one look.
- Reframes the actual value of accumulation here: not bigger point cloud for shape (see above), but access to the time axis, something the current single-scan feature set and DeepReflecs' order-blind max-pool architecture were never built to use.
- A flat point-set network could still capture the aggregate version (spread of RCS/velocity over the window, order-independent). It cannot capture the actual pattern (rhythmic vs. random) without explicit sequence structure, real motivation for the transformer/Mamba direction, not just a later nice-to-have.
- One worked example per class, not yet a distribution.

## Instance definition and sliding window design

- Naive proposal (accumulate a track's points into one point set, one label, one classification) trades away today's per-scan granularity: a track detected across 50 scans goes from 50 independent (if correlated) verdicts to one all-or-nothing call.
- Resolution: keep classifying every scan, each classification draws on up to the last N accumulated scans (sliding window) instead of just the current one. Restores many independent verdicts per track while keeping the accumulation benefit.
- N is a cap, not a required size: a track with fewer than N scans just contributes what it has. No padding at the scan level, DeepReflecs sees a flat point bag regardless of scan count, and an "empty padded scan" contributes zero points, identical to not existing. Padding only becomes meaningful again for a future sequence-model architecture (transformer needs it, Mamba doesn't).
- Side effect to expect, not a blocker: consecutive per-scan predictions now overlap N-1 of N scans, so errors will cluster in runs along a track rather than scatter independently, a smeared-out version of the same all-or-nothing risk, and the same sequence-correlation mechanism `MLP_Report.md` finding 4 already identified for `two_wheeler`.
- Training must match this: generate windows of varying length (1 up to N) per track, not just complete tracks, or the model never sees the small/early-window regime it will actually be served at inference. Sampling too many overlapping window positions per long track reintroduces the same correlated-near-duplicate problem finding 4 already flagged, splits still need to group by track/sequence.
- One cheap sanity check not yet done: confirm `label_id` is constant across every scan of a track before assuming one label per accumulated instance is safe.
- Window generation has a stride knob (how far apart consecutive window start positions are within a track) that trades off two separate failure modes. Stride 1 (a window at every scan) maximizes training volume and covers every temporal phase, but an 800-scan track then contributes ~800 near-duplicate windows overlapping by N-1 scans, dominating the fold and any per-window validation metric, same mechanism as the sequence-correlation issue in `MLP_Report.md` finding 4. Stride 100 nearly eliminates that redundancy (at ~0.074 s/scan for sensor 2, 100 scans is ~7.4 s apart, enough for real decorrelation) but any track shorter than 100 scans then contributes at most one window, risking starving already-thin classes (`motorized_two_wheeler` had only 28 qualifying tracks in the aspect-angle study, the one class most plausibly helped by accumulation).
- Track-level fold grouping (already standard practice, v1.0 onward) prevents cross-split leakage regardless of stride. It does not fix the imbalance/redundancy problem above, that's a within-split effect grouping has no bearing on. The two are independent fixes for independent problems.
- Future step, not resolved now: pick N and stride from empirical statistics (per-class track length distribution, scan-to-scan decorrelation timescale) rather than a round number.
- For now: N = 10, stride = 1.

## Accumulation baseline: 6-fold validation

Settled config after a window-size sweep and a `range_sc` handling ablation (not detailed here): N = 5 (diminishing returns already set in by N = 10), stride = 1, raw `range_sc` (broadcasting the target scan's value to every pooled point vs. leaving each point's own value untouched were statistically indistinguishable at N = 5, raw is simpler). "The accumulation baseline."

Same 6 sequence-grouped val carves `deepreflecs_split_sensitivity.py` uses for the single-scan model (`select_best_split`, same `classes`/`n_seeds`/`base_random_state`, so fold identity matches bit-for-bit), val split, test untouched:

| fold | car | large_vehicle | two_wheeler | pedestrian | pedestrian_group | macro F1 |
|---|---|---|---|---|---|---|
| 0 | 0.899 | 0.773 | 0.700 | 0.761 | 0.768 | 0.781 |
| 1 | 0.919 | 0.700 | 0.883 | 0.799 | 0.827 | 0.826 |
| 2 | 0.903 | 0.848 | 0.713 | 0.829 | 0.825 | 0.824 |
| 3 | 0.899 | 0.788 | 0.662 | 0.733 | 0.771 | 0.770 |
| 4 | 0.889 | 0.793 | 0.719 | 0.823 | 0.832 | 0.811 |
| 5 | 0.904 | 0.853 | 0.659 | 0.776 | 0.751 | 0.789 |
| mean | 0.902 | 0.793 | 0.723 | 0.787 | 0.796 | 0.800 |

- Mean macro F1 0.800 (std 0.023) vs. single-scan DeepReflecs' 0.735 (std 0.025) on the identical 6 folds: 6/6 wins, ranges barely overlap (0.770 to 0.826 vs. 0.706 to 0.779). Same bar `MLP_Report.md` uses to call a result real rather than noise.
- `two_wheeler` still swings hardest fold to fold (0.659 to 0.883), the same per-fold instability the single-scan model already had. Accumulation improved its mean but did not fix that instability.

## Architecture comparison: DeepReflecs vs. histogram MLP

Same accumulated data (N = 5, raw `range_sc`), same fixed canonical split (not one of the 6 folds above, the one split used for the earlier single-run N sweep), two different architectures consuming it:

| | DeepReflecs (point set) | quantile-bin MLP (5 features, no `doppler_spread`) |
|---|---|---|
| car | 0.905 | 0.899 |
| large_vehicle | 0.782 | 0.741 |
| two_wheeler | 0.759 | 0.783 |
| pedestrian | 0.800 | 0.820 |
| pedestrian_group | 0.812 | 0.833 |
| macro F1 | 0.812 | 0.815 |

- Within 0.004 macro F1 of each other. A histogram encoding can't see per-point spatial pattern or order at all, it only sees the fraction of a window's points landing in each quantile bin. If DeepReflecs' point-set architecture were exploiting some fine-grained pattern the histogram structurally can't represent, a real gap would show up here. It doesn't.
- Leans the interpretation toward accumulation helping via a more representative aggregate summary of a sparse, noisy point cloud (denoising), not toward either architecture learning something temporal-pattern-specific.
