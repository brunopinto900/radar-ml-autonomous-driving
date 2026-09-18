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

The N=1→5→10 curve already saturated hard, most of the gain was captured by 5 scans of dumb pooling

## Capacity ablation: doubled CONV_DIM/POINT_DIM

Same accumulation baseline (N = 5, stride = 1, raw `range_sc`), same 6 folds, only `CONV_DIM`/`POINT_DIM` doubled (16/32 to 32/64), to check whether the baseline model is under-capacity for the richer windowed input distribution.

| fold | baseline (16/32) | doubled (32/64) | delta |
|---|---|---|---|
| 0 | 0.781 | 0.788 | +0.008 |
| 1 | 0.826 | 0.833 | +0.008 |
| 2 | 0.824 | 0.828 | +0.004 |
| 3 | 0.770 | 0.779 | +0.009 |
| 4 | 0.811 | 0.814 | +0.003 |
| 5 | 0.789 | 0.801 | +0.013 |
| mean | 0.800 | 0.807 | +0.007 |
| std | 0.023 | 0.022 | |

6/6 folds improve, same direction every time, so this is a real effect, not noise. But the size of it, +0.007 mean macro F1, is smaller than the fold-to-fold noise band itself (std ~0.022) and far smaller than the accumulation effect (+0.065). Capacity was a real but minor bottleneck. Not worth pursuing further, the win here was accumulation, not architecture size.

## Temporal variation feature: first test of across-scan dynamics

First attempt at giving a model access to something order-0 pooling structurally cannot see: not more points, but how a scan's own points change from one scan to the next within a window. Two engineered scalars, one for `rcs` one for `vr_compensated`, added to the quantile-bin MLP (N=5, raw `range_sc`, no `doppler_spread`) alongside the existing 5 histogram-encoded features:

- Per scan value: median of that scan's own points (mean rejected, collapses the two-reflector bimodal structure the car worked example already showed is real).
- Per window: mean absolute consecutive diff across those per-scan medians, divided by number of gaps (scans in the window minus 1), so window length at the start of a track doesn't confound the statistic.

Single canonical split (same one used for the architecture comparison above, not fold-validated):

| class | no temporal variation | with temporal variation | delta |
|---|---|---|---|
| car | 0.899 | 0.902 | +0.003 |
| large_vehicle | 0.741 | 0.748 | +0.007 |
| two_wheeler | 0.783 | 0.791 | +0.008 |
| pedestrian | 0.820 | 0.814 | -0.006 |
| pedestrian_group | 0.833 | 0.833 | 0.000 |
| macro F1 | 0.815 | 0.818 | +0.003 |

Flat. `two_wheeler` and `pedestrian_group` are the classes the erratic-vs-stable RCS/Doppler story is actually about, and neither shows a real move, `pedestrian_group` didn't budge at all. All deltas sit well inside the fold-to-fold noise band this project has consistently measured (std ~0.02 to 0.03), well short of a signal worth a 6-fold check.

Likely reason: this statistic only measures magnitude of scan to scan change, not its shape. It cannot distinguish a scatterer moving randomly from one that oscillates in a structured, class-characteristic way, both produce the same average jump size. A transition model (discretize into bins, count which bin follows which, a bigram over quantile bins instead of a scalar diff) would test the shape hypothesis directly. This result rules out the cheap magnitude-only version, not the underlying idea.

Same feature, same fixed split, retried at N=10 instead of N=5 (control run at N=10 without the feature was needed too, this pipeline had only ever been run at N=5 before):

| class | N=10 no temporal variation | N=10 with temporal variation | delta |
|---|---|---|---|
| car | 0.909 | 0.917 | +0.008 |
| large_vehicle | 0.755 | 0.778 | +0.023 |
| two_wheeler | 0.803 | 0.824 | +0.021 |
| pedestrian | 0.841 | 0.843 | +0.002 |
| pedestrian_group | 0.846 | 0.861 | +0.015 |
| macro F1 | 0.831 | 0.845 | +0.014 |

Not flat this time, roughly 4 to 5x the macro F1 movement seen at N=5, and concentrated in `large_vehicle`, `two_wheeler`, `pedestrian_group`, the classes the stability/erratic story was about, while `pedestrian` barely moves despite being the original example of erratic RCS. Likely explanation: 5 scans (4 gaps) wasn't enough window for a diff-based statistic to average over cleanly, 10 scans (9 gaps) gives a less noisy statistic and more span to see real dynamics in. Single split, not fold-validated yet, first real signal in the temporal-structure line of investigation so far.

## Shape/sign attempts: three ways to keep more than magnitude, three losses

Three follow-up attempts to give the temporal feature access to sign or shape instead of just magnitude of scan-to-scan change, all at N=10, all on top of the same 5-feature histogram, no `doppler_spread`:

- **Signed diff vector**: one column per gap (9 for N=10) per feature, signed (not absolute) consecutive diff of per-scan medians, right-aligned to the target scan, zero-padded for windows shorter than N.
- **Diff vector + window length**: the above plus one extra column, how many real scans the window has, meant to let the model tell padded columns from real ones.
- **Transition matrix**: a 4x4 bin-to-bin transition count per feature (which quantile bin the previous scan's median was in vs. the current one), normalized by number of transitions present, no padding needed since the matrix shape never depends on window length.

| variant | car | large_vehicle | two_wheeler | pedestrian | pedestrian_group | macro F1 |
|---|---|---|---|---|---|---|
| N=10 control | 0.909 | 0.755 | 0.803 | 0.841 | 0.846 | 0.831 |
| N=10 + summed absolute diff (magnitude only) | 0.917 | 0.778 | 0.824 | 0.843 | 0.861 | **0.845** |
| N=10 + transition matrix (keeps sign) | 0.911 | 0.757 | 0.827 | 0.842 | 0.858 | 0.839 |
| N=10 + signed diff vector | 0.910 | 0.758 | 0.814 | 0.845 | 0.855 | 0.836 |
| N=10 + diff vector + window length flag | 0.911 | 0.756 | 0.814 | 0.840 | 0.847 | 0.834 |

All three shape/sign-preserving variants land behind the dumb magnitude-only statistic, consistently, not scattered randomly around it. The window-length flag also didn't move the diff vector's number in the direction hoped for (0.834 vs 0.836 without it), a single scalar buried among 99 columns wasn't enough for the MLP to learn to use it as a padding indicator. Read together with the leading "denoising, not shape-learning" explanation for why accumulation helps at all (DeepReflecs vs. histogram MLP tie, `Architecture comparison` section above), this argues against an RNN or transformer paying off here: three independent attempts to expose order/shape found nothing beyond what magnitude alone already gave, and the data (sparse per-scan point counts, thin track counts for the rarer classes) is a plausible reason there isn't much more to find, not just bad luck.

## Pushing N further

Base accumulation for the histogram MLP hadn't actually saturated by N=10 the way the DeepReflecs point-set sweep suggested; it keeps climbing at N=20:

| | N=10 | N=20 |
|---|---|---|
| control | 0.831 | 0.852 |
| + temporal variation | 0.845 | 0.859 |
| temporal variation's own delta | +0.014 | +0.007 |

| class | N=20 control | N=20 + temporal variation | delta |
|---|---|---|---|
| car | 0.920 | 0.922 | +0.002 |
| large_vehicle | 0.772 | 0.772 | 0.000 |
| two_wheeler | 0.834 | 0.851 | +0.017 |
| pedestrian | 0.859 | 0.866 | +0.007 |
| pedestrian_group | 0.875 | 0.884 | +0.009 |

Absolute macro F1 keeps improving with N (0.845 to 0.859), but the temporal feature's own marginal contribution roughly halved between N=10 and N=20, and `large_vehicle` went from the biggest single beneficiary at N=10 (+0.023) to no benefit at all at N=20. Reads like a bigger pooled window starts to implicitly carry some of what the engineered variation feature was adding on its own, the two sources of "more temporal information" overlap rather than stack cleanly. `two_wheeler` is the one class still getting its full benefit from the explicit feature at N=20. Single split, not fold-validated.

## Post-hoc probability smoothing along the track

Free addition on top of the current best model (N=10 + temporal variation), no retraining: since a track's label can't actually change mid-track (`check_label_consistency`), any scan-to-scan flip in the model's raw predictions along one track is necessarily a mistake, not a real change. Averaged each class's softmax probability over a causal trailing window of the last 5 windows within the same track (a moving-average low-pass filter over the probability trajectory), then took the argmax of the smoothed probabilities instead of the raw ones.

| class | raw | smoothed (K=5) | delta |
|---|---|---|---|
| car | 0.917 | 0.920 | +0.003 |
| large_vehicle | 0.778 | 0.783 | +0.005 |
| two_wheeler | 0.824 | 0.829 | +0.006 |
| pedestrian | 0.843 | 0.846 | +0.003 |
| pedestrian_group | 0.861 | 0.867 | +0.006 |
| macro F1 | 0.845 | 0.849 | +0.005 |

5/5 classes improve, same direction, unlike the shape/sign attempts above which scattered around the baseline. Consistent with what smoothing actually is here: not new information, denoising of flicker using a constraint (constant label per track) the model was never explicitly given. Stacks on top of whatever else this branch settles on. Single split, not fold-validated.

## Cross-sensor track handoff, not just more time on one sensor

Every experiment above used sensor 2 only (`build_points_table.py`'s `SENSOR_ID = 2` filter). RadarScenes' 4 sensors cover different, overlapping parts of the scene, and a `track_id` is a stable ground-truth identity across sensor handoffs, not per-sensor (confirmed earlier: 14/24 tracks on one checked sequence were seen by 3 different sensors over their lifetime; same-instant cross-sensor detections are essentially never seen, 1 overlap in 500k+, so this is about extending a track's real coverage over time, not enriching one instant). Verified directly that all 4 sensors share one global clock (overlapping timestamp ranges, same units, zero exact-timestamp collisions across sensors within a sequence), so a plain sort by timestamp, dropping sensor_id from the grouping, correctly interleaves multi-sensor detections in true chronological order with no change to the window-building logic itself.

Symmetric rule, same in train and inference, to avoid a train/serve mismatch: build every window from whichever sensor(s) are actually detecting that track right now, not an enriched multi-sensor union at train time against a sensor-2-only reality at inference. `build_and_save_points_table(sensor_id=None)` keeps every sensor's points instead of filtering to one.

N=10 + temporal variation, same split, same config, sensor-2-only vs all-sensor:

| class | sensor-2 only | all-sensor (handoff) | delta |
|---|---|---|---|
| car | 0.917 | 0.925 | +0.008 |
| large_vehicle | 0.778 | 0.769 | -0.009 |
| two_wheeler | 0.824 | 0.831 | +0.007 |
| pedestrian | 0.843 | 0.883 | +0.040 |
| pedestrian_group | 0.861 | 0.861 | 0.000 |
| macro F1 | 0.845 | 0.854 | +0.009 |

Real gain, and from a different axis than N: more real detections per track through sensor handoffs, not more real time spanned on one sensor. Test-set support roughly 2.5x larger (car support 76,572 vs 29,958), confirming tracks are genuinely picking up far more real detections, not just being padded out. `pedestrian` jumped hard (+0.040), `large_vehicle` dropped slightly (-0.009), not a uniform win across classes. Reaches at N=10 what sensor-2-alone needed N=20 for (0.852 control). Single split, not fold-validated.

Pushed further, sensor-2-only N=50 + temporal variation and all-sensor N=20 + temporal variation:

| N | control | + temporal variation | delta |
|---|---|---|---|
| 5 | 0.815 | 0.818 | +0.003 |
| 10 | 0.831 | 0.845 | +0.014 |
| 20 | 0.852 | 0.859 | +0.007 |
| 50 | 0.858 | 0.867 | +0.009 |

Sensor-2-only N=50 + temporal variation reached macro F1 0.867, new best for that line, and not a clean monotonic shrink of the temporal feature's own delta (+0.014 at N=10, +0.007 at N=20, back up to +0.009 at N=50), more consistent with single-split noise on that specific number than a real trend. Every class improved from N=20 to N=50 with the feature included though, first time that's been true across the board (`large_vehicle` had been flat between N=10 and N=20, +0.012 here).

All-sensor N=20 + temporal variation:

| class | sensor-2 N=20 + temporal | all-sensor N=20 + temporal | delta |
|---|---|---|---|
| car | 0.922 | 0.936 | +0.014 |
| large_vehicle | 0.772 | 0.797 | +0.025 |
| two_wheeler | 0.851 | 0.850 | -0.001 |
| pedestrian | 0.866 | 0.897 | +0.031 |
| pedestrian_group | 0.884 | 0.874 | -0.010 |
| macro F1 | 0.859 | 0.871 | +0.012 |

Macro F1 0.871, new overall best in the branch, and cross-sensor handoff at N=20 beats pushing N to 50 on one sensor alone (0.871 vs 0.867). More real detections per track is turning out to be a stronger lever than more real time on a single sensor. Same asymmetric pattern as the N=10 comparison: `pedestrian` gains hard both times, `pedestrian_group` gives a little back both times.

Caveat that applied to both all-sensor numbers above (0.854 and 0.871): `build_windowed_temporal_features` normalized the variation statistic by number of gaps, which silently assumes every gap is the same real duration. True for sensor-2-only (~0.074s apart, always), not true once sensor handoffs mix short (multi-sensor overlap) and long (single-sensor) gaps into the same window. Fixed to normalize by total elapsed real time instead (last scan's timestamp minus first, in seconds), a no-op for sensor-2-only (gaps already uniform there) but a real change for all-sensor.

Re-ran all-sensor N=20 + temporal variation with the fix:

| class | uncorrected | delta_t corrected | delta |
|---|---|---|---|
| car | 0.936 | 0.933 | -0.003 |
| large_vehicle | 0.797 | 0.794 | -0.003 |
| two_wheeler | 0.850 | 0.841 | -0.009 |
| pedestrian | 0.897 | 0.894 | -0.003 |
| pedestrian_group | 0.874 | 0.869 | -0.005 |
| macro F1 | 0.871 | 0.866 | -0.005 |

Small but real correction, all classes down a little, `two_wheeler` the most. Changes the earlier read: corrected all-sensor N=20 (0.866) is no longer clearly ahead of sensor-2-only N=50 (0.867), they're essentially tied, not a clean win for cross-sensor over more real time. The uncorrected version was quietly overstating the cross-sensor advantage. Neither being dominant is a good sign for combining them, all-sensor N=50 + temporal variation is running next.

## Stacking both axes: all-sensor, N=50, temporal variation

Both real levers found so far (more real time on a track via larger N, more real detections per track via cross-sensor handoff) combined in one run, delta_t-corrected temporal variation, same canonical split:

| class | sensor-2 N=50 + temporal | all-sensor N=20 + temporal | all-sensor N=50 + temporal | delta vs best of the two |
|---|---|---|---|---|
| car | 0.926 | 0.933 | 0.940 | +0.007 |
| large_vehicle | 0.784 | 0.794 | 0.814 | +0.020 |
| two_wheeler | 0.858 | 0.841 | 0.853 | -0.005 |
| pedestrian | 0.876 | 0.894 | 0.905 | +0.011 |
| pedestrian_group | 0.890 | 0.869 | 0.894 | +0.004 |
| macro F1 | 0.867 | 0.866 | **0.881** | +0.014 |

The two axes stack rather than cancel or merely tie: 0.881 clears both individual bests (0.867, 0.866) by roughly the same margin either one had over the other. Only `two_wheeler` gives a little back relative to its sensor-2 N=50 number, every other class sets a new high. New overall best in the branch. Single split, not fold-validated.

## Post-hoc smoothing revisited: helps at N=10, doesn't at N=50

Same K=5 causal probability-smoothing technique as before, applied to the new best model (all-sensor N=50 + temporal variation):

| class | raw | smoothed (K=5) | delta |
|---|---|---|---|
| car | 0.940 | 0.940 | 0.000 |
| large_vehicle | 0.814 | 0.813 | -0.001 |
| two_wheeler | 0.853 | 0.851 | -0.002 |
| pedestrian | 0.905 | 0.902 | -0.003 |
| pedestrian_group | 0.894 | 0.893 | -0.001 |
| macro F1 | 0.881 | 0.880 | -0.001 |

Flat to slightly negative, the opposite of N=10 (5/5 classes improved there). Likely explanation: at N=50 with stride 1, consecutive windows already share 49 of 50 scans, so the raw probability trajectory is already heavily autocorrelated before any explicit smoothing is applied, there's little flicker left for a moving average to remove. Stacking K=5 on top mostly just risks dragging a stale window's probability across a real class transition. Reads as smoothing being useful in proportion to how noisy the raw per-window signal is, not a free win regardless of N. Not worth keeping at N=50, best number stays the raw (unsmoothed) 0.881.

## Consolidated results

DeepReflecs (point-set) family, single canonical split:

| variant | car | large_vehicle | two_wheeler | pedestrian | pedestrian_group | macro F1 |
|---|---|---|---|---|---|---|
| single-scan baseline (pre-branch, x_cc/y_cc) | 0.880 | 0.744 | 0.673 | 0.724 | 0.726 | 0.749 |
| N=1 control (this branch, x_seq/y_seq) | 0.879 | 0.752 | 0.674 | 0.723 | 0.731 | 0.752 |
| N=5, broadcast range_sc | 0.904 | 0.782 | 0.761 | 0.798 | 0.809 | 0.811 |
| N=5, raw range_sc | 0.905 | 0.782 | 0.759 | 0.800 | 0.812 | 0.812 |
| N=10 | 0.909 | 0.786 | 0.776 | 0.820 | 0.834 | 0.825 |

Quantile-bin histogram MLP family, all raw range_sc, no `doppler_spread` unless noted, same split:

| variant | car | large_vehicle | two_wheeler | pedestrian | pedestrian_group | macro F1 |
|---|---|---|---|---|---|---|
| N=5 (clean 5-feature) | 0.899 | 0.741 | 0.783 | 0.820 | 0.833 | 0.815 |
| N=5 + temporal variation | 0.902 | 0.748 | 0.791 | 0.814 | 0.833 | 0.818 |
| N=10 | 0.909 | 0.755 | 0.803 | 0.841 | 0.846 | 0.831 |
| N=10 + temporal variation | 0.917 | 0.778 | 0.824 | 0.843 | 0.861 | 0.845 |
| N=10 + transition matrix | 0.911 | 0.757 | 0.827 | 0.842 | 0.858 | 0.839 |
| N=10 + signed diff vector | 0.910 | 0.758 | 0.814 | 0.845 | 0.855 | 0.836 |
| N=10 + diff vector + window length | 0.911 | 0.756 | 0.814 | 0.840 | 0.847 | 0.834 |
| N=20 | 0.920 | 0.772 | 0.834 | 0.859 | 0.875 | 0.852 |
| N=20 + temporal variation | 0.922 | 0.772 | 0.851 | 0.866 | 0.884 | 0.859 |
| N=10 + temporal variation + post-hoc smoothing | 0.920 | 0.783 | 0.829 | 0.846 | 0.867 | 0.849 |
| N=50 | 0.918 | 0.768 | 0.837 | 0.882 | 0.887 | 0.858 |
| N=50 + temporal variation | 0.926 | 0.784 | 0.858 | 0.876 | 0.890 | 0.867 |
| N=10 + temporal variation, all-sensor (uncorrected gap normalization) | 0.925 | 0.769 | 0.831 | 0.883 | 0.861 | 0.854 |
| N=20 + temporal variation, all-sensor (uncorrected gap normalization) | 0.936 | 0.797 | 0.850 | 0.897 | 0.874 | 0.871 |
| N=20 + temporal variation, all-sensor (delta_t corrected) | 0.933 | 0.794 | 0.841 | 0.894 | 0.869 | 0.866 |
| N=50 + temporal variation, all-sensor (delta_t corrected) | 0.940 | 0.814 | 0.853 | 0.905 | 0.894 | **0.881** |
| N=50 + temporal variation, all-sensor + post-hoc smoothing (K=5) | 0.940 | 0.813 | 0.851 | 0.902 | 0.893 | 0.880 |

6-fold validated (val split, mean across folds, different rotating splits, not the canonical one above):

| variant | car | large_vehicle | two_wheeler | pedestrian | pedestrian_group | macro F1 | std |
|---|---|---|---|---|---|---|---|
| accumulation baseline, N=5 raw (16/32 capacity) | 0.902 | 0.793 | 0.723 | 0.787 | 0.796 | 0.800 | 0.023 |
| same, doubled capacity (32/64) | 0.907 | 0.793 | 0.736 | 0.794 | 0.807 | 0.807 | 0.022 |
| single-scan baseline, 6-fold (reference, pre-branch) | — | — | — | — | — | 0.735 | 0.025 |

Best single-split number in the branch is all-sensor N=50 + temporal variation at 0.881 (raw, unsmoothed), stacking both real levers found this session: more real time per track (large N) and more real detections per track (cross-sensor handoff). Post-hoc smoothing helps at N=10 (+0.005) but not at N=50 (-0.001), likely because heavily overlapping N=50 windows are already autocorrelated before any smoothing. Nothing here is fold-validated. Everything from the shape/sign attempts onward is from one session's worth of exploration on the same canonical split. Next step: fold-validate this configuration before treating 0.881 as an established number rather than a single-split result.
