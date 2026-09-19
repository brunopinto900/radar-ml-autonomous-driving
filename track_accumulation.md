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

## DeepReflecs + GRU: per-scan embeddings, learned temporal aggregation

Different mechanism than everything above: instead of pooling raw points across scans into one bigger point cloud, each scan keeps its own point set, gets reduced to one embedding vector by a frozen, already-trained DeepReflecs encoder (the N=1 control, `x_seq`/`y_seq`, per-scan recentered), and a GRU consumes the sequence of a track's embeddings. Precompute-then-train-RNN: the encoder runs once per scan across the whole dataset (cached), not once per window, so a stride=1 sliding window's N-1 scan overlap costs nothing extra in encoder compute. Unidirectional/causal by construction (a scan's verdict only depends on scans up to and including it), deliberately not mirroring the bidirectional LSTM in Hassan et al. 2024 (EuRAD, multi-frame RadarScenes classification), since a BiLSTM needs future scans and can't run in real-time streaming inference. No padding needed for short windows either, unlike the diff-vector attempt: the GRU just runs fewer steps for a young track (`pack_padded_sequence`).

N=10, stride=1, sensor-2-only, GRU hidden_size=64, 1 layer, MLP head (1 hidden layer, matching `mlp_classifier.MLP`'s shape), same canonical split:

| class | GRU N=10 (h=64) | histogram MLP N=10 + temporal variation | histogram MLP N=10 control |
|---|---|---|---|
| car | 0.918 | 0.917 | 0.909 |
| large_vehicle | 0.775 | 0.778 | 0.755 |
| two_wheeler | 0.842 | 0.824 | 0.803 |
| pedestrian | 0.839 | 0.843 | 0.841 |
| pedestrian_group | 0.862 | 0.861 | 0.846 |
| macro F1 | 0.847 | 0.845 | 0.831 |

Beats the plain control by +0.016, and edges out the best hand-engineered feature (temporal variation) by +0.002, a tie everywhere except `two_wheeler`, where the GRU is clearly ahead (+0.018 over temporal variation, +0.039 over control). `two_wheeler` is the one class this doc already flagged as having genuine aspect-angle/shape diversity to exploit, so that gap being real and not scattered is a meaningful signal, even though the headline macro F1 doesn't clear the existing best. Training curve caveat: train accuracy was still climbing at epoch 100 (0.894) while val accuracy plateaued/got noisy around 0.855 to 0.865 from roughly epoch 40 on, more consistent with a generalization/noise ceiling than starved capacity. Single split, not fold-validated.

### Model size and automotive SoC memory

19,941 params total (18,816 in the GRU, 1,125 in the MLP head), 80KB float32 or ~20KB int8-quantized. For scale, the frozen per-scan DeepReflecs encoder underneath it is only 1,317 params, so the GRU is already ~15x bigger than the thing feeding it. Per-step compute is roughly 3 x hidden_size x (hidden_size + input_dim) MACs, about 19k MACs per tracked object per scan at hidden_size=64, trivial next to what a radar SoC already runs every frame (FFTs, CFAR, beamforming), with or without a dedicated NN accelerator.

The real deployment number isn't parameter count, it's memory for buffering raw points, and this architecture is cheaper there than every pooling variant in this doc, not more expensive. Sensor-2 N=50 pooling holds up to 50 scans' worth of raw points per tracked object; all-sensor N=50 holds up to 4x that. The GRU needs none of it: a scan's points get encoded and discarded immediately, and all that persists per tracked object between scans is the GRU's hidden state, a fixed hidden_size-length vector (256 bytes at hidden_size=64), regardless of how long the track has run. Going from N=10 to N=50 costs the pooling variants real memory per tracked object; it costs the GRU variant nothing, since there's no window to hold, just a running state. Tradeoff made for that: the encoder's embeddings are whatever the N=1 model already learned for single-scan classification, not fine-tuned for temporal usefulness (the precompute-then-train choice made when planning this).

### Capacity ablation: hidden_size 64 vs 128

Same N=10, stride=1, sensor-2-only precompute GRU, only hidden_size doubled (64 to 128, ~19.9k to ~64.4k params):

| | GRU h=64 | GRU h=128 | best non-GRU (all-sensor N=50 + temporal variation) |
|---|---|---|
| macro F1 | 0.847 | 0.862 | 0.881 |

+0.015 from doubling hidden_size, a real, non-trivial gain (bigger than the earlier CONV_DIM/POINT_DIM capacity ablation's +0.007 on the pooled DeepReflecs model). Read against the training-curve caveat above (train still climbing, val flat/noisy at h=64): that pattern looked more like a generalization ceiling than starved capacity, but the ablation says capacity was still a real, if partial, factor. Still short of the best pooling-based result (0.881), by a smaller margin than h=64 was (0.019 vs 0.034 back). Single split, not fold-validated.

### Causal transformer variant: same embeddings, self-attention instead of recurrence

Same precomputed per-scan DeepReflecs embeddings and windowing as the GRU (N=10, stride=1, sensor-2-only), swapped the GRU for a small causal self-attention encoder (`DeepReflecsTransformer`): d_model=32 (matches the embedding dim, no input projection needed), 4 heads, 1 layer, feedforward dim 64, dropout 0. Needs two things a GRU gets for free: an explicit causal mask (attention has no inherent notion of order or direction) and a learned positional embedding per FIFO-buffer slot (0..9), added to the input embeddings before the first layer. The output at each sequence's own last real position (direct analog of the GRU's final hidden state h_t) feeds the same MLP head. Deliberately not sinusoidal positional encoding: sinusoidal's main advantage is generalizing to sequence lengths unseen in training, irrelevant here since every window is capped at N=10 by construction, and a learned table over only 10 positions costs a trivial 320 parameters.

| config | params | macro F1 |
|---|---|---|
| GRU, h=64 | 19,941 | 0.847 |
| Transformer, d=32 | 9,477 | 0.846 |

Essentially tied with the GRU (-0.001), not a loss, using roughly half the parameters. Reading this as "attention doesn't help here" would be too strong a conclusion from one run: the GRU's own capacity ablation (previous section) found a real +0.015 from doubling hidden_size, so a same-scale untuned transformer landing at parity rather than below suggests it isn't obviously worse per parameter, not that it's hit some fundamental ceiling. A small capacity/learning-rate sweep would be needed before concluding anything stronger than "roughly matches the GRU out of the box." Single split, not fold-validated.

Ran that capacity check: d_model doubled (32 to 64, dim_feedforward doubled alongside it to keep the usual 2x-d_model ratio, ~9.5k to ~37.3k params).

| model | small | large | delta |
|---|---|---|---|
| GRU (h=64 → h=128) | 0.847 (19.9k params) | 0.862 (64.4k params) | +0.015 |
| Transformer (d=32 → d=64) | 0.846 (9.5k params) | 0.848 (37.3k params) | +0.002 |

Different result from the GRU's own ablation, and a more telling one: the GRU clearly wasn't capacity-saturated at h=64, more width bought something real. The transformer barely moved despite a comparable (if not larger) relative jump in parameters. Combined, this points toward the transformer's ~0.846-0.848 being closer to an actual ceiling for this architecture on this data at N=10, not a capacity bottleneck the earlier untuned run just hadn't reached yet. Single split, not fold-validated.

### End-to-end variant: an OOM, and the per-batch collation fix

Planned and built the end-to-end counterpart (encoder not frozen, gradients flow through it every window, see module docstring in `scripts/deepreflecs_rnn_track_accumulation.py`), two init strategies: warm-start (encoder from the N=1 checkpoint, GRU+head from the already-trained precompute GRU checkpoint, nothing random at step 0) and random-init (everything trained jointly from scratch).

First implementation padded every window to one dense `(n_windows, max_seq_len, max_points_per_scan, n_features)` array per split, built once up front, the same convention `pad_to_fixed` already uses for the flat pooled representation elsewhere in this branch. On the full sensor-2-only N=10 dataset this OOM-killed the process (confirmed via `dmesg`: `oom-kill`, ~7.35GB resident on a 12GB machine, worse once a second concurrent run pushed it to 8.8GB before that one was stopped too). Root cause: points-per-scan is heavily skewed in this dataset (median 2, 99th percentile 13, single dataset-wide max 45), so forcing every one of N=10 scan-slots in every one of several hundred thousand windows to the global worst case (45) wastes roughly 45/2.9, about 15x, the memory a typical window actually needs. The flat pooled representation elsewhere in this branch doesn't have this problem since it only pads to the largest actual total points observed in any one window, not (worst-case per scan) x (scans per window).

Fix: pad fresh per mini-batch instead of once for the whole split (`collate_scan_sequences`), using only that batch's own max points-per-scan rather than the dataset-wide one. Verified correct on a tiny subset first (small train_acc/val_acc numbers matched the pre-fix run closely, same code path otherwise), then confirmed on the full dataset: resident memory stayed under 3GB combined for both variants running concurrently, versus the 7 to 9GB that triggered the kill. Costs a small amount of repeated Python-level padding work per training step, negligible next to the per-step point-encoder forward/backward pass that step already does.

Results, same N=10/stride=1/sensor-2-only config as the precompute GRU:

| config | macro F1 |
|---|---|
| end-to-end, random-init | 0.839 |
| GRU, h=64, precompute (frozen encoder) | 0.847 |
| Transformer, d=32, precompute | 0.846 |
| end-to-end, warmstart | 0.853 |
| GRU, h=128, precompute | 0.862 |

Clean ordering: warmstart beats frozen precompute, random-init loses to it. Letting an already-good encoder keep adapting to the temporal objective helps a little (+0.006 over the frozen version at the same hidden_size). Throwing away the single-scan pretraining entirely and learning everything jointly from scratch hurts, landing below even the simplest frozen-encoder baseline, worse than not fine-tuning the encoder at all. Reads as single-scan supervision carrying real, hard-to-recover value that a windowed multi-scan objective alone, with this much data, doesn't reconstruct from random weights. Still nothing in this whole GRU/Transformer/end-to-end line beats the pooled histogram MLP's best (0.881). Single split, not fold-validated.

### Late fusion: pooled DeepReflecs + GRU hidden state, concatenated

Direct test of whether the sequence branch (GRU over per-scan embeddings) knows anything the pooled branch (DeepReflecs on all N scans' points concatenated, order-blind) doesn't already have: both frozen, reused from already-trained checkpoints (pooled N=10 DeepReflecs, precompute GRU h=64), only a small new MLP head trained on their concatenated output (32-d pooled embedding + 64-d GRU hidden state = 96-d input). Same U-Net-style skip-connection reasoning as discussed: splice in a view that the other branch's own bottleneck would otherwise discard, rather than force the final decision through only one path.

| config | macro F1 |
|---|---|
| pooled DeepReflecs, N=10, alone | 0.825 |
| GRU, h=64, alone (frozen encoder) | 0.847 |
| fusion (pooled + GRU h=64) | **0.853** |
| end-to-end GRU, warmstart | 0.853 |

+0.006 over the GRU alone, real but modest, the same order of magnitude as most gains in this whole line. Notably, two completely different ways of extracting more signal, fine-tuning the encoder end-to-end versus bolting on an entirely separate pooled branch, land at almost the identical number (0.853 vs 0.8529). Reads as partial complementarity: the pooled branch carries a bit of information the GRU's own embedding sequence doesn't fully capture, but not enough to suggest the two views are seeing fundamentally different things. Still short of 0.881. Single split, not fold-validated.

### Temporal variation scalar, added to the GRU's own head

Same summed-diff scalar that helped the pooled histogram MLP (0.831 to 0.845 at N=10), now concatenated directly onto the GRU's final hidden state before the MLP head, instead of relying on the GRU to have learned that signal implicitly from the raw embedding sequence.

| config | macro F1 |
|---|---|
| GRU, h=64, alone (frozen encoder) | 0.847 |
| GRU, h=128, alone (frozen encoder) | 0.862 |
| fusion (pooled DeepReflecs + GRU h=64) | 0.853 |
| GRU, h=64, + temporal variation scalar | **0.860** |

+0.013 over the plain GRU h=64, bigger than the fusion gain (+0.006) and bigger than what the histogram MLP got from the same scalar at N=10 (+0.014, close call). Beats fusion despite using half the extra parameters (one scalar concatenated vs. a whole second 32-d embedding branch), and lands almost as high as doubling GRU capacity to h=128, for a fraction of the cost. Reads as evidence that the summed-diff signal isn't fully recoverable by the GRU from the embedding sequence alone, cheap explicit features still carry information a learned aggregator doesn't automatically reconstruct. Still short of 0.881. Single split, not fold-validated.

**Pushed to N=50, sensor-2-only, same architecture (GRU h=64 + temporal variation scalar).** Every earlier GRU/Transformer/fusion result in this branch stayed at N=10; this is the first time the sequence-model line has been pushed to the same window size that produced the branch's overall best pooled result.

| config | car | large_vehicle | two_wheeler | pedestrian | pedestrian_group | macro F1 |
|---|---|---|---|---|---|---|
| GRU+temporal, N=10, sensor-2-only | 0.929 | 0.807 | 0.845 | 0.845 | 0.873 | 0.860 |
| GRU+temporal, N=50, sensor-2-only | 0.952 | 0.790 | 0.884 | 0.871 | 0.899 | **0.883** |
| pooled MLP, all-sensor N=50 + temporal variation (prior overall best) | 0.940 | 0.814 | 0.853 | 0.905 | 0.894 | 0.881 |

Beats the branch's prior overall best, and does it on sensor-2-only data, no cross-sensor handoff needed. This changes the earlier read: it isn't that architecture never beats scaling N/sensors, it's that the GRU/Transformer/fusion line simply hadn't been pushed past N=10 yet, one axis (window size) was never actually varied for the sequence models until now. `two_wheeler` and `pedestrian_group` both jump hard (+0.039, +0.026 vs GRU+temporal N=10), `large_vehicle` drops (-0.017), an uneven pattern similar to the cross-sensor handoff results, not a uniform scaling effect.

Caveat, same as flagged above: no early stopping or best-checkpoint selection. This run's training curve is different in character from every other GRU/Transformer run so far, train accuracy is still climbing meaningfully in the last 20 epochs (0.9413 to 0.9456) while val accuracy oscillates in a tighter but still real band (0.887 to 0.894), a widening train/val gap that reads as the model starting to overfit rather than being fully converged. Unlike the orthogonalized-fusion caveat above, this one cuts the other way: the epoch-100 snapshot could still be missing a bit more real signal (undertrained relative to its own train curve) or could already be past its best val epoch (overfitting), and there's no way to tell which without checkpoint selection. This result has not been fold-validated, and given it's now the new best in the branch, it's the more urgent fold-validation candidate, alongside the pooled 0.881 already queued.

Same scalar, added to the Transformer's final hidden state instead:

| config | macro F1 |
|---|---|
| Transformer, d=32, alone (frozen encoder) | 0.846 |
| Transformer, d=32, + temporal variation scalar | 0.845 |
| GRU, h=64, + temporal variation scalar | 0.860 |

Flat, effectively zero change, within noise of the Transformer alone. Sharp contrast with the GRU's own +0.013 from the identical scalar on the identical embeddings. Consistent with the capacity ablation asymmetry already seen (GRU +0.015 from doubled width, Transformer +0.002): the Transformer's attention pooling over the embedding sequence already seems to be extracting close to what it's going to extract from this data at N=10, so bolting on an extra scalar doesn't move it, while the GRU's more constrained recurrent summary still had room for an explicit feature to add something it wasn't inferring on its own. Single split, not fold-validated.

### Design note: why a three-way fusion (pooled skip + temporal scalar) likely won't stack cleanly

Not run, reasoning only. The pooled DeepReflecs skip connection (fusion, +0.006 over GRU alone) and the temporal variation scalar (+0.013 over GRU alone) are both patches for the same underlying weakness, the frozen N=1 embedding sequence not fully capturing something the GRU needs. They are not obviously independent sources of information, so naively concatenating both onto the GRU head is unlikely to give a stacked gain anywhere near +0.019.

Domain reason they overlap: the pooled DeepReflecs branch is an order-blind aggregate shape over all N scans' points. An object that changes a lot scan to scan, exactly what the temporal variation scalar measures directly, will also show up as more spread in that pooled aggregate. The scalar isn't really an independent feature, it's a compressed, explicit version of a variance signal the pooled embedding already carries implicitly and diffusely.

Two ways to check this properly, in increasing order of rigor, before trusting any three-way fusion number:

1. Cheap diagnostic: fit a linear probe predicting the temporal variation scalar from the pooled DeepReflecs embedding (linear regression, R² on held-out data). High R² confirms the two are carrying mostly the same signal, and shows which one is more compressed (the scalar) versus more expensive (the 32-d embedding) for that same information.
2. If overlap is confirmed and both are still wanted, orthogonalize instead of concatenating raw: regress the temporal-predictable component out of the pooled embedding and feed the GRU head only the residual. Any gain measured afterward is then guaranteed non-redundant, rather than hoping a small MLP head disentangles the overlap on its own from limited data.

If the mechanism above is right, the actually clean fix isn't fusion math at all: restrict the pooled branch to central-tendency stats (means) and let the temporal scalar own dispersion, so each branch has a distinct job instead of both branches smearing across the same axis.

**Diagnostic result.** Linear regression predicting the two temporal variation scalars from the pooled DeepReflecs embedding (32-d), fit on train, evaluated on test:

| temporal scalar | R² train | R² test |
|---|---|---|
| rcs (summed diff) | 0.402 | 0.387 |
| vr_compensated (summed diff) | 0.356 | 0.334 |

Real overlap (roughly a third to 40% of variance recoverable), but partial, not near total: 60 to 65% of each scalar's variance is not linearly recoverable from the pooled embedding, so the two are neither independent nor redundant, somewhere in between. This is high enough to make a raw three-way concatenation (pooled + GRU hidden + temporal) unlikely to add the full +0.019 the two individual gains would suggest, but not so high that the temporal scalars are pure restatement of the pooled embedding either.

**Orthogonalized three-way fusion result.** Built the fix described above: regressed the pooled embedding on the temporal scalars (fit on train), kept only the residual, concatenated `[pooled residual (32) ; GRU hidden (64) ; temporal (2)]` into one 98-dim input for a fresh MLP head.

| config | macro F1 |
|---|---|
| GRU, h=64, alone (frozen encoder) | 0.847 |
| fusion (pooled + GRU h=64) | 0.853 |
| GRU, h=64, + temporal variation scalar | 0.860 |
| orthogonalized fusion (pooled residual + GRU + temporal) | 0.8544 |

Worse than GRU+temporal alone, not just short of an additive +0.019. The orthogonalization did its job (removing the ~35% shared variance so the pooled branch can no longer restate the temporal scalar), but what's left in the residual isn't itself useful for classification, it reads as mostly noise once the shared part is gone. So the earlier fusion gain (0.853, pooled+GRU without temporal) was likely riding on exactly the variance that overlaps with temporal variation, not on some separate signal the pooled branch uniquely holds. Once that shared part is made explicit and available directly (via the temporal scalar), the pooled branch has nothing left to add, and the extra 32-dim input just gives the small MLP head more to overfit against the same amount of data. Confirms the design note's mechanism, and answers the open question from it: no, three-way fusion isn't worth it, in either the naive or the orthogonalized form. Single split, not fold-validated.

**Caveat on all of these single-split GRU/Transformer/fusion numbers.** None of the trainers in this line (GRU, Transformer, fusion, either temporal-variation variant, the orthogonalized fusion above) do early stopping or best-checkpoint selection. Every one saves whatever the model looks like at epoch 100, whatever that happens to be. Checked directly on this orthogonalized fusion run and the plain GRU+temporal run: train accuracy is flat by epoch 100 in both, but val accuracy is still oscillating by about a full point (0.86 to 0.87) even after training accuracy has stopped moving. The 0.0056 gap between orthogonalized fusion (0.8544) and GRU+temporal (0.860) is smaller than that oscillation band, so it's fair to say fusion added nothing, not fair to read the two numbers as precisely ranked. Same caveat applies to every close call in this table (GRU h=64 vs Transformer d=32, fusion vs end-to-end warmstart, etc.), differences under roughly a point should be read as ties, not as ordered results, unless checked against the training curve first.

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
| N=10, DeepReflecs+GRU (h=64), sensor-2-only | 0.918 | 0.775 | 0.842 | 0.839 | 0.862 | 0.847 |
| N=10, DeepReflecs+GRU (h=128), sensor-2-only | 0.930 | 0.813 | 0.845 | 0.848 | 0.874 | 0.862 |
| N=10, DeepReflecs+causal Transformer (d=32), sensor-2-only | 0.917 | 0.785 | 0.826 | 0.839 | 0.861 | 0.846 |
| N=10, DeepReflecs+causal Transformer (d=64), sensor-2-only | 0.914 | 0.771 | 0.834 | 0.847 | 0.872 | 0.848 |
| N=10, DeepReflecs+GRU end-to-end, warmstart, sensor-2-only | 0.923 | 0.786 | 0.847 | 0.839 | 0.870 | 0.853 |
| N=10, DeepReflecs+GRU end-to-end, random-init, sensor-2-only | 0.913 | 0.773 | 0.823 | 0.835 | 0.850 | 0.839 |
| N=10, fusion (pooled DeepReflecs + GRU h=64), sensor-2-only | 0.923 | 0.787 | 0.847 | 0.840 | 0.868 | 0.853 |
| N=10, DeepReflecs+GRU (h=64) + temporal variation scalar, sensor-2-only | 0.929 | 0.807 | 0.845 | 0.845 | 0.873 | 0.860 |
| N=10, DeepReflecs+causal Transformer (d=32) + temporal variation scalar, sensor-2-only | 0.914 | 0.782 | 0.818 | 0.844 | 0.867 | 0.845 |
| N=10, orthogonalized fusion (pooled residual + GRU h=64 + temporal variation), sensor-2-only | 0.923 | 0.785 | 0.850 | 0.842 | 0.872 | 0.854 |
| N=50, DeepReflecs+GRU (h=64) + temporal variation scalar, sensor-2-only | 0.952 | 0.790 | 0.884 | 0.871 | 0.899 | **0.883** |

6-fold validated (val split, mean across folds, different rotating splits, not the canonical one above):

| variant | car | large_vehicle | two_wheeler | pedestrian | pedestrian_group | macro F1 | std |
|---|---|---|---|---|---|---|---|
| accumulation baseline, N=5 raw (16/32 capacity) | 0.902 | 0.793 | 0.723 | 0.787 | 0.796 | 0.800 | 0.023 |
| same, doubled capacity (32/64) | 0.907 | 0.793 | 0.736 | 0.794 | 0.807 | 0.807 | 0.022 |
| single-scan baseline, 6-fold (reference, pre-branch) | — | — | — | — | — | 0.735 | 0.025 |

Best single-split number in the branch is all-sensor N=50 + temporal variation at 0.881 (raw, unsmoothed), stacking both real levers found this session: more real time per track (large N) and more real detections per track (cross-sensor handoff). Post-hoc smoothing helps at N=10 (+0.005) but not at N=50 (-0.001), likely because heavily overlapping N=50 windows are already autocorrelated before any smoothing. Nothing here is fold-validated. Everything from the shape/sign attempts onward is from one session's worth of exploration on the same canonical split. Next step: fold-validate this configuration before treating 0.881 as an established number rather than a single-split result.

## TLDR-Comparison

Every variant tried in this branch, one row each, sorted by macro F1 descending. Smoothing variants excluded (post-hoc, not a distinct architecture/feature). "What was done" is the method itself, "Motivation" is why it was tried, "Result" is the outcome. Single-split numbers unless a fold mean/std is noted.

### Results (macro F1 known)

| Variant | What was done | N | Model | Sensor | Macro F1 | Motivation | Result |
|---|---|---|---|---|---|---|---|
| GRU h=64 + temporal variation | Concatenated the two temporal-variation scalars onto the GRU's final hidden state, pushed to N=50 instead of N=10 | 50 | DeepReflecs | sensor2 | **0.883** | Line had never been pushed past N=10, test if it scales with window size like the pooled MLP did | New overall best; uneven per-class (two_wheeler/pedestrian_group up hard, large_vehicle down); train/val gap still widening at epoch 100, not fully converged |
| Pooled + temporal variation, delta_t corrected | Histogram MLP trained on all-4-sensor windowed point sets (N=50) plus elapsed-time-normalized temporal variation scalars | 50 | MLP | all-sensor | 0.881 | Combine both known levers (bigger N, cross-sensor handoff) in one run | Axes stack rather than cancel, clears both individual bests; was overall best until GRU N=50 above |
| Pooled + temporal variation, uncorrected gap-norm | Same all-sensor histogram MLP pipeline at N=20, temporal variation normalized by gap count, not yet elapsed time | 20 | MLP | all-sensor | 0.871 | Push cross-sensor handoff to a bigger window | Was best at the time; later found to overstate the gain (see corrected row) |
| Pooled + temporal variation | Histogram MLP on sensor-2-only N=50 windows plus temporal variation scalars | 50 | MLP | sensor2 | 0.867 | Same feature at the largest single-sensor window tried | New best for sensor2-only line; feature's own delta not monotonic across N, reads as noise not trend |
| Pooled + temporal variation, delta_t corrected | Re-ran N=20 all-sensor with temporal variation renormalized by real elapsed time instead of gap count | 20 | MLP | all-sensor | 0.866 | Uncorrected gap-normalization assumed uniform gap duration, false once sensor handoffs mix short/long gaps | Small real correction, all classes down a little; erases the earlier win over sensor2 N=50, now roughly tied |
| GRU h=128 | Doubled the GRU's hidden_size from 64 to 128, otherwise identical config | 10 | DeepReflecs | sensor2 | 0.862 | Check if h=64 GRU was capacity-limited | Real, non-trivial +0.015; GRU not capacity-saturated at h=64 |
| GRU h=64 + temporal variation | Concatenated the two temporal-variation scalars onto the GRU's h=64 hidden state before the MLP head | 10 | DeepReflecs | sensor2 | 0.860 | Add the cheap temporal scalar directly, bypass the frozen encoder's bottleneck | Biggest single gain in this line at N=10 (+0.013), bigger than fusion, far fewer added params |
| Pooled + temporal variation | Histogram MLP, N=20, temporal variation scalars added | 20 | MLP | sensor2 | 0.859 | Check if temporal feature's contribution holds at bigger window | Real but roughly half the marginal gain of N=10, bigger window starts implicitly carrying what the feature added |
| Pooled control | Histogram MLP, N=50, no extra features | 50 | MLP | sensor2 | 0.858 | Push N further on one sensor | Small further gain over N=20, diminishing but not flat |
| Pooled + temporal variation, uncorrected gap-norm | Switched df to the all-sensor points table (sensor_id=None), otherwise same N=10+temporal histogram MLP pipeline | 10 | MLP | all-sensor | 0.854 | Test cross-sensor handoff vs. more time on one sensor | Real gain over sensor2 N=10+temporal; pedestrian jumped hard, large_vehicle dropped slightly; carries the uncorrected gap-norm bias, never re-run with the fix |
| Orthogonalized fusion (pooled residual + GRU + temporal) | Regressed the pooled embedding on the temporal scalars (fit on train), kept only the residual, concatenated with GRU hidden state and temporal scalars into one MLP head | 10 | DeepReflecs | sensor2 | 0.854 | Probe found ~35% shared variance between pooled embedding and temporal scalar; test if removing overlap recovers a cleaner additive gain | Worse than GRU+temporal alone; residual carries no independent signal, closes the three-way fusion question |
| End-to-end GRU, warmstart | Unfroze the DeepReflecs encoder, backpropagated through it every window, initialized from the separately-trained N=1 encoder + precompute GRU checkpoints | 10 | DeepReflecs | sensor2 | 0.853 | Let the frozen encoder keep adapting to the temporal objective | Real +0.006 over frozen GRU |
| Fusion (pooled + GRU h=64) | Concatenated the pooled (order-blind, all-N-scans) DeepReflecs embedding with the GRU's final hidden state, trained a new small MLP head, both branches frozen | 10 | DeepReflecs | sensor2 | 0.853 | Test if the order-blind pooled branch knows anything the GRU sequence branch doesn't | Real but modest +0.006, lands almost identical to end-to-end warmstart via a totally different mechanism |
| Pooled control | Histogram MLP, N=20, no extra features | 20 | MLP | sensor2 | 0.852 | Base accumulation hadn't saturated at N=10 like the DeepReflecs curve suggested | Real +0.021 over N=10 |
| Transformer d=64 | Doubled d_model from 32 to 64 (and dim_feedforward alongside it), otherwise identical Transformer config | 10 | DeepReflecs | sensor2 | 0.848 | Check if the Transformer was capacity-limited like the GRU turned out to be | Barely moved (+0.002) despite comparable relative param jump; reads as a real architectural ceiling |
| GRU h=64 | Ran a frozen per-scan DeepReflecs encoder over each scan, fed the embedding sequence into a causal unidirectional GRU (packed variable-length sequences), MLP head on the final hidden state | 10 | DeepReflecs | sensor2 | 0.847 | Test per-scan embeddings + learned temporal aggregation instead of pooling raw points, causal for real-time deployment | Beats histogram control (+0.016), edges out best hand-engineered feature (+0.002); two_wheeler clearly ahead |
| Transformer d=32 | Same precomputed embeddings/windowing as the GRU, swapped GRU for a causal self-attention encoder with a learned positional embedding table | 10 | DeepReflecs | sensor2 | 0.846 | Self-attention over the same embeddings instead of recurrence | Essentially tied with GRU h=64 (-0.001) at about half the params |
| Pooled + temporal variation | Histogram MLP, N=10, added the two temporal-variation scalars (median-based summed absolute diff per gap) | 10 | MLP | sensor2 | 0.845 | Retry temporal-variation feature at a longer window (more gaps to average) | Real signal (+0.014, 4-5x the N=5 move), concentrated in large_vehicle/two_wheeler/pedestrian_group |
| Transformer d=32 + temporal variation | Concatenated the temporal-variation scalars onto the Transformer's final-position output before the MLP head | 10 | DeepReflecs | sensor2 | 0.845 | Same scalar addition as the GRU version, on the Transformer | Flat (-0.001), sharp contrast with GRU's +0.013; consistent with Transformer already near its capacity ceiling |
| Pooled + transition matrix | Replaced the temporal-variation scalar with a 4x4 bin-to-bin transition count per feature | 10 | MLP | sensor2 | 0.839 | Test if preserving bin-to-bin transition shape beats magnitude-only diff | Worse than magnitude-only (-0.006) |
| End-to-end GRU, random-init | Same end-to-end setup as warmstart, but every weight (encoder, GRU, head) initialized randomly | 10 | DeepReflecs | sensor2 | 0.839 | Isolate how much of warmstart's gain depends on single-scan pretraining vs. the end-to-end objective itself | Worse than even frozen baseline; single-scan pretraining carries value a windowed objective alone doesn't reconstruct from random weights |
| Pooled + signed diff vector | Replaced the summed-absolute-diff scalar with one signed column per gap, right-aligned and zero-padded for shorter windows | 10 | MLP | sensor2 | 0.836 | Test if keeping sign (not just magnitude) of scan-to-scan diff helps | Worse than magnitude-only (-0.009) |
| Pooled + diff vector + window length | Added one extra column (real scan count) to the signed diff vector | 10 | MLP | sensor2 | 0.834 | Let the model tell padded columns from real ones | No improvement (-0.002), one scalar among 99 columns wasn't enough signal |
| Pooled control | Histogram MLP, N=10, no extra features | 10 | MLP | sensor2 | 0.831 | Push N past 5 for the cheaper histogram MLP now that it ties DeepReflecs | Real +0.016 over N=5, MLP hadn't saturated at N=5 |
| Pooled control | DeepReflecs point-set encoder run directly on N=10 pooled points (all scans' points concatenated, order-blind max-pool) | 10 | DeepReflecs | sensor2 | 0.825 | Continue the point-set N sweep past 5 | Real +0.013 over N=5 raw; this is where the point-set line's own N sweep stopped |
| Pooled + temporal variation | Same temporal-variation scalar as N=10, first tried at N=5 | 5 | MLP | sensor2 | 0.818 | First test of scan-to-scan magnitude of change at N=5 | Flat (+0.003), inside noise band, signal too weak to detect at this N |
| Pooled control (clean 5-feature) | Quantile-bin histogram encoding (5 features, no doppler_spread) plus MLP, on the same N=5 windowed point sets DeepReflecs uses | 5 | MLP | sensor2 | 0.815 | Test if accumulation helps via denoised aggregate summary, not point-set-specific pattern | Ties DeepReflecs at N=5 (within 0.004), favors "denoising" over architecture-specific learning |
| Pooled control, raw range_sc | Left each pooled point's own range_sc untouched instead of broadcasting the target scan's value | 5 | DeepReflecs | sensor2 | 0.812 | Check if broadcasting vs. leaving each point's own range matters | Statistically indistinguishable from broadcast (+0.001), raw kept as simpler |
| Pooled control, broadcast range_sc | Pooled 5 scans' points per track into one DeepReflecs classification, target scan's range_sc broadcast to every point | 5 | DeepReflecs | sensor2 | 0.811 | First real accumulation test | Large real gain over N=1 (+0.059) |
| Accumulation baseline, 6-fold, doubled capacity | Doubled CONV_DIM/POINT_DIM (16/32 to 32/64) on the N=5 accumulation baseline, same 6 folds | 5 | DeepReflecs | sensor2 | 0.807 (std 0.022) | Check if the N=5 baseline is under-capacity for pooled input | 6/6 folds improve (+0.007 mean), real but minor, not the main lever |
| Accumulation baseline, 6-fold | Re-trained the N=5 raw accumulation baseline across the same 6 sequence-grouped val carves used for the single-scan model | 5 | DeepReflecs | sensor2 | 0.800 (std 0.023) | Confirm the N=5 accumulation gain survives split variance | 6/6 folds beat single-scan 6-fold baseline, ranges barely overlap, real effect |
| N=1 control (x_seq/y_seq) | Single-scan DeepReflecs, recentered points using the global (x_seq/y_seq) frame instead of the car frame | 1 | DeepReflecs | sensor2 | 0.752 | Re-baseline under the frame needed once scans get pooled | Neutral vs. pre-branch (+0.003), frame change isn't confounding the N sweep |
| Single-scan baseline (pre-branch) | Original single-scan DeepReflecs classifier, car-frame (x_cc/y_cc) recentering | 1 | DeepReflecs | sensor2 | 0.749 | Original reference point before any accumulation | Superseded once accumulation began |
| Single-scan baseline, 6-fold | Same pre-branch single-scan model, evaluated across 6 sequence-grouped val carves instead of one split | 1 | DeepReflecs | sensor2 | 0.735 (std 0.025) | Establish the fold-to-fold noise floor before crediting any later gain as real | Sets the reference band (0.706-0.779) later gains had to clear |

### Not yet run (gaps)

| Variant | N | Model | Sensor | Motivation |
|---|---|---|---|---|
| Pooled + temporal variation, delta_t corrected | 10 | MLP | all-sensor | Fix was applied at N=20/50 but never re-run at N=10 |
| Pooled control (no temporal) | 10, 20, 50 | MLP | all-sensor | Isolate handoff alone vs. handoff + temporal feature, every all-sensor run bundles both |
| DeepReflecs point-set pooling | 20, 50 | DeepReflecs | sensor2 | Would show if point-set N-scaling matches the MLP's; MLP took over for pushing N instead |
| GRU h=64 | 20 | DeepReflecs | sensor2 | Fill the gap between N=10 and N=50 GRU results |
| GRU h=64 (no temporal) | 50 | DeepReflecs | sensor2 | Isolate how much of the 0.883 is N=50 alone vs. N=50+temporal |
| GRU h=128 | 50 | DeepReflecs | sensor2 | Check if the h=64→h=128 capacity gain still holds at N=50 |
| Transformer d=32 (plain) | 20, 50 | DeepReflecs | sensor2 | Check if the Transformer's apparent ceiling at N=10 is a window-size artifact |
| Transformer d=32 + temporal variation | 50 | DeepReflecs | sensor2 | Direct Transformer counterpart to the GRU N=50 jump |
| Fusion (pooled + GRU) | 20, 50 | DeepReflecs | sensor2 | Check if fusion's gain persists once GRU itself is stronger at larger N |
| Orthogonalized fusion | 50 | DeepReflecs | sensor2 | Direct counterpart to the N=10 orthogonalized fusion, at the new best N |
| End-to-end GRU (warmstart/random) | 20, 50 | DeepReflecs | sensor2 | Check if end-to-end's edge over frozen GRU holds at larger N, also needs the collation fix's memory profile re-checked |
| GRU h=64 | 10 | DeepReflecs | all-sensor | Sequence-model line never combined with cross-sensor handoff, the MLP's second-biggest lever |
| Transformer d=32 | 10 | DeepReflecs | all-sensor | Same, for the Transformer |
| Fusion (pooled + GRU) | 10 | DeepReflecs | all-sensor | Same, for fusion (would need an all-sensor pooled DeepReflecs encoder too) |
