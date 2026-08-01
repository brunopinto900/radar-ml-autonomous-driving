# EDA Plan (Day 4)

Radar point clouds have a few domain-specific quirks that generic tabular/image EDA
doesn't cover. This is the plan before running anything, plus a place to record what
the plots actually show once run. Uses `results/train_points.parquet` only (never
val/test, per DESIGN_DECISIONS.md's split discipline).

## 1. Test the train/truck/bus/large_vehicle merge directly

DESIGN_DECISIONS.md decision 1 merged these into one `large_vehicle` class based on
instance counts and the official RadarScenes precedent, but flagged that RCS
separability wasn't actually checked. Do this first, before anything else, since it
validates (or undermines) a decision everything downstream assumes.

Not "merged vs. sub-class" - that's circular, the merged distribution is just the union
of the four sub-classes by construction. Instead:

- **Overlay the four original sub-classes** (`label_name`, not the merged `class_name`:
  `train`, `truck`, `bus`, `large_vehicle`) on the same histogram, for RCS, Doppler
  spread (per-instance `std(vr_compensated)`, not the pooled point-wise Doppler - bulk
  radial velocity is a behavioral/context property, e.g. trains cruise at fixed track
  speed vs. road vehicles in stop-and-go traffic, not an intrinsic radar signature, so
  it's not informative here), and spatial extent (instance `x_cc`/`y_cc` bounding-box
  diagonal - not azimuth, which reflects where in the scene something was recorded, not
  what it looks like to radar). Substantial overlap = merging loses nothing. One
  sub-class sitting clearly apart = merging hides a real distinction.
- **Check the merged group is still separated from the other final classes** (car,
  pedestrian, pedestrian_group, two_wheeler) - the actual bar that matters, since the
  MLP only ever sees the 5 merged classes, not the original 12.
- Visual/eyeball comparison (per the plan's own "eyeball separability" framing), not a
  formal statistical test - `train` at only 57-163 instances would make any significance
  test badly underpowered anyway.

**Findings:**

- **RCS** (`results/eda/01a_large_vehicle_subclasses.png`): `truck` and `large_vehicle` overlap
  closely. `train` is shifted right (higher RCS) and `bus` is shifted left (lower RCS) relative
  to the other three - the sub-classes aren't uniform here, but the shifts are moderate, not a
  clean separation.
- **Doppler spread** (per-instance `std(vr_compensated)`, not pooled point-wise Doppler - bulk
  radial velocity is a behavioral/context property, e.g. trains cruise at fixed track speed vs.
  road vehicles in stop-and-go traffic, not an intrinsic radar signature): `truck`, `bus`,
  `large_vehicle` all collapse to small, similar spreads (median near 0 m/s, whiskers under
  ~2.5 m/s). `train` stands completely apart (median ~12 m/s, IQR 8-15, whisker to ~22 m/s).
  Likely explanation: not articulation (trains are rigid), but geometry - a long rigid object
  (median extent ~19m) gets genuine radial-velocity disagreement across its own length from
  points sitting at different angles to the sensor. So this feature is confounded by size, not
  a pure rigidity/articulation signal.
- **Spatial extent** (`x_cc`/`y_cc` bounding-box diagonal, per instance): the largest divergence
  of the three. `train` median ~19m (whiskers to 35m) vs. `large_vehicle` median ~5m - boxes
  barely overlap. `truck` (~7m) and `bus` (~13.5m) sit in between.
- **Sequence concentration check** (before trusting the `train` divergence above): counted
  distinct sequences contributing instances per sub-class. `truck` (51 seqs), `bus` (22),
  `large_vehicle` (22) are all reasonably spread out, no single sequence dominating (top
  truck sequence is ~9% of its instances) - those findings are solid. **All 57 `train`
  instances come from a single sequence** (`sequence_2`) - not "mostly," literally `n=1` at
  the sequence level. That means the Doppler-spread and spatial-extent divergence found for
  `train` is statistically indistinguishable from a property of that one recording (one
  specific train, one speed, one curvature, one calibration state, one approach geometry) -
  there is no second sequence to check whether it generalizes. The geometric explanation
  (long rigid body -> radial-velocity disagreement across its length) is still a plausible
  mechanism, but with one sequence's worth of evidence it cannot actually be confirmed as a
  class-level effect rather than a single-recording artifact.
- **Verdict on the merge:** `truck` and `large_vehicle` genuinely look alike on all three
  features - merging those two loses little. `bus` diverges somewhat on RCS but is spread
  across enough sequences (22) that this looks like a real class effect. `train` is the weak
  link, but not for the reason initially thought - the concern isn't really "does train look
  different," it's "we only have one sequence's worth of evidence either way, so this
  particular merge decision is being made almost blind." Given `train` is also the rarest
  sub-class (57 instances, all from one sequence) and RadarScenes' own reduced-label scheme
  merges it the same way, keeping the merge is still the reasonable default - but call it what
  it is: a decision made on precedent and low sample size, not on validated evidence, since
  the one sequence available can't distinguish a class effect from a recording artifact.
- **Merged `large_vehicle` vs. other final classes**
  (`results/eda/01b_merged_vs_other_classes.png`): on RCS, `large_vehicle` is shifted right of
  `pedestrian`/`pedestrian_group`/`two_wheeler`, with partial overlap against `car`. On Doppler
  spread, `car` is compressed near zero (rigid body, points agree on radial velocity) while
  `pedestrian`/`pedestrian_group`/`two_wheeler` all sit meaningfully higher (median ~0.3-0.4
  m/s) - consistent with the articulation hypothesis (swinging limbs, pedaling legs) actually
  holding up. `large_vehicle`'s Doppler-spread box is low-median but unusually wide (0-0.58) -
  a visible signature of the merge itself, mixing the near-zero `truck`/`bus`/`large_vehicle`-
  subclass population with the high-spread `train` population. Doesn't cleanly separate
  `large_vehicle` from `car` (their medians are close there) - that separation is carried more
  by RCS. No pair of classes is fully non-overlapping on any single feature; these look like
  contributing signals for a model, not standalone discriminators.

## 2. Per-class distributions of RCS, Doppler, point count

Histograms/box plots, one per class (`car`, `large_vehicle`, `pedestrian`,
`pedestrian_group`, `two_wheeler`). Baseline EDA.

**Findings:** the per-class baseline itself is still TBD. One preliminary side-check has
been done ahead of it, prompted by a radar-DSP point about the sensor's own field of view:
antenna gain rolls off away from boresight, so a detection near the edge of the FOV could
show depressed RCS purely from reduced illumination/sensitivity, independent of the object's
own size or orientation - a sensor artifact, not a target property. This is directly
checkable, unlike the earlier (untestable - no per-object heading in this dataset) point
about RCS varying with the *object's own* aspect angle relative to the radar (see item #1's
RCS discussion re: `bus`'s wide spread from specular glint off flat panels at different
facet angles).

Checked median `rcs` (all classes pooled) against `|azimuth_sc|` bins (FOV spans about
±70°):

| \|azimuth\| | 0-5° | 5-10° | 10-15° | 15-20° | 20-25° | 25-30° | 30-40° | 40-50° | 50-60° | 60-90° |
|---|---|---|---|---|---|---|---|---|---|---|
| median RCS | -7.4 | -7.6 | -6.7 | -5.7 | -1.7 | -0.3 | -2.5 | -4.2 | -2.5 | -3.4 |

Not a monotonic roll-off - it's a hump, peaking (highest RCS) around 20-30° off boresight
and lower both near boresight and at the wide edges. **Inconclusive**, not a clean
confirmation or rejection of the FOV-gain hypothesis: could mean the released `rcs` field is
already antenna-pattern-compensated by the sensor's internal processing before being logged,
or (more likely) that this pooled, uncontrolled view is dominated by other confounds -
range and class mix both vary systematically with azimuth in real driving scenes (e.g.
distant lead vehicles cluster near 0° azimuth, roadside traffic at favorable range/aspect
sits around 20-30°), so azimuth's effect can't be isolated without controlling for range
first. Would need range-binned or range-controlled azimuth analysis to actually separate a
sensor-gain effect from a scene-geometry effect - not done yet, noted as a follow-up if the
per-class RCS baseline below turns up something azimuth might explain.

**Follow-up: same check, broken out per class** (`results/eda/02a_rcs_vs_azimuth_by_class.png`)
- more informative than the pooled version. Every class shows the same qualitative hump:
low near boresight (0-20°), peaking around 20-30°, declining/flattening past that -
`car`, `pedestrian`, `pedestrian_group`, `two_wheeler` all move together despite being
physically very different objects with very different typical scene placement. If the
pooled pattern were purely a scene-geometry/class-mix confound (different classes just
happening to sit at different azimuths for layout reasons), each class's curve would be
expected to look different, shaped by its own typical position in a scene. Instead they
track each other closely - stronger evidence for a shared, sensor-level effect (antenna
gain/beam pattern) than for a pure confound, though range still isn't controlled for, so
this isn't a final answer. One real exception: `large_vehicle` breaks the shared pattern at
wide angles, climbing steadily past 30° instead of flattening like the other four - plausibly
its own physical size (a truck/bus seen broadside at a wide angle presents a large flat
panel, which could dominate over whatever the shared sensor effect is doing). `two_wheeler`
shows a sharper, narrower spike at 25-30° than the smooth humps of the other classes -
possibly just smaller sample size in that class/bin combination rather than a real
distinct effect; worth not over-reading.

> **Headline finding, bigger than the azimuth-artifact question itself:** the *between-class*
> separation in this same plot is stable across the entire angular range - `large_vehicle`
> sits on top at every single azimuth bin, `car` is consistently second, `pedestrian_group`/
> `pedestrian` sit together in the middle, `two_wheeler` is consistently lowest. The ranking
> never inverts or collapses anywhere in the FOV. Whatever the boresight hump is (sensor gain
> or otherwise), it's a small wiggle (a few dB) riding on top of a much larger, stable
> between-class gap (5-8+ dB between adjacent classes). This is effectively a preview of item
> #2's core question - median RCS looks like it separates these 5 classes cleanly, almost
> independent of viewing angle.

## 3. Point-count-vs-range confound

Point count isn't a pure size signal - a farther-away object returns fewer detections
regardless of true size, because it subtends a smaller angle at the sensor's angular
resolution. Comparing point counts across classes without checking range risks
measuring "how far away this class tends to be recorded" rather than "how big this
class is." Scatter of point count vs. range, colored by class, before trusting point
count as a discriminative feature.

**Findings:** _(TBD)_

## 4. Doppler spread within an instance, not just its center

Already seen on the large_vehicle example from Day 2/3: most points sat around -4 to
-13 m/s, but a couple wheel-related points spiked to -33/-42 m/s (micro-Doppler).
Articulated objects (pedestrians swinging limbs, cyclists pedaling) tend to show wider
per-instance Doppler variance than rigid bodies (cars) even at the same bulk velocity.
Check per-instance `std(vr_compensated)`, not just the mean, as a candidate feature.

**Findings:** already computed and plotted as a byproduct of item #1's merge check
(`results/eda/01b_merged_vs_other_classes.png`, right panel) - reused here rather than
regenerated. Confirms the hypothesis, most clearly for the comparison that matters most:

- `car` (rigid body): median `vr_std` ~0.03 m/s - points barely disagree with each other.
- `pedestrian` (articulated - swinging limbs): median ~0.38 m/s, ~12x higher than `car`.
- `pedestrian_group` (~0.32) and `two_wheeler` (~0.29) sit in between - a pedestrian group
  is multiple independently-moving bodies rather than one articulated body, and a
  cyclist/motorcyclist has pedaling legs but also a rigid frame and wheels, so partial
  articulation is physically expected for both.
- `large_vehicle` looks low-median (~0.07, close to `car`) but with a wide tail (whisker
  to ~1.43 m/s) - that's the `train` sub-population (see item #1) dragging the tail up; a
  typical truck/bus/large_vehicle instance behaves like a rigid body, same as `car`.

Per-instance `std(vr_compensated)` is a real, usable micro-Doppler feature - it separates
rigid bodies (`car`, most of `large_vehicle`) from articulated/multi-body ones
(`pedestrian`, `pedestrian_group`, `two_wheeler`) by roughly an order of magnitude in the
median, though boxes still overlap so it's a contributing signal, not a standalone
discriminator.

**Distribution shape** (`results/eda/01c_vr_std_histograms.png`, generated to check whether
the boxplot quartiles reflect a Gaussian shape): neither class is actually Gaussian-shaped.
Every class's `vr_std` distribution is monotonically decaying from zero (a spike at 0 that
falls off), not a bell curve centered away from zero - expected, since `vr_std` is itself a
derived spread statistic (closer to a chi-distribution shape than Gaussian). What differs
between classes is the decay rate: `car`/`large_vehicle` spike very high and narrow right at
0, `pedestrian`/`pedestrian_group`/`two_wheeler` have a lower, flatter peak that decays much
more slowly - i.e. "car is sharply concentrated near zero, pedestrian is more spread out but
still zero-peaked," not "car concentrated vs. pedestrian Gaussian."

**Aliasing evidence:** the full-range log-scale panel shows the outlier tail isn't a smooth
continuous decay - there are distinct bumps/local peaks (~15-20 m/s, and separate spikes
around ~60-65 m/s for `large_vehicle` and ~80+ m/s for `car`). A smooth measurement-noise
tail would decay continuously; discrete clusters at specific velocity offsets is the
signature expected from Doppler/velocity ambiguity aliasing (a point's true radial velocity
exceeding the radar's unambiguous velocity limit wraps around to a specific PRF-dependent
offset, not a random value) rather than ordinary noise or track-ID contamination.

**Recommendation:** because `std()` is highly sensitive to a single extreme value, one
aliased point within an otherwise-normal instance can dominate that instance's `vr_std` and
corrupt the feature. A more robust per-instance dispersion measure - MAD (median absolute
deviation) or IQR of `vr_compensated` - would resist a single aliased outlier far better than
raw `std` and should be preferred over `vr_std` if this becomes an actual model feature
(rather than just an EDA statistic).

**MAD/IQR computed** (`results/eda/01d_vr_dispersion_robust.png`) - and this surfaced a real
bug, not just a robustness upgrade. First pass gave `pedestrian` a median MAD *and* IQR of
exactly `0.0`, despite having the highest `vr_std` of any class. Cause: `std` of a 1-point
instance is mathematically undefined and comes out `NaN` (correctly dropped), but MAD/IQR of
1 point both come out to a "valid" `0.0` (a point can't disagree with itself) and don't get
dropped. **54.4% of `pedestrian` instances have exactly 1 point** (81.6% have <=2) - see
item #2/#3 for why (weak, diffuse RCS below the CFAR threshold for most of the body, plus a
pedestrian's small angular extent often collapsing into a single radar resolution cell) - so
over half the class was silently contributing "measured zero dispersion" when the honest
answer is "not enough points to measure dispersion at all." Fixed by requiring
`n_points >= 3` before computing any of `vr_std`/`vr_mad`/`vr_iqr`, masking the rest to `NaN`
consistently across all three (previously only `vr_std` excluded low-count instances).

After the fix, all three measures agree with each other for the first time:

| class | median vr_std | median vr_mad | median vr_iqr |
|---|---|---|---|
| car | 0.080 | 0.017 | 0.053 |
| large_vehicle | 0.082 | 0.016 | 0.036 |
| pedestrian | 0.518 | 0.299 | 0.514 |
| pedestrian_group | 0.380 | 0.200 | 0.368 |
| two_wheeler | 0.381 | 0.188 | 0.359 |

`car`/`large_vehicle` cluster at the low end, `pedestrian`/`pedestrian_group`/`two_wheeler`
cluster ~5-10x higher, across every measure - a materially more trustworthy confirmation of
the micro-Doppler hypothesis than the original `vr_std`-only pass, since it no longer rests
on a single statistic that happened to handle low-point instances inconsistently. Boxplots
now show clean separation with no medians collapsed to zero.

## 5. Spatial extent (bounding-box size) as a range-robust size feature

The diagonal or area of an instance's `x_cc`/`y_cc` spread (already visualized as the
"tracked object" boxes in `dataloader.py`'s `plot_scene`) degrades less with range than
point count does - a candidate size feature that isn't as confounded by #3. Also used
in #1's sub-class comparison; here the question is whether it separates the 5 final
classes generally, not just the large_vehicle sub-classes.

**Findings:** _(TBD)_
