# Feature-map scatter findings

Separate from `EDA.md`'s numbered plan/findings structure - these are 2D scatter plots
looking at joint feature relationships (not single-feature distributions), generated from
`scripts/eda.py`'s `eda_2_rcs_vs_point_count_scatter*` functions. Uses
`results/train_points.parquet` only, same split discipline as `EDA.md`.

## 02c — mean RCS vs. point count, by final class

`results/eda/02c_rcs_vs_point_count_scatter.png`. The literal deliverable named in
`TODO.md`'s Day 4 line ("2D scatter plots ... to eyeball separability") - missed in the
first pass at Day 4 and added after the gap was caught.

At low point counts (1-10, where the vast majority of every class's data actually lives),
**all 5 classes heavily overlap across nearly the entire RCS range**. This is a materially
different picture than `EDA.md`'s marginal RCS histograms (item #1/#2) suggested - those
showed separated *medians*, but this joint per-scan scatter shows no clean visual cluster
structure at the individual-scan level. A single scan's mean RCS combined with its point
count doesn't cleanly separate the classes by eye - the color cloud in the low-point-count
region is thoroughly mixed.

One structural signal does show up clearly: only `large_vehicle` extends to high point
counts (beyond ~20, up to 45) - no other class reaches there at all. Point count has some
discriminative value at the extreme tail (very high count essentially rules out everything
but `large_vehicle`), even though it doesn't separate classes in the typical low-count
regime where most data lives.

**Takeaway:** RCS's separability (established elsewhere in `EDA.md`) is real but
statistical/aggregate - it shifts each class's distribution center, which is why medians
ordered cleanly - not something visible as separated clusters from two features alone.
Useful expectation-setter for Day 5/6: this is why an MLP over the full point cloud is the
right tool here, not a simple 2-feature threshold rule.

## 02d — same scatter, colored by range instead of class

`results/eda/02d_rcs_vs_point_count_by_range.png`. Same axes (mean RCS vs. point count, per
scan), all classes pooled, colored by `mean_range` instead of class - checks how much of
02c's overlap is a range effect rather than a class effect.

Strong, counter-intuitive pattern: at any fixed point count, color shifts clearly with
height. High RCS (+20 to +40) skews toward longer range (60-100m, green/yellow); low RCS
(-20 to -30) skews toward short range (<20m, dark purple). **Farther objects show higher
RCS, closer objects show lower RCS** - backwards from raw uncompensated received power
(which drops with range per the radar equation). Since `rcs` is meant to already be a
range-compensated quantity, a residual correlation this strong is worth taking seriously.

### Follow-up: mean range per class (resolves the open range-per-class caveat)

Flagged as an open, unchecked caveat in both `EDA.md` items #3 and #5 ("if classes differ in
their typical recording range in this dataset, part of any gap between them could still be
range, not size") - finally checked directly:

| class | mean range [m] | median range [m] |
|---|---|---|
| two_wheeler | 20.7 | 15.1 |
| pedestrian | 29.9 | 28.9 |
| pedestrian_group | 37.1 | 36.8 |
| car | 41.3 | 37.6 |
| large_vehicle | 47.3 | 43.1 |

This ordering **tracks the class RCS ordering almost exactly** - `two_wheeler` is both the
closest-range class and the lowest-RCS class; `large_vehicle` is both the farthest-range
class and the highest-RCS class. So yes, classes do differ substantially in typical
recording range, confirming the long-standing open caveat, and this substantially explains
02d's pooled range-RCS gradient as a class-mix effect rather than a pure sensor artifact.

**Important nuance on causality, not just "confound, discount it":** this doesn't
necessarily mean the RCS-class relationship is spurious. Radar detection range itself
depends on RCS - for a fixed noise floor/CFAR threshold, maximum detection range scales
with RCS to the 1/4 power (radar equation). A genuinely higher-RCS object (like a truck) is
detectable and gets tracked from farther away than a genuinely low-RCS object (like a
pedestrian, who can usually only be detected once close). Under that reading, range isn't an
independent confound corrupting the RCS-class relationship - it's a *second, physically
expected consequence* of the same true underlying RCS difference between classes, not a
separate spurious correlation to net out. Both readings (class-mix confound vs. shared
physical cause) are consistent with the data here; distinguishing them would need something
like within-class RCS-vs-range slopes, not done.

## 02e — within-class RCS-vs-range trend (resolves the confound-vs-physics question above)

`results/eda/02e_rcs_vs_range_by_class.png`. Median RCS vs. `range_sc`, one line per final
class, per point - the diagnostic named as the way to resolve 02d's open question. A flat
within-class line would mean class-mix fully explains the pooled gradient; a trending line
means there's a real per-object range effect on top of it.

Every class trends, and trends strongly: roughly +15 to +18 dB from 5m to 90m, essentially
the same slope for all five (not perfectly parallel - see 02f). This rules out "pure
class-mix" - there's a real, common, per-point range effect baked into `rcs` itself, on top
of whatever class-mix contributes.

**Likely mechanism**, tying back to `EDA.md` item #3's point-count-vs-range confound -
probably the same root cause (detection-threshold selection) showing up twice: at short
range, both strong and weak scattering points on an object clear the detection threshold. At
long range, only the object's strongest local scattering points still clear it - weaker
points fall below the noise floor and go unreported. The points that *survive* to be
measured at long range are a biased-high sample of that object's own point-level RCS
distribution - not because the object got more reflective, but because detection is
selectively keeping only the strongest returns as range increases.

Because the trend is close to the same slope for every class, the between-class *vertical
gaps* stay roughly constant across most of the range axis - `large_vehicle` sits ~3-5 dB
above `car`, which sits ~5-8 dB above the pedestrian/two_wheeler cluster, fairly
consistently. That suggested range-correcting RCS was worth trying: a shared correction
should remove a substantial noise source (~15-18 dB, larger than most between-class gaps)
without erasing the separation between classes.

## 02f — range-detrended RCS: does it work?

`results/eda/02f_rcs_range_detrended.png`. Fit one shared, non-parametric range trend -
pooled (all classes) median RCS per range bin, used as "expected RCS at this range" - and
subtracted it per point. Non-parametric on purpose: 02e's curve isn't linear (steep at short
range, flattening at long range), so an empirical per-bin median avoids assuming a
functional form. Checked two things on the residual.

**Left panel - is the within-class trend now flat? Not fully.** The single pooled correction
doesn't fully flatten every class - there's a real class x range *interaction* the shared
trend can't capture. `large_vehicle` starts high (+7 dB residual at 5m) and steadily loses
that advantage down to ~+1 dB by 90m. `pedestrian`/`pedestrian_group`/`two_wheeler` start
around -1 to -2 dB and sink further to -6 dB by 90m. Visible in 02e too, in hindsight -
`car` and `large_vehicle`'s lines converge and nearly cross by 90m rather than staying
parallel, which the "roughly constant gaps" reading above glossed over before checking.

**Right panel - does class separation survive anyway? Yes, clearly.** Even with the
imperfect correction, `car`/`large_vehicle` still sit shifted right (higher residual RCS,
extending into +10 to +25 dB) versus `pedestrian`/`pedestrian_group`/`two_wheeler` clustered
left (peak around -2 to -3 dB) - the same broad two-cluster structure as the raw RCS
separation in `01b`.

**Verdict:** range-correcting RCS is safe and doesn't destroy the class signal, even with a
crude shared correction. But a *better* correction would model the range trend per class (or
with a class interaction term) rather than one pooled curve, since `large_vehicle`
specifically behaves differently from the rest. Noted in `TODO.md`'s future work rather than
pursued further here - not needed to keep moving through EDA.

## Range-compensated re-run of every RCS plot (`_range_comp_rcs` suffix)

Prompted by a direct challenge to the range-RCS finding above ("i am not trusting this
results") - independently re-verified the raw relationship first (three checks bypassing all
plotting/binning code: overall Pearson correlation +0.47, within-`car`-only correlation
+0.42 computed via a fresh one-line `.corr()`, and the 10 closest/farthest points in the
whole dataset inspected directly with zero aggregation - all confirm the same effect, no bug
found). With the effect confirmed real, `eda.py`'s RCS-consuming plot functions
(`eda_1_large_vehicle_merge`, `eda_2_rcs_vs_azimuth`, `eda_2_rcs_vs_point_count_scatter`,
`eda_2_rcs_vs_point_count_scatter_by_range`) were parameterized with `rcs_col`/`suffix` and
re-run against `rcs_range_comp` (the same per-point range-compensation from 02f, via the
shared `add_range_compensated_rcs()` helper) instead of raw `rcs`, producing a full parallel
set of plots (`results/eda/*_range_comp_rcs.png`) to check whether any conclusion changes.

**None of them do, qualitatively:**

- **`01a_..._range_comp_rcs`** (sub-class merge check): `train` still stands apart with a
  right-shifted, narrower peak; `bus` still shifted left/lower; `truck`/`large_vehicle` still
  overlap closely with each other - same pattern as the raw-RCS version. Whatever drives the
  sub-class RCS differences, it isn't explained by these sub-classes sitting at different
  typical ranges.
- **`01b_..._range_comp_rcs`** (final-class separation): `car`/`large_vehicle` still shifted
  right, `pedestrian`/`pedestrian_group`/`two_wheeler` still clustered left - same two-cluster
  structure as raw RCS and as 02f's pooled residual check, now confirmed again per-class
  alongside the (unchanged, since they don't use RCS) Doppler-spread and extent panels.
- **`02a_..._range_comp_rcs`** (RCS vs. azimuth) and the `02c`/`02d` scatters: not
  individually re-described here since they weren't re-examined in detail, but generated for
  completeness and available if the azimuth-hump question (item #2) needs revisiting with
  range-compensated RCS specifically.

**Follow-up that closed the azimuth-hump question:** the RCS-vs-range relationship
established here (02e/02f) turned out to fully explain `EDA.md` item #2's azimuth "hump" -
range itself varies with azimuth (median range peaks at 20-30°, the same place RCS peaked,
and collapses to <10m at wide angles, where RCS also dropped), so the chain is
azimuth -> range -> RCS, not a shared antenna-gain effect as item #2 had tentatively
concluded. Full writeup and the median-range-per-azimuth-bin table are in `EDA.md` item #2
(search "Correction: the shared sensor effect") rather than duplicated here, since it's
really an azimuth-item finding that happens to depend on this file's range work.

**Net conclusion:** the RCS-class separation found throughout `EDA.md`/this file is not an
artifact of the range-RCS relationship - it survives range compensation essentially
unchanged. Given that, and given the per-class detrending refinement is already logged as
future work rather than urgent, raw `rcs` remains a reasonable feature to carry into Day 5/6
as-is; range-compensation is a documented, available option, not a required preprocessing
step based on what's been checked so far.

## 02g — mean RCS vs. Doppler spread, by final class (the two strongest features, jointly)

`results/eda/02g_rcs_vs_vr_std_scatter.png`. RCS (item #2/this file) and Doppler spread
(`EDA.md` item #4) were each established as the strongest single feature independently, but
never checked together until now. Per scan; restricted to `n_points >= 3` (same masking as
item #4 - `vr_std` is undefined below that), so this necessarily has less data than 02c,
especially for `pedestrian` (>54% single-point scans excluded).

**First pass** turned up a bonus finding, independent confirmation of the item #4 aliasing
hypothesis: distinct vertical bands were visible around `vr_std` ~17 and ~18.5 - not a
smooth scatter. Matches the PRF-dependent wraparound signature inferred earlier from the
log-scale `vr_std` histogram tail (`01c`), now visible directly in a completely different
plot (a 2D scatter rather than a 1D histogram), which strengthens that this is a real,
discrete-offset artifact and not a quirk of how `01c` happened to bin the histogram. A
per-scan Doppler spread of 17+ m/s isn't physically plausible for any of these classes -
correctly read as confirmation the aliasing story is real, not as a finding about the
classes themselves.

**Filtered to `vr_std <= 3.0` m/s** to drop that artifact and look at the physically
plausible range on its own merits (excludes only the aliased tail, not a meaningful chunk of
real data - see the plot for the resulting distribution shape). Separability: still a dense,
heavily overlapping cloud at the low end - no clean 2D clusters by eye, same conclusion as
02c. But real structure exists within it: `pedestrian`/`pedestrian_group`/`two_wheeler`
cluster tightly at low `vr_std` (0-1) with mostly negative RCS (-5 to -25); `car`/
`large_vehicle` spread across the full 0-3 range and skew toward higher RCS (0 to +30),
though `car` also has plenty of points overlapping the low-`vr_std` pedestrian-type cluster.
RCS's vertical separation (vehicle classes higher, pedestrian-type classes lower) holds up
throughout the plausible range, consistent with every other RCS check in this document.

## 02h — same scatter, using MAD instead of std

`results/eda/02h_rcs_vs_vr_mad_scatter.png`. Same joint check as 02g (mean RCS vs. Doppler
dispersion, per scan, `n_points >= 3`), but with `vr_mad` in place of `vr_std` - the more
robust dispersion statistic recommended by item #4 (a single aliased point pulls `std` far
more than it pulls a median-based measure). No hardcoded cutoff needed here: unlike `vr_std`
(aliasing tail out to ~19 m/s, forcing the `<= 3.0` cutoff on 02g), `vr_mad`'s own 99.5th
percentile lands around ~4 m/s, so a plain percentile-based xlim is enough - the aliasing
artifact is naturally suppressed by using a robust statistic rather than filtered out after
the fact.

Qualitatively the same picture as the filtered 02g: dense, heavily overlapping cloud at low
dispersion, no clean 2D cluster separation by eye. `car`/`large_vehicle` skew toward higher
RCS and spread further out along the dispersion axis; `pedestrian`/`pedestrian_group`/
`two_wheeler` cluster tighter at low dispersion with mostly negative RCS. No new
separability insight beyond 02g - the value here is confirming the relationship is real and
not an artifact of which dispersion statistic was used, and demonstrating in practice why
MAD is the better choice to actually carry into Day 5/6 (robust by construction, no manual
cutoff required).

**Note:** `vr_mad`/`vr_iqr` computation is commented out in `instance_features` (and the
`eda_2_rcs_vs_vr_mad_scatter` call in `__main__`) after generating this plot, for the same
reason `eda_vr_dispersion_robust`/`01d` is commented out - it's slow and not needed again
until it's time to actually pick the Day 5 feature set.

## Per-instance feature histograms (Day 5 - `scripts/instance_histograms.py`)

Everything above this section works on hand-picked scalar statistics (`mean_rcs`, `vr_std`,
`extent`, ...). Day 5's actual task is the histogram-encoding scheme itself - the same idea
Tatarchenko & Rambach use (see `TODO.md` Day 8): instead of a few hand-picked scalars per
instance, compute a raw-count histogram (K=8 bins) of every point's value for each feature,
per instance, and let those bins be the model's input vector. This section checks whether
that representation actually looks separable before committing to it, using a handful of
example instances per class (`n_points >= 5`, 8 examples per class, seed 42) rather than the
full dataset - a visual sanity check, not a population-level claim (see item #4's median
`vr_std`/`vr_mad` table for the population-level numbers this cross-checks against).

Visualization note: showing each of the 8 example instances as its own overlaid step outline
produced a cluttered stack of many small overlapping rectangles, not a readable shape. Fixed
by plotting the *mean* histogram (± std across the 8 instances) as solid bars - one clean
column set per class, with the error bars still showing instance-to-instance variability.

### Raw per-point-value histograms: good for RCS, degenerate for position

`results/instance_histograms/{rcs,vr_compensated,range_sc,azimuth_sc,x_cc,y_cc}_instance_examples.png`.

- **`rcs`**: real payoff. Same-class example instances share a recognizable envelope (`car`
  peaked -10 to 0 dBsm, `large_vehicle` similar but a longer tail past +20, `pedestrian`/
  `pedestrian_group` both tight and low, `two_wheeler` most concentrated at the bottom) -
  different instances of the same class look like variations on one shape, different classes
  look visually distinct.
- **`vr_compensated`**: unstable for `car`/`large_vehicle` - the bump sits in a different place
  along the velocity axis almost every instance, since raw bulk velocity is behavioral (how
  fast this particular vehicle happened to be going), not structural. `pedestrian`/
  `pedestrian_group` are tightly and consistently peaked near 0 across every example.
- **`range_sc`/`azimuth_sc`/`x_cc`**: collapse to a near-single-bin spike within one instance,
  because one object's own physical footprint (a few meters) is tiny compared to the sensor's
  ~100m range - only the *location* of the spike varies instance-to-instance, not its shape.
  A per-instance histogram of a position feature carries essentially the same information as
  just the scalar mean position would; the histogram encoding buys nothing extra here.
- **`y_cc`** is the one exception among the position features: the spike location is
  noticeably more consistent within a class (e.g. nearly all `large_vehicle` examples spike
  around -5 to -10) than `range_sc`/`x_cc`/`azimuth_sc` are - still a single-bin spike
  structurally, but a more class-diagnostic one, plausibly reflecting which side of the
  road/which lane a class tends to occupy relative to this specific sensor.

### The environment-vs-shape problem, and the fix: relative (de-meaned) features

Raised directly: *"we are not really encoding shape, we are just encoding environment"*. Raw
`range_sc`/`azimuth_sc`/`x_cc`/`y_cc` mostly tell you *where in the world* an object was
recorded, not what it looks like - same root problem as the point-wise position features
discussed earlier in this project. Fix: histogram each feature *relative to that instance's
own mean* instead of the raw/absolute value - the same move Tatarchenko & Rambach make with
their Cartesian coordinates (their quote: *"the object shape in Cartesian coordinates is
independent of the distance between the object and the radar sensor"*), just applied without
needing their upstream tracker since we already have `track_id` for training-time grouping.

**Mechanism, concretely, using azimuth as the example (`azimuth_sc` vs. `azimuth_rel`):**
`azimuth_sc` is the point's raw angular position as measured by the sensor - an absolute
bearing in the sensor's own frame. Histogramming raw `azimuth_sc` per instance bins those
absolute bearings - and since one object's own angular footprint is only a few points wide
compared to the sensor's full ~140° FOV, essentially every point in an instance lands in the
same 1-2 bins, wherever that object happened to be in the FOV. That's *where the object is*,
not what it looks like - the "degenerates to a single-bin spike" problem above.
`azimuth_rel[i] = azimuth_sc[i] - mean(azimuth_sc over the instance)` re-centers every point on
the *instance's own average bearing* instead of the sensor's zero. What's left after
subtracting that instance-specific reference is each point's deviation from its own object's
center - i.e. the object's own angular width, independent of where in the FOV it was. This is
the general recipe behind every `_rel` feature in this section: subtract the instance's own
central tendency (mean, or median for `vr_rel` - see below) from each raw value, then histogram
what's left. "Spread" is exactly that residual distribution's shape - concentrated near zero
across few bins means points barely disagree with each other (tight/rigid), spread across many
bins further from zero means points disagree a lot (wide/articulated or, per the correction
above, orientation-dependent).

`results/instance_histograms/{rcs,vr,range,azimuth,x,y,extent}_rel_instance_examples.png`.

- **`x_rel`/`y_rel`/`azimuth_rel`: this works.** Once centered on the object's own centroid,
  the spread tracks real physical size - `large_vehicle` clearly widest in both `x_rel` and
  `y_rel`, `pedestrian` tightest (near-single-spike), `pedestrian_group` visibly wider than
  lone `pedestrian` (matches item #5's extent numbers), `two_wheeler` narrow. `azimuth_rel`
  shows the same ordering from an independent (angular, not Cartesian) axis.
- **`extent_rel`** (`sqrt(x_rel**2 + y_rel**2)`, i.e. each point's distance from the instance's
  own centroid) was added to make the spatial-extent signal rotation-invariant - `x_rel`/
  `y_rel` individually are tied to the sensor's own axes, so an object's "spread in x" vs.
  "spread in y" partly reflects its orientation relative to the sensor, not just its shape.
  This is the cleanest, best-separated feature in the whole set: `car` peaks 0-1m tapering by
  3m, `large_vehicle` clearly broadest (real mass out to 6-8m), `pedestrian` a single tight
  bar at 0-1m, `pedestrian_group` visibly two-bin-wide, `two_wheeler` narrow. Bin width (~1m)
  is well-matched to the real scale of the signal here, unlike the two issues below.
  **Caveat inherited, not fixed:** de-meaning removes the *environment* confound but not the
  *range-degeneracy* one from item #3 - at long range every class's own points still collapse
  toward a single point regardless of true size, so `extent_rel` should still be expected to
  wash out for far-away instances.
- **`rcs_rel`/`range_rel`** included in the same spirit (RCS spread ties to item #1's `bus`/
  specular-glint observation; range spread could hint at object orientation/aspect angle) but
  not separately characterized here.

### `vr_rel`: two real bugs caught building it, both fixed

De-meaning `vr_compensated` per instance should recover Doppler *spread* (the micro-Doppler
signal from item #4) as a full histogram shape instead of a single `vr_std`/`vr_mad` scalar -
but the first two versions were both broken, for different reasons.

1. **Aliasing outliers, even after de-meaning - one bad point corrupts the whole instance, not
   just itself.** Centering on the instance's own *mean* doesn't fix the item #4 aliasing
   problem, it spreads it. Example: a `car` instance's true velocities are
   `[-2, -3, -2.5, -2, -3]` (tight, rigid, mean ≈ -2.5). If one point is actually aliased and
   reads `+18` instead of its true value, the set becomes `[-2, -3, -2.5, -2, +18]` and the
   mean jumps to `+2.1`. Subtracting that corrupted mean from *every point* turns the four
   genuinely tight points - which should de-mean to ~0 - into `-4.1, -5.1, -4.6, -4.1`: a
   perfectly rigid instance now looks like a widely-dispersed, articulated one. One aliased
   point didn't just corrupt its own value, it dragged the reference point for the whole
   instance (`large_vehicle` showed isolated spikes at -13 and +18 m/s in the first pass,
   consistent with this). Same fix as item #4's `std`-vs-`MAD` recommendation: switched the
   centering from mean to **median** (`CENTERING = {"vr_rel": "median"}` in
   `instance_histograms.py`) - a median barely moves when one of several values is extreme, so
   for the example above the median stays `-2`, de-meaning gives `[0, -1, -0.5, 0, +20]`: the
   four good points correctly land near zero, and the corruption stays localized to the one
   point that actually caused it.
2. **Bin width too coarse to show the real signal, even after fixing #1 - wider bins mean
   less resolution, and these were far wider than the signal being measured.** Median-centering
   stops one bad point from corrupting the *rest of its own instance*, but does nothing about
   how the bin edges themselves are chosen - those still came from the 0.5th-99.5th percentile
   of `vr_rel` across the *entire* dataset, genuinely wide (~-15 to +30 m/s) because rare
   fast-crossing traffic and leftover aliased points elsewhere in ~1M points really do reach
   that scale. Spread ~45 m/s across 8 bins and each bin is ~5.6 m/s wide. Item #4's real class
   gap is sub-1-m/s (`car` median std 0.08 vs. `pedestrian` median std 0.52) - the entire gap
   fits inside half a meter-per-second, so against a 5.6 m/s bin both classes round to
   "essentially zero, dead center." A `car` instance and a `pedestrian` instance, despite
   genuinely different physical behavior, would both just show one tall spike in the middle bin
   - not because the classes don't differ, but because the ruler had centimeter-scale objects
   marked on a meter-scale ruler. Fixed with an explicit range override
   (`EXPLICIT_RANGES = {"vr_rel": (-3.0, 3.0)}`, same cutoff already used for the `vr_std`
   scatter in `02g`), giving ~0.75 m/s bins - close enough to the real scale of the signal to
   actually resolve it.

   **Why this would have been dangerous left undetected:** it wouldn't have failed loudly.
   Training would still converge and reach decent overall accuracy, since `rcs`/`extent_rel`
   carry real signal on their own - gradient descent would just correctly learn that `vr_rel`'s
   8 bins carry ~no class-discriminative variance and quietly stop using them, even though
   Doppler spread is a real, validated, physically-grounded signal. There'd be no obvious
   symptom pointing at `vr_rel` specifically - just a model somewhat worse than it should be at
   separating classes that lean on Doppler spread, with the natural (wrong) explanations being
   "need more data" or "the histogram approach doesn't capture this well," not "one feature's
   bin width was miscalibrated by ~5x."

**Result after both fixes**: `car` tightly concentrated in the two central bins with
near-zero mass beyond ±0.75 m/s (rigid body). `large_vehicle` has taller central bins *and* a
real bar out at +1.5 to +2.25 with large error bars - the sub-class mixture signature again
(`train` dragging a wider tail against an otherwise `car`-like `truck`/`bus`/`large_vehicle`
population, per item #1). `pedestrian`/`pedestrian_group` both show real mass spread across
nearly the whole ±2 m/s range, matching the articulated-body hypothesis. `two_wheeler` sits in
between. That ordering now tracks item #4's MAD numbers reasonably well - still coarser than
the finest real gaps, but showing real signal instead of hiding it.

### Remaining structural limitation: histograms have no cross-feature correspondence

Not fixed by the relative-feature extension, and not fixable without a different architecture.
A histogram counts how many points fall in each bin *per feature, independently* - it never
records that a specific high-RCS point was also the point with the odd velocity, or where on
the object it physically sat relative to the others. Two objects with identical marginal
RCS/Doppler distributions but completely different spatial layouts produce identical
histograms. Tatarchenko & Rambach are explicit that this is a deliberate trade, not an
oversight (their quote: *"Instead of preserving per-point correspondences between individual
features as done in DeepReflecs, we replace the raw point clouds with high-level feature
statistics. The fact that such a seemingly severe transformation does not lead to any
performance drop... indicates that point-wise correspondences between features may not be
crucial for the task"*) - evidence from their sensor/task/dataset, not a guarantee it holds
here. Real shape-preserving alternatives exist (occupancy grids - Lombacher et al., cited in
the same paper; PointNet-style architectures) but are meaningfully bigger builds than a
histogram-to-MLP pipeline, out of scope for this sprint.

### Verdict for Day 6

Genuinely separable as histograms: `rcs`, `extent_rel` (clean, rotation-invariant, but expect
range-degeneracy at long range per item #3), `vr_rel` with median-centering and the ±3 m/s
range clip (real signal, still coarse).

**Correction: `azimuth_rel` is not redundant with `extent_rel` - they measure different
things, on purpose.** `extent_rel` is rotation-invariant by construction (distance from
centroid doesn't change when a rigid object is reoriented), so it answers "how big is this
object" independent of heading. `azimuth_rel` answers a different question: how much angular
width the object's silhouette presents to the sensor *right now*, which depends on
orientation as much as size - a car broadside to the sensor shows a large azimuth spread and
small range spread; the same car nose-on shows the reverse (its length now runs along range,
which `azimuth_rel` can't see but `range_rel` would). So `azimuth_rel`/`range_rel` together
could carry real orientation-relative-to-sensor information that `extent_rel` deliberately
discards - potentially useful if orientation correlates with class for behavioral reasons
(e.g. how a `two_wheeler` typically presents itself in traffic vs. a parked/turning `car`).
That's also exactly the same *scene/behavioral* confound risk already flagged for
`range_sc`/`x_cc`/`y_cc` in the Day 6 ablation plan - not a fixed physical property of the
class, a property of how this dataset's scenes happen to be composed. Verdict: keep
`azimuth_rel`/`range_rel` as real candidates alongside `extent_rel` rather than dropping them
as redundant, and let the same physics-only-vs-full-feature-set ablation (Day 6) determine
empirically how much they're contributing versus shortcutting.

Raw `range_sc`/`azimuth_sc`/`x_cc`/`y_cc` (not the `_rel` versions) are still dead weight as
histograms - degenerate to a single-bin spike, no better than their own scalar mean. All of
this still depends on `track_id`-based grouping at training time, and doesn't solve the
real-world clustering gap (`TODO.md`'s v2 future work) or the cross-feature-correspondence
limitation above.
