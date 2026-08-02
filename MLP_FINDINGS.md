# MLP Findings (Day 6 real run)

Results from actually running the design in `MLP_DESIGN.md` on the full data
(`scripts/train_mlp_full.py`), not the smoke test (`scripts/train_mlp.py`).
Findings only - design rationale lives in `MLP_DESIGN.md`, dataset/encoding
rationale in `DESIGN_DECISIONS.md`.

Setup: paper-faithful 3-layer MLP (`80 -> 16 -> 16 -> 5`), `lr = 1e-5`,
`batch_size = 256`, full train (354,277 instances) / val (67,663 instances)
splits, seed 42.

## 1. 1000 epochs vs. 20 epochs: the paper's epoch count is excessive here

First run: full 1000 epochs, ~23.2 min (`results/mlp_full_run/*_1000epoch.*`).
Second run: same everything, cut to 20 epochs, ~0.4 min
(`results/mlp_full_run/*_20epoch.*`) - deterministic (same seed), so this is
literally the first 20 epochs of the same trajectory, not a separate run.

| | 20 epochs | 1000 epochs |
|---|---|---|
| train_acc | 73.7% | 77.2% |
| val_acc (raw) | 74.1% | 75.6% |
| val_acc (balanced, macro recall) | 74.6% | 75.1% |
| wall-clock | 0.4 min | 23.2 min |

Val accuracy/loss are essentially flat past ~epoch 20 (`cost_1000epoch.png`,
`accuracy_1000epoch.png`) - the remaining 980 epochs buy on the order of
1-1.5pp raw accuracy for ~60x the training time. Not zero benefit (the
1000-epoch confusion matrix does show real, if small, gains concentrated in
`car` and `pedestrian_group` specifically), but nowhere close to proportional
to the extra epochs. **Decision: use the 20-epoch checkpoint going forward**
(`results/mlp_full_run/model_20epoch.pt`) rather than the paper's 1000 -
cheap to retrain if a later change (features, architecture) shifts this.

**Not overfitting at either length.** Train/val accuracy track each other
closely the whole way (train leads val by ~1.6pp at epoch 1000, both still
rising together, no divergence). If overfitting were the concern, more epochs
would be the wrong direction to worry about; it isn't the concern here -
the plateau is a ceiling, not a train/val split.

**Is the plateau a local minimum (i.e., would a different `lr` escape it)?**
Zooming into the first 20 epochs shows something worth noting: train/val
*loss* decreases smoothly and monotonically from epoch 1, but train/val
*accuracy* actually **dips** between epochs 2 and 5 (val_acc 0.543 -> 0.504)
before climbing back past its epoch-2 value around epoch 9 and continuing up.
This is not a sign of a local-minimum trap - loss (continuous, what the
optimizer actually descends) and accuracy (a non-differentiable argmax
readout) are not required to move together. A very early, lucky
near-random-guess accuracy from initialization can be higher than the
accuracy of a model that has since started genuinely reshaping its decision
boundaries but hasn't finished - the loss curve, not the accuracy curve, is
the honest signal of optimizer progress here, and it never stalls or reverses
at any point across all 1000 epochs. **Conclusion: don't decrease `lr`** -
a smaller step size would only slow arrival at the same plateau, not help
escape it; nothing in the shape of either curve looks like classic
local-minimum stagnation (an extended stall followed by a jump). The
plateau's likely cause is capacity/feature-separability (see below), not
optimization.

## 2. Gradient norms: no vanishing or exploding gradients

Checked directly rather than inferred from curve shape - the section 1 local-minimum
discussion argued from the loss curve's shape alone, so this closes the loop with a harder
measurement. Tracked the mean per-batch L2 gradient norm of each Linear layer's weight matrix
(`grad_norms_20epoch.png`), input-side to output-side.

All three layers stay within the same order of magnitude throughout training (roughly
0.05-0.35). The input-side layer (closest to the 80-dim feature vector - where vanishing
gradients would bite first) is the **largest** by epoch 20, not the smallest - the opposite of
the vanishing-gradient signature, where gradients shrink by orders of magnitude the further
back they propagate. The middle layer dips slightly relative to the other two but never
collapses toward zero. Nothing exceeds ~0.35 at any point - no explosion either. All three
trend mildly upward and track together, consistent with the loss curve's smooth, unbroken
descent.

**Conclusion: not a vanishing/exploding gradient problem.** Makes sense in hindsight - this is
a 3-layer, ~1.6k-parameter ReLU network, well below the depth where this typically bites
(mostly a 10+ layer or RNN problem). Confirms the section 1 conclusion (plateau is a
capacity/feature-separability ceiling, not an optimization pathology) with a direct
measurement instead of just curve-shape inference.

## 3. Per-class recall is very uneven - the 75% headline number hides this

Confusion matrix, 1000-epoch model (`validation_1000epoch.png`):

| class | recall |
|---|---|
| two_wheeler | 91.8% |
| pedestrian | 88.7% |
| car | 85.5% |
| pedestrian_group | 57.3% |
| large_vehicle | 52.0% |

Two specific, large, structural confusions - not noise:

- **`pedestrian_group` -> `pedestrian`** (37.6% of true `pedestrian_group`
  instances, 8,111 of 21,562). These two classes are separated mainly by
  point count/density in this feature space, and `pedestrian` is >54%
  single-point instances (`EDA.md`) - a sparse group can look identical to a
  single walker to a model with no explicit point-count feature.
- **`large_vehicle` -> `car`** (46.3% of true `large_vehicle` instances, 2,064
  of 4,456). `large_vehicle` merges truck/bus/train (`DESIGN_DECISIONS.md`
  decision 1) into one class with a much less homogeneous radar signature
  than `car`, and it's the second-smallest class even after loss weighting.

Both are consequences of the class-grouping and feature-set decisions
(decision 1, and the paper-faithful feature set not including anything
point-count-aware), not evidence of a training bug - the model converges
cleanly and isn't stuck, it just can't separate what the chosen features
don't distinguish well.

## Next steps this suggests (not yet decided)

- The `_rel`/spread features deferred to v1.1 (`vr_rel`, `extent_rel`,
  `rcs_rel`) were built specifically because point count/density (spread)
  carries information the raw paper features don't - directly relevant to
  the `pedestrian`/`pedestrian_group` confusion. Worth testing before
  revisiting the class-grouping decision.
- Revisit whether `large_vehicle`'s merge (decision 1) is costing more than
  it's saving, now that there's a real number attached (52% recall) rather
  than just the "not yet checked" caveat in decision 1.
