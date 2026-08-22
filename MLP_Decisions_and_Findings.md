## Model config

**Architecture:** 3 layer MLP (2 hidden layers of 16 units each, ReLU), input 65 features, output 5 classes (`MLP_CLASSES`: `car`, `large_vehicle`, `two_wheeler`, `pedestrian`, `pedestrian_group`; bus merged into large_vehicle per decision 1's final resolution).

**Feature design:** per instance histogram encoding. `rcs`, `vr_compensated`, `x_rel`, `y_rel` each binned into N_BINS=16 fraction of points per bin columns (64 dims total), plus `doppler_spread` appended unbinned (1 dim), for 65 dims total. Bin edges use the 1st to 99th percentile range, fit on train only and reused unchanged for val/test.

**Training:** Adam optimizer, class weighted cross entropy loss (weight = max class count / class count). LEARNING_RATE=4e-5, EPOCHS=50, BATCH_SIZE=128, RANDOM_STATE=0.

**Data:** fixed sequence grouped train/val/test split (`Design_Decisions.md` decision 5).

## 1. Batch size and learning rate

Batch size affects whether rare classes contribute to each gradient update. bus makes up only 1.9% of the 358,210 train instances (vs. 42.8% for car); at that frequency, bus is absent from 54% of batches at bs=32 and 30% at bs=64.

**Decision:** use batch_size=128, the smallest candidate keeping every class's miss probability below 10%.

Scale the learning rate accordingly:

```
lr = 1e-5 × (128 / 32) = 4e-5
```

This follows the linear scaling rule and avoids slowing training through both larger batches and a reduced learning rate.

See `scripts/batch_size_selection.py`.

## 2. Epoch count: no gain past roughly epoch 200

Increasing training from 50 to 1000 epochs produced only a small macro F1 improvement (0.611 to 0.626 in the original 6 class taxonomy) and mostly added training cost and validation noise, validation accuracy plateaued after roughly 150 to 200 epochs. bus specifically got slightly worse with more training while most other classes improved, a gradient starvation signature (bus remained poorly represented in individual batches) that is now moot, since bus was merged into large_vehicle (see `Design_Decisions.md` decision 1).

**Decision:** retain EPOCHS=50.

## 3. Confusion matrix findings (epochs=50, original 6 class taxonomy, `results/mlp/bus_separate/mlp_confusion_matrix.png`)

Row-normalized, val set:

- `car`: 73% correct - the most accurate label, and the majority class. 10% confused as `large_vehicle`.
- `large_vehicle`: confused as `car` and `bus`.
- `bus`: confused as `large_vehicle`.
- `two_wheeler`: confused as `pedestrian` and `pedestrian_group`.
- `pedestrian`: recall high.
- `pedestrian_group`: confused a lot into `pedestrian` as expected, given point sparsity (an isolated pedestrian and a sparse group can look similar with few points to work with).

**Cross-check against section 2 above:** the confusion isn't scattered, it clusters within car/large_vehicle/bus (all "vehicle" classes) and separately within pedestrian/pedestrian_group, rather than spreading across unrelated classes. That pattern reads more like a feature-separability ceiling (the histogram features can't fully tell `bus` and `large_vehicle` apart) than pure gradient starvation. It's also in tension with the starvation story that `bus`'s recall (71%) is *higher* than `two_wheeler`'s (52%), despite `bus` being the far rarer class (1.9% vs 7.0% of train). If rarity/starvation were the dominant driver, that ordering would likely be reversed. This is what motivated the taxonomy experiment below.

## 4. Class taxonomy experiment (bus merged vs `bus_separate` vs `truck_separate`, 50 epochs each)

Macro F1: `bus_separate` (original 6 classes) 0.611, bus merged (5 classes, now `baseline`) 0.686, `truck_separate` (7 classes) 0.541.

Bus merged: folding `bus` into `large_vehicle` raises the combined class to F1=0.756, above either `bus` (0.543) or `large_vehicle` (0.492) alone in `bus_separate`. Supports merging `bus` in too, extending `Design_Decisions.md` decision 1's logic. This variant is now `mlp_variants.py`'s `baseline`, and its results live at `results/mlp/` directly rather than under the experiment subfolder.

`truck_separate`: splitting `truck` back out leaves pure `large_vehicle` at only 1,292 train instances, and it becomes nearly unusable (F1=0.098). Confirms decision 1's `truck` merge was correct.

Scripts: `scripts/class_taxonomy_experiment.py`, `scripts/mlp_variants.py`. Results: `results/mlp/` (baseline), `results/mlp/bus_separate/`, `results/mlp/class_taxonomy_experiment/truck_separate/`.

This directly informed `Design_Decisions.md` decision 1's final resolution: merge `bus` into `large_vehicle` too.
