## 1. Confusion matrix (epochs=50 baseline, `results/mlp/mlp_confusion_matrix.png`)

Row-normalized, val set:

- `car`: 73% correct - the most accurate label, and the majority class. 10% confused as `large_vehicle`.
- `large_vehicle`: confused as `car` and `bus`.
- `bus`: confused as `large_vehicle`.
- `two_wheeler`: confused as `pedestrian` and `pedestrian_group`.
- `pedestrian`: recall high.
- `pedestrian_group`: confused a lot into `pedestrian` as expected, given point sparsity (an isolated pedestrian and a sparse group can look similar with few points to work with).

**Cross-check against `MLP_Design.md` decision 2:** the confusion isn't scattered - it clusters
within car/large_vehicle/bus (all "vehicle" classes) and separately within
pedestrian/pedestrian_group, rather than spreading across unrelated classes. That pattern reads
more like a feature-separability ceiling (the histogram features can't fully tell `bus` and
`large_vehicle` apart) than pure gradient starvation. It's also in tension with the starvation
story that `bus`'s recall (71%) is *higher* than `two_wheeler`'s (52%), despite `bus` being the
far rarer class (1.9% vs 7.0% of train) - if rarity/starvation were the dominant driver, that
ordering would likely be reversed. Oversampling may still help, but might not fully close the
bus/large_vehicle gap if the ceiling is feature overlap rather than exposure.
