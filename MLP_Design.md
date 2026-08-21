## 1. Batch size and learning rate: derived from class frequency, not guessed

Batch size determines how many training instances contribute to one gradient update. A rare
class can be entirely absent from a batch - contributing nothing to its weighted loss term that
step. For a class that's a fraction `p` of the training set, the probability a batch of size
`bs` misses it entirely is `(1-p)^bs` (binomial P(zero successes)).

Train split (358,210 instances) class frequencies and miss-probability by candidate batch size:

| class | freq | bs=32 | bs=64 | bs=128 |
| --- | --- | --- | --- | --- |
| car | 42.8% | 0.000 | 0.000 | 0.000 |
| large_vehicle | 7.0% | 0.097 | 0.009 | 0.000 |
| bus | 1.9% | 0.544 | 0.296 | 0.088 |
| two_wheeler | 7.0% | 0.099 | 0.010 | 0.000 |
| pedestrian | 15.8% | 0.004 | 0.000 | 0.000 |
| pedestrian_group | 25.5% | 0.000 | 0.000 | 0.000 |

`bus` is the binding constraint: at bs=32 it's missing from 54% of batches, and even bs=64 still
misses it 30% of the time.

**Decision:** batch_size=128 (smallest power of two keeping every class's miss-probability
<=10%), with learning rate scaled up from the original hand-picked, empirically-stable baseline
(batch_size=32, lr=1e-5) via the linear scaling rule - a bigger batch means fewer, less noisy
gradient updates per epoch, so LR should grow with it, not shrink, to avoid compounding the
slowdown twice:

```
lr = 1e-5 * (128 / 32) = 4e-5
```

Computed by `scripts/batch_size_selection.py` (prints the table above and the recommendation;
does not write to `mlp_classifier.py`'s hyperparameter MACROS itself).
