## 1. Batch size and learning rate

Batch size affects whether rare classes contribute to each gradient update. bus makes up only 1.9% of the 358,210 train instances (vs. 42.8% for car); at that frequency, bus is absent from 54% of batches at bs=32 and 30% at bs=64.

**Decision:** use batch_size=128, the smallest candidate keeping every class's miss probability below 10%.

Scale the learning rate accordingly:

```
lr = 1e-5 × (128 / 32) = 4e-5
```

This follows the linear scaling rule and avoids slowing training through both larger batches and a reduced learning rate.

## 2. Oversampling for bus

Increasing training from 50 to 1000 epochs did not improve bus F1; it slightly decreased, while most other classes improved. Validation accuracy also plateaued after roughly 150 to 200 epochs.

This indicates that the problem is gradient starvation rather than insufficient training: bus remains poorly represented in individual batches.

**Decision:** test weighted oversampling for bus rather than increasing the number of epochs. Oversampling should replace or reduce the existing class weighting to avoid overcompensation.

**Decision:** retain EPOCHS=50. The additional 950 epochs produced only a small macro-F1 improvement and mostly added training cost and validation noise.
