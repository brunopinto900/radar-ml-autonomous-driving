I spent the last few weeks building and stress testing a 5 class radar object classifier on RadarScenes, radar point clouds only, no camera or lidar.

The most useful thing I found was not a modeling trick. It was where the model actually hits its ceiling.

Macro F1 roughly doubles just from going from 1 radar point per object to 5 (0.381 to 0.764), same trained model, no retraining, nothing else changed. I then tried ten different fixes: bigger and deeper networks, six different feature representations, different bin edges. All ten landed inside a measured noise floor. None of them moved the number that mattered.

One thing did work: stacking every previously tested feature together, additively, won every single one of 6 cross validation folds (mean +0.036 macro F1, about a 1.6% chance of that happening by luck). Real, but about a tenth the size of the sparsity effect.

Cross validation also caught something I almost missed: one class's instances can pile up in a handful of recording sequences, one tracked object seen over hundreds of scans, which inflates how noisy a single train/val split looks and, traced fully, explains part of the model's actual confusion pattern too.

Held out test set, touched exactly once at the very end: 0.699 macro F1 vs val's 0.686. No overfitting.

Full breakdown and the raw experimental log linked below if you want the details.
