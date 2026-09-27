# respiration

Estimating ventilatory thresholds from CPET data with an LSTM.

An incremental cardiopulmonary exercise test (CPET) crosses two thresholds: the
**anaerobic threshold (AT)** and the **respiratory compensation point (RC)**.
Locating them is work a sports physician does by hand, reading the plots. Here
the problem is framed as per-breath classification: given the time series of
respiratory variables, label every instant with the zone it falls in.

| class | meaning | share of samples |
|---|---|---|
| 0 | below AT | 30.1% |
| 1 | between AT and RC | 15.5% |
| 2 | above RC | 54.4% |

## The data

82 Excel files in `data/File_CPET/`, one per subject, exported by the metabolic
cart. Each file has four sheets: `Test` (the breath-by-breath series), `AT` and
`RC` (the timestamps of the two thresholds, marked by an operator), and
`Media Rest` (resting values, used for normalisation).

- 43199 breaths in total
- sequences from 80 to 1090 steps, median 508
- every file is a different subject, so one sequence equals one subject

`resp-prepare-data` packs them into `data/sequence_dataset.pkl`, normalising
each column by its own resting value and deriving the labels from the AT and RC
timestamps.

## The pipeline

The three stages must run in this order, because each one produces the
configuration for the next.

```
resp-prepare-data      xlsx -> data/sequence_dataset.pkl
      |
resp-select-features   compares the 15 feature groups in 5-fold CV
      |                -> conf/features/selected.yaml
      |                -> results/feature_results.json
resp-tune              Optuna on the winning group, 5-fold CV
      |                -> conf/model/tuned.yaml, conf/optimizer/tuned.yaml
      |                -> results/best_params.json
resp-train             final training and measurement on the test set
                       -> results/test_results.json
                       -> checkpoints/threshold_estimator.pt
```

There is also `resp-experiment-load`, which is not part of the pipeline: it
answers the question about the scale of `Load` described below.

The generated files are ordinary Hydra config groups, versioned in the repo and
selected in the `defaults` of `config.yaml`. They are written by the scripts and
should not be edited by hand: copying tuning results into `config.yaml` manually
is what previously produced inline overrides inconsistent with the group files.
Because they are groups and not global overrides, an explicit choice on the
command line still wins over them.

`batch_size` lives in the `optimizer` group alongside `lr` and `weight_decay`:
it is an optimisation hyperparameter like the others, and the value found by the
tuner needs somewhere to be written.

### How the data is split

`src/respiration/splits.py` holds back **20% of the subjects (17) as a test
set** and divides the remaining 65 into 5 folds. Feature selection,
hyperparameter search and early stopping work only on the folds. The test set is
read once, at the very end.

This is necessary, not pedantry: with 13 subjects per fold the F1 swings by up
to 0.16 from one fold to another. A single split could return 0.86 or 0.70 for
the same model depending on how it happened to fall.

## Running it

The project is an installable package: an editable install puts the commands on
PATH and makes imports independent of the directory you launch from.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

Then, from any directory:

```bash
resp-prepare-data        # only if the pickle needs regenerating from the xlsx
resp-select-features
resp-tune
resp-train
pytest
tensorboard --logdir runs
```

Or through the Makefile targets: `make install`, `make data`, `make features`,
`make tune`, `make train`, `make test`, `make all`.

Everything is configured in `src/respiration/conf/config.yaml` and overridable
from the command line, following Hydra conventions:

```bash
resp-train optimizer=sgd model=tenet_lstm epochs_final=10
resp-train features=columns_all optimizer.batch_size=8
resp-tune n_trials=100 search.hidden_max=128
```

To ignore the stored results and start from the defaults:
`features=columns_all model=tenet_lstm optimizer=adam`.

The configs live inside the package because Hydra resolves `config_path` as a
module: with an editable install they are still plain files you can edit on
disk. Data paths do not depend on the current directory but on the
`${project_root:}` resolver, registered in `respiration/paths.py`.

## Results

### Which features matter

15 combinations of the four groups, 5-fold CV, the same baseline model for all
of them so that the comparison measures the features and not the
hyperparameters.

| group | mean F1 | n. features |
|---|---|---|
| **respiratory + cardiac + metabolic + load** | **0.7924** | 8 |
| respiratory + metabolic + load | 0.7898 | 6 |
| respiratory + metabolic | 0.7861 | 5 |
| respiratory + cardiac + metabolic | 0.7844 | 7 |
| cardiac + metabolic + load | 0.7573 | 5 |
| cardiac + metabolic | 0.7474 | 4 |
| metabolic | 0.7471 | 2 |
| metabolic + load | 0.7396 | 3 |
| respiratory + cardiac + load | 0.7113 | 6 |
| respiratory + cardiac | 0.7065 | 5 |
| cardiac + load | 0.6864 | 3 |
| respiratory + load | 0.6014 | 4 |
| respiratory | 0.5964 | 3 |
| cardiac | 0.5500 | 2 |
| load | 0.5238 | 1 |

The highest number belongs to the full group, but **the top four are tied**:
0.7924 down to 0.7844 is a 0.008 spread, inside a fold-to-fold noise worth
0.08-0.09 on this data. Claiming the full group beats the other three is not
supported by the numbers.

On parsimony: `respiratory + metabolic` reaches 0.7861 with five features
(`Rf, VE/VO2, VE/VCO2, VO2, VCO2`), indistinguishable from the full eight. And
`metabolic` on its own, meaning **`VO2` and `VCO2` and nothing else**, reaches
0.7471: it gives up 0.045 against eight features while using two. Almost all the
signal is in gas exchange; the rest adds little.

The `load` group **on its own** is the worst of all (0.5238): power output
relative to the subject's own peak is not enough, by itself, to place the
thresholds.

### `Load`, and a bigger problem: the network follows scale

`data/cpet.py` normalises every column by its resting value, but only when that
value is non-zero. At rest, power output is 0 in **all 82 files**, so the
division is always skipped: `Load` was entering the network as raw watts, mean
66 and standard deviation 76, while every other feature is a ratio with a mean
between 0.96 and 4.55.

`resp-experiment-load` measures three forms of the same column with identical
folds, epochs and model. `Load / 100` is the key experiment: it has the
**identical shape** to raw watts, so it carries exactly the same information, and
only the magnitude changes.

Added to the seven physiological features:

| variant | mean F1 | delta |
|---|---|---|
| without `Load` | 0.7844 | — |
| raw `Load` (watts) | 0.7677 | −0.0167 |
| `Load` / subject's peak | 0.7924 | +0.0080 |
| `Load` / 100 (constant) | 0.7847 | +0.0003 |

Alone, as the only feature:

| variant | mean F1 | per fold |
|---|---|---|
| raw `Load` (watts) | 0.6844 | 0.716 0.701 0.699 0.690 0.617 |
| `Load` / subject's peak | 0.5238 | 0.353 0.589 0.580 0.512 0.585 |
| `Load` / 100 (constant) | 0.4783 | 0.390 0.451 0.627 0.389 0.534 |

Two conclusions, the second more important than the first.

**On power output.** In combination, raw watts hurt the model and the same
column rescaled does not: the damage was the scale. So the hypothesis that
`Load` hurt because it describes the protocol rather than the subject's response
**is not supported**. On the contrary, on its own absolute power in watts scores
0.6844, better than any other form: the thresholds fall at absolute workloads
that are reasonably reproducible across subjects, and dividing by each
subject's peak (`Load_peak`) erases that information.

**On the network.** Raw `Load` and `Load / 100` carry the same information, and
on their own they score 0.6844 against 0.4783: **a 0.206 gap for a plain
division by 100**, with folds that do not overlap. The model is not responding
to what the features contain but to how large they are. The pipeline
standardises nothing: each column reaches the network with whatever scale it
happens to get from the ratio to rest, and those scales differ widely (`VO2` has
a standard deviation of 2.44, `VE/VCO2` of 0.17).

This is a confounder across **the whole** group table above: part of those
comparisons measures the columns' luck of scale rather than their informative
content. The fix is to standardise all features on dev-set statistics and redo
the comparison. Until that is done, the group ranking should be read as
indicative.

### Which hyperparameters matter

40 Optuna trials (10 pruned) on the seven selected features, 5-fold CV. Best
configuration found: `hidden 151, 1 layer, adam lr 2.09e-3, weight decay
4.31e-6, batch 32`.

Compared with the hand-picked baseline using identical folds and epochs:

| configuration | mean F1 | per fold |
|---|---|---|
| baseline (hidden 128, 2 layers, dropout 0.2, lr 1e-3, batch 16) | **0.7844** | 0.832 0.767 0.824 0.753 0.746 |
| Optuna's best | 0.7792 | 0.807 0.742 0.825 0.759 0.763 |

**The search found nothing better.** The gap is 0.005 and the two
configurations trade folds with each other: it is noise. The problem is
sensitive to the features — 0.557 to 0.784 between the worst and the best group
— and insensitive to the hyperparameters.

### The final model

Baseline configuration (the validation winner), eight features from the selected
group, early stopping at epoch 15 with the best weights from epoch 10. On the
**17 subjects never seen by any selection step**:

```
macro F1 0.7260    precision 0.7305    recall 0.7224
```

| class | F1 | precision | recall | samples |
|---|---|---|---|---|
| below AT | 0.825 | 0.828 | 0.822 | 2764 |
| **between AT and RC** | **0.440** | 0.464 | 0.420 | 1597 |
| above RC | 0.912 | 0.900 | 0.926 | 5964 |

Confusion matrix (rows = true, columns = predicted):

```
                predicted
              0     1     2
true  0    2272   489     3
      1     314   670   613
      2     157   286  5521
```

**The number that matters is the 0.440.** The model does a poor job on the zone
between AT and RC, which is exactly the zone the project exists for. Of the 1597
true breaths in that band it gets 670 right: 613 end up above RC and 314 below
AT. The band is squeezed from both sides, so in practice the model **places AT
late and RC early**. The macro F1 of 0.73 is held up by the two easy classes,
which are long homogeneous stretches where following the overall trend is enough.

For comparison, the same model trained on the seven features without power
output (the group that won before `Load` was normalised) gives a macro F1 of
0.7295 but **0.398 on the middle band**: slightly better on average, worse on
the class that matters. The two macro F1 values differ by 0.0035 on a single
measurement over 17 subjects, so that difference means nothing; the 0.042 gap on
the middle class is more substantial but is still a single measurement.

It should be stated that the test set has been read **twice**, once for each of
the two feature sets. Neither reading influenced a choice — the feature group
changed because the normalisation of `Load` changed, and that was decided in
cross validation — but two readings are two readings, and a third round should
be run on new data.

Runs are deterministic: separate executions give the same numbers to four
decimal places. The trained model ends up in
`checkpoints/threshold_estimator.pt` (not versioned: it can be regenerated).

## Why the results changed

The numbers in this README are not comparable with those produced earlier,
because the code had five defects that distorted training and the evaluation
protocol held nothing back.

**The bugs.**

- `model.train()` and `model.eval()` were missing: dropout stayed active during
  validation, so the metrics were noisy and early stopping decided on wrong
  numbers.
- The snapshot of the best weights was `model.state_dict()` without `deepcopy`,
  i.e. references to the model's tensors, which later epochs overwrote in place.
  Early stopping reloaded the last weights, not the best ones: it was inert.
- `data.to(device)` did not assign the result, and validation moved nothing at
  all. On CPU you don't notice; on GPU it crashes.
- Adam received only `lr`: `betas` and `weight_decay` were read from the config
  and never passed, so the `weight_decay` found by the tuner was never applied.
  SGD used momentum and nesterov hardcoded in the source instead of the config.
- The `SummaryWriter` was global at module level, and the tuning and feature
  selection scripts imported it: every trial wrote to the same tags of the same
  TensorBoard run.

On top of these there was a trap in the configs: `config.yaml` redefined `model`
and `optimizer` inline after the `defaults`, so `optimizer=sgd` on the command
line still built Adam. This is why tuning results now land in generated config
groups (`conf/model/tuned.yaml`, `conf/optimizer/tuned.yaml`) instead of being
copied by hand.

**The protocol.** There was a single split, and that same 25% of subjects served
as validation for early stopping, as the objective for the Optuna trials, and as
the criterion for choosing the feature group. The F1 values reported were
therefore the maximum, over epochs and configurations, measured on the data used
to make those choices: not estimates of generalisation. Now 20% of the subjects
is kept out of every selection step and read once.

## Limits, i.e. what these numbers do not show

- **No comparison against the clinical method.** The real question is not
  whether per-breath F1 is 0.73, it is whether this model lands closer to the
  thresholds than the V-slope and the ventilatory equivalents used today.
  Without that comparison there is no way to know whether 0.73 is a good result.
- **The metric is a surrogate.** Nobody cares about classifying individual
  breaths: what matters is by how many seconds or how many watts the estimated
  threshold is off. That error is not measured yet, and it is the metric to add.
- **The test set is a sample of 17 subjects.** The confidence interval around
  0.7260 is wide. The CV mean (0.784) is more stable but is computed on data
  that took part in the choices.
- **The labels come from a single operator** and are treated as ground truth.
  There is no estimate of inter-operator variability, which is not negligible in
  manual threshold reading.
- **No feature standardisation**, and the model is demonstrably sensitive to
  scale (0.206 of F1 for a division by 100, see above). This partly confounds
  the whole feature group table. It is the most important thing to fix.
- **Normalisation is skipped silently** for columns whose resting value is 0 or
  missing (`data/cpet.py`). On this data that happens for `Load` in all 82
  files, with no warning.
- **`data/*.xlsx`** holds 8 files outside `File_CPET/`, three of which
  (`Id_10`, `Id_58`, `Id_70`) are not in the dataset. Why they are excluded is
  not documented.

## Layout

```
pyproject.toml              installable package, dependencies, entry points
Makefile                    shortcuts for the pipeline order
src/respiration/
  paths.py                  project root and the ${project_root:} resolver
  splits.py                 test set + cross validation folds
  training.py               loops, metrics, cross_validate, length bucketing
  data/
    cpet.py                 reading the xlsx, normalisation, labels
    dataset.py              Dataset and collate
  models/
    threshold_estimator.py  the LSTM
  cli/
    prepare_data.py         resp-prepare-data
    select_features.py      resp-select-features
    tune.py                 resp-tune
    train.py                resp-train
    experiment_load.py      resp-experiment-load
  conf/                     Hydra configs (dataset, features, model, optimizer)
tests/                      dataset, splits, batching
results/                    the json files produced by the runs
data/                       xlsx and pickle
outputs/ runs/ checkpoints/  run artefacts (git-ignored)
```

Sequences of similar length end up in the same batch
(`LengthBucketBatchSampler`): with random batches 40% of the computation went
into padding, now 18%.

### To fix

- **standardise the features** on dev-set statistics, and redo the group
  comparison: it is the change that would move the results the most
- **measure the threshold error** in seconds or watts, not just per-breath F1:
  it is the metric that makes this work legible to a physician
- **compare against the clinical method** (V-slope), otherwise there is no
  telling whether 0.73 is a good number
- `data/` holds 8 xlsx outside `File_CPET/`, three of which (`Id_10`, `Id_58`,
  `Id_70`) are not in the dataset: it is not documented why
- `PROJECT_ROOT` in `paths.py` assumes an editable install, which is how the
  project is meant to be used but not the only possibility
