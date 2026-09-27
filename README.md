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

### Feature scale, and why the first ranking was wrong

`data/cpet.py` normalises every column by its resting value, but only when that
value is non-zero. At rest power output is 0 in all 82 files, so `Load` was
entering the network as raw watts — mean 66, standard deviation 76 — while every
other feature is a ratio to rest with a mean between 0.96 and 4.55.

Chasing that down turned out to matter far more than the one column. An LSTM
applies a single weight matrix to all inputs, so a feature with a wider spread
contributes proportionally more to the gates at equal weight. Among the
physiological features the spreads differ by **18x**:

```
VE/VCO2    std 0.17   #
Load_peak  std 0.35   ##
VE/VO2     std 0.39   ##
HR         std 0.53   ###
Rf         std 0.73   ####
VO2/HR     std 1.05   ######
VO2        std 2.44   ##############
VCO2       std 3.16   ##################
```

`resp-experiment-load` measures three forms of the same column, identical folds
and model. `Load / 100` is the decisive case: same shape as raw watts, therefore
the same information, only the magnitude changes.

| variant | with 7 physiological features | alone |
|---|---|---|
| without `Load` | 0.7844 | — |
| raw `Load` (watts) | 0.7677 | 0.6844 |
| `Load` / subject's peak | 0.7924 | 0.5238 |
| `Load` / 100 (constant) | 0.7847 | 0.4783 |

Raw `Load` and `Load / 100` carry the same information and, on their own, score
0.6844 against 0.4783: **a 0.206 gap for a plain division by 100**, with folds
that do not overlap. The network responds to magnitude, not to content.

So the features are now standardised: mean 0 and standard deviation 1, with the
statistics computed on **the training subjects of each fold only** and applied
unchanged to validation and test. Computing them over the whole dataset would
push test-set information into the inputs. `standardize=false` reproduces the
old behaviour.

### Which features matter

15 combinations of the four groups, 5-fold CV, the same baseline model for all
of them so the comparison measures the features and not the hyperparameters.

| group | mean F1 | n. features | vs unstandardised |
|---|---|---|---|
| **respiratory + load** | **0.8127** | 4 | +0.2113 |
| respiratory + cardiac + load | 0.7959 | 6 | +0.0846 |
| respiratory + metabolic + load | 0.7927 | 6 | +0.0029 |
| respiratory + metabolic | 0.7882 | 5 | +0.0021 |
| respiratory + cardiac + metabolic + load | 0.7865 | 8 | -0.0059 |
| respiratory + cardiac + metabolic | 0.7683 | 7 | -0.0161 |
| cardiac + load | 0.7622 | 3 | +0.0758 |
| respiratory + cardiac | 0.7586 | 5 | +0.0522 |
| respiratory | 0.7449 | 3 | +0.1485 |
| cardiac + metabolic + load | 0.7394 | 5 | -0.0180 |
| metabolic + load | 0.7270 | 3 | -0.0126 |
| load | 0.7117 | 1 | +0.1879 |
| cardiac + metabolic | 0.6651 | 4 | -0.0823 |
| cardiac | 0.5929 | 2 | +0.0429 |
| metabolic | 0.5801 | 2 | -0.1670 |

The last column is the point. Standardising did not lift everything: **7 groups
rose, 3 fell, and every group that fell contains `metabolic`**. `VO2` and `VCO2`
are the two widest columns in the dataset, and on their own they drop from
0.7471 to 0.5801 — they were not as informative as they looked, they were loud.
Meanwhile `respiratory` (the three narrowest columns, `VE/VCO2` at 0.17) gains
0.149 and `respiratory + load` gains 0.211, moving from **12th place to 1st**.

This retracts an earlier claim in this file, that almost all the signal sits in
gas exchange. It does not: that was `VO2` and `VCO2` drowning out the
ventilatory equivalents, which is also the physiologically odd part of the old
result, since `VE/VCO2` is one of the classic markers for RC.

The winner is `respiratory + load` — `Rf, VE/VO2, VE/VCO2, Load_peak` — with
four features.

**On how much of this ranking can be trusted.** With 5 folds, a paired t-test
across folds cannot separate the middle of the table. `load` alone sits 0.101
below the winner and still comes out "not distinguishable" (t = 1.77), while
`respiratory` sits a *smaller* 0.068 below and does come out worse (t = 9.38):
the test reacts to how *consistent* a difference is across folds, not to how
large it is. So neither "the winner wins" nor "these N groups are equivalent" is
a supportable reading. What the design does support:

- the four clear failures (`respiratory`, `cardiac`, `metabolic`,
  `cardiac + metabolic` on their own) are genuinely behind
- the standardisation effect is large and systematic, not noise: its sign tracks
  column spread across all 15 groups
- `cardiac` adds nothing on top of `respiratory + load` (0.7959 vs 0.8127, paired
  t = 1.09), so the 4-feature group is preferable on parsimony, not accuracy

Ranking the middle of the table needs more folds — 10-fold, or repeated 5-fold —
which is cheap on this data and is the obvious next step.

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
sensitive to the features and insensitive to the hyperparameters.

Note that this search ran **before standardisation**, on unstandardised inputs
and on the feature group that was winning then. Standardising changes the input
distribution, which is exactly the kind of change that can move the optimal
learning rate and hidden size, so the search is worth repeating now — this is
the one place where redoing it is justified. The stored
`conf/model/tuned.yaml` and `conf/optimizer/tuned.yaml` therefore describe a
pipeline that no longer exists; the final model above uses the baseline instead.

### The final model

Baseline configuration, the four features of the selected group, standardised
inputs, early stopping at epoch 31 with the best weights from epoch 26. On the
**17 subjects never seen by any selection step**:

```
macro F1 0.8262    precision 0.8462    recall 0.8128
```

| class | F1 | precision | recall | samples |
|---|---|---|---|---|
| below AT | 0.906 | 0.909 | 0.904 | 2764 |
| **between AT and RC** | **0.619** | 0.703 | 0.552 | 1597 |
| above RC | 0.954 | 0.927 | 0.982 | 5964 |

Confusion matrix (rows = true, columns = predicted):

```
                predicted
              0     1     2
true  0    2499   265     0
      1     251   882   464
      2       0   107  5857
```

The band between AT and RC is still the weak class at 0.619, and it is still
squeezed from both sides — 464 of its breaths land above RC and 251 below AT, so
the model places AT late and RC early. But it is a different picture from before
standardisation: that band scored 0.440, and the two easy classes were carrying
the average. Two zone-skipping errors have also disappeared: the model now never
confuses "below AT" with "above RC" in either direction (0 and 0 cases).

For reference, the progression of the same measurement across the three pipeline
states:

| pipeline | features | macro F1 | between AT and RC |
|---|---|---|---|
| 7 features, no standardisation | 7 | 0.7295 | 0.398 |
| 8 features, no standardisation | 8 | 0.7260 | 0.440 |
| **4 features, standardised** | **4** | **0.8262** | **0.619** |

Almost all of that gain belongs to standardisation plus the honest feature
re-selection it made possible. Standardising the *previous* feature group changed
nothing on its own (0.7924 to 0.7865 in CV, inside the noise): the gain comes
from re-ranking the groups on a basis where the comparison means what it claims,
and then picking a different, smaller group.

It must be stated that the test set has now been read **three times**, once per
pipeline state. None of the readings fed back into a choice — the feature group
changed because the normalisation changed, and that was decided in cross
validation — but three readings are three readings, and a fourth round belongs on
new data.

Runs are deterministic: separate executions give the same numbers to four
decimal places. The trained model lands in
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

**The scale.** Nothing in the pipeline standardised the features, and the
network turned out to respond to input magnitude rather than content — 0.206 of
F1 for a division by 100. The feature group ranking was therefore partly
measuring which columns happened to be widest. Fixing it moved the winning group
from eight features to four and the test-set macro F1 from 0.7260 to 0.8262.

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
  0.8262 is wide, and the set has now been read three times, once per pipeline
  state. The CV mean (0.8127) is more stable but is computed on data that took
  part in the choices.
- **5 folds cannot rank the middle of the feature table.** A paired t-test across
  5 folds calls a 0.101 gap "not distinguishable" and a 0.068 gap "worse",
  because it reacts to the consistency of a difference and not its size. The
  clear failures are separable; the top two thirds of the table are not. More
  folds, or repeated CV, would fix this cheaply.
- **The labels come from a single operator** and are treated as ground truth.
  There is no estimate of inter-operator variability, which is not negligible in
  manual threshold reading.
- **Normalisation is skipped silently** for columns whose resting value is 0 or
  missing (`data/cpet.py`). On this data that happens for `Load` in all 82
  files. It now logs at debug level, but the pipeline still proceeds.
- **The hyperparameters were tuned before standardisation** and have not been
  re-searched since, so the stored tuned configs describe a pipeline that no
  longer exists.
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

- **more folds** (10-fold, or repeated 5-fold) so the middle of the feature table
  can actually be ranked: cheap on this data, and the current design cannot do it
- **redo the hyperparameter search** on standardised inputs: the stored tuned
  configs predate standardisation, and changing the input distribution is exactly
  what can move the optimal learning rate
- **measure the threshold error** in seconds or watts, not just per-breath F1:
  it is the metric that makes this work legible to a physician
- **compare against the clinical method** (V-slope), otherwise there is no
  telling whether 0.73 is a good number
- `data/` holds 8 xlsx outside `File_CPET/`, three of which (`Id_10`, `Id_58`,
  `Id_70`) are not in the dataset: it is not documented why
- `PROJECT_ROOT` in `paths.py` assumes an editable install, which is how the
  project is meant to be used but not the only possibility
