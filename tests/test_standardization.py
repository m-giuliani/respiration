import numpy as np

from respiration.data.dataset import TimeSeriesDataset
from respiration.paths import DATA_DIR
from respiration.training import feature_stats

PATH = DATA_DIR / 'sequence_dataset.pkl'
COLS = ['Rf', 'HR', 'VO2']
TRAIN = list(range(0, 40))
HELD_OUT = list(range(40, 50))


def test_stats_have_one_value_per_feature():
    mean, std = feature_stats(PATH, TRAIN, COLS)
    assert mean.shape == (len(COLS),)
    assert std.shape == (len(COLS),)
    assert (std > 0).all()


def test_training_data_becomes_centred_and_unit_scale():
    stats = feature_stats(PATH, TRAIN, COLS)
    d = TimeSeriesDataset(PATH, TRAIN, columns=COLS, stats=stats)
    valori = np.concatenate([d[i][0].numpy() for i in range(len(d))])
    assert np.allclose(valori.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(valori.std(axis=0), 1, atol=1e-4)


def test_held_out_data_uses_the_training_statistics():
    # I soggetti mai visti non devono essere centrati esattamente su 0: userebbero
    # le proprie statistiche, e sarebbe una fuga di informazione.
    stats = feature_stats(PATH, TRAIN, COLS)
    d = TimeSeriesDataset(PATH, HELD_OUT, columns=COLS, stats=stats)
    valori = np.concatenate([d[i][0].numpy() for i in range(len(d))])
    assert not np.allclose(valori.mean(axis=0), 0, atol=1e-6)
    # ma devono restare su una scala comparabile
    assert np.abs(valori.mean(axis=0)).max() < 2


def test_statistics_depend_only_on_the_subjects_given():
    a = feature_stats(PATH, TRAIN, COLS)
    b = feature_stats(PATH, TRAIN, COLS)
    c = feature_stats(PATH, HELD_OUT, COLS)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], c[0])


def test_without_stats_the_raw_values_are_returned():
    grezzo = TimeSeriesDataset(PATH, TRAIN, columns=COLS)[0][0].numpy()
    stats = feature_stats(PATH, TRAIN, COLS)
    norm = TimeSeriesDataset(PATH, TRAIN, columns=COLS, stats=stats)[0][0].numpy()
    assert not np.allclose(grezzo, norm)
    assert np.allclose((grezzo - stats[0]) / stats[1], norm, atol=1e-5)
