import math

from respirazione.splits import make_splits

N = 82


def test_test_set_is_disjoint_from_dev():
    folds, dev, test = make_splits(N)
    assert not set(dev) & set(test)
    assert set(dev) | set(test) == set(range(N))


def test_folds_never_touch_the_test_set():
    folds, dev, test = make_splits(N)
    for train_index, val_index in folds:
        assert not set(train_index) & set(test)
        assert not set(val_index) & set(test)


def test_train_and_validation_are_disjoint_inside_each_fold():
    folds, dev, test = make_splits(N)
    for train_index, val_index in folds:
        assert not set(train_index) & set(val_index)
        assert set(train_index) | set(val_index) == set(dev)


def test_folds_cover_the_dev_set_exactly_once():
    folds, dev, test = make_splits(N)
    validazioni = [i for _, val_index in folds for i in val_index]
    assert sorted(validazioni) == sorted(dev)


def test_split_is_reproducible():
    assert make_splits(N) == make_splits(N)


def test_test_size_is_respected():
    folds, dev, test = make_splits(N, test_size=0.2)
    # train_test_split arrotonda per eccesso: 82 * 0.2 = 16.4 -> 17 soggetti
    assert len(test) == math.ceil(N * 0.2)
    assert len(folds) == 5
