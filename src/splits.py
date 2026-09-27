"""Suddivisione del dataset in test set, fold di cross validation e dev set.

Ogni file Id_XX.xlsx e' il test di un soggetto diverso, quindi una sequenza
corrisponde a un soggetto: non serve raggruppare, basta partizionare gli indici.
"""

from sklearn.model_selection import KFold, train_test_split


def make_splits(n_samples, n_folds=5, test_size=0.2, seed=42):
    """Separa un test set e divide il resto in fold di cross validation.

    Restituisce (folds, dev_index, test_index), dove folds e' una lista di
    coppie (train_index, val_index). Il test set non deve mai essere usato per
    scegliere iperparametri, feature o epoca migliore: serve solo per la misura
    finale, una volta sola.
    """
    dev_index, test_index = train_test_split(
        range(n_samples), test_size=test_size, random_state=seed
    )
    dev_index = sorted(dev_index)
    test_index = sorted(test_index)

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [
        ([dev_index[i] for i in train_pos], [dev_index[i] for i in val_pos])
        for train_pos, val_pos in kfold.split(dev_index)
    ]
    return folds, dev_index, test_index
