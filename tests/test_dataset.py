import os
from pathlib import Path

import pandas as pd
import pytest

from respirazione.data.dataset import TimeSeriesDataset
from respirazione.paths import DATA_DIR

CPET_DIR = DATA_DIR / 'File_CPET'


def xlsx_files():
    return sorted(f for f in os.listdir(CPET_DIR) if Path(f).suffix == '.xlsx')


class TestTimeSeriesDataset:

    @pytest.fixture(scope="class")
    def dataset(self):
        return TimeSeriesDataset(DATA_DIR / 'sequence_dataset.pkl')

    @pytest.fixture(scope="class")
    def original_data(self):
        raw_data = dict()
        for f in xlsx_files():
            raw_data[f] = pd.read_excel(pd.ExcelFile(CPET_DIR / f), 'Test')
        return raw_data

    def test_len(self, dataset):
        # Dedotto dai file, non scritto a mano: aggiungere un soggetto non deve
        # far fallire il test.
        assert len(dataset) == len(xlsx_files())

    def test_getitem(self, dataset, original_data):
        for i in range(len(dataset)):
            file_name = dataset.file_name(i)
            sample, target = dataset[i]
            # le prime due righe del foglio Test contengono le unita' di misura
            assert len(sample) == len(original_data[file_name]) - 2
            assert len(target) == len(sample)

    def test_features_are_names(self, dataset):
        assert dataset.features is not None
        for f in dataset.features:
            assert isinstance(f, str)

    def test_labels_are_the_three_classes(self, dataset):
        visti = set()
        for i in range(len(dataset)):
            visti |= set(dataset[i][1].flatten().tolist())
        assert visti <= {0.0, 1.0, 2.0}

    def test_sequence_lengths_match_getitem(self, dataset):
        lengths = dataset.sequence_lengths()
        assert len(lengths) == len(dataset)
        assert lengths[0] == len(dataset[0][0])

    def test_column_subset_is_respected(self):
        colonne = ['VO2', 'VCO2']
        d = TimeSeriesDataset(DATA_DIR / 'sequence_dataset.pkl', columns=colonne)
        assert d[0][0].shape[1] == len(colonne)

    def test_index_subset_selects_those_subjects(self):
        d = TimeSeriesDataset(DATA_DIR / 'sequence_dataset.pkl', index=[0, 1, 2])
        assert len(d) == 3

    def test_missing_path_raises(self):
        with pytest.raises(FileNotFoundError):
            TimeSeriesDataset(DATA_DIR / 'non_esiste.pkl')
