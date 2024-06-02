from config.definitions import ROOT_DIR
from src.timeseriesdataset import TimeSeriesDataset
from pathlib import Path
import pytest
import pandas as pd


class Test_TimeSeriensDataset:

    @pytest.fixture()
    def init_dataset(self):
        dataset = TimeSeriesDataset(Path(ROOT_DIR) / 'data/sequence_dataset.pkl')
        return dataset

    @pytest.fixture()
    def original_data(self):
        data_path = Path(ROOT_DIR) / 'data/File_CPET'
        import os
        file_list = [f for f in os.listdir(data_path) if f.split('.')[1] == 'xlsx']
        raw_data = dict()
        for f in file_list:
            xls = pd.ExcelFile(data_path / f)
            raw_data[f] = pd.read_excel(xls, 'Test')
        return raw_data

    def test__len__(self, init_dataset):
        dataset = init_dataset
        assert len(dataset) == 82

    def test__getitem__(self, init_dataset, original_data):
        dataset = init_dataset
        original_df_dict = original_data
        for i in range(len(dataset)):
            file_name = dataset.file_name(i)
            sample, target = dataset[i]
            # original data has two rows with invalid numbers...
            assert len(sample) == len(original_df_dict[file_name]) - 2

    def test_features(self, init_dataset):
        dataset = init_dataset
        for f in dataset.features:
            assert isinstance(f,str)