import numpy as np

from src.scripts.load_data import load_and_convert
import torch
from torch.nn.utils.rnn import pad_sequence
import os
from pathlib import Path
from torch.utils.data import Dataset
from pickle import load
import hydra


class TimeSeriesDataset(Dataset):

    def __init__(self, data_path, index=None, columns=None):
        if isinstance(data_path, Path) or isinstance(data_path, str):
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Unable to find the data path {data_path}")

        # load the dataset
        self.df_map = None
        if str(data_path).split('.')[1] == 'pkl':
            # raise ValueError(f"Maybe your data in not in the correct pickle format? {data_path}")
            with open(data_path, 'rb') as f:
                self.df_map = load(f)
        else:
            self.df_map = load_and_convert(data_path)
        # create a numeric index for the loaded files
        self.df_idx = sorted(self.df_map.keys())
        # select a subset of the input data eventually
        if index:
            self.df_idx = [self.df_idx[i] for i in index]
            self.df_map = {k: v for (k, v) in self.df_map.items() if k in self.df_idx}

        if columns:
            self.features = columns
        else:
            self.features = self.df_map[self.df_idx[0]].columns[1:-1]

    def __len__(self):
        return len(self.df_map)

    def __getitem__(self, item):
        #print("the item: ", item)
        #print("the file name: ", self.file_name(item))
        df = self.df_map[self.file_name(item)]
        features = df.loc[:, self.features].astype(np.float32).to_numpy()
        labels = df.loc[:, df.columns[-1:]].astype(np.float32).to_numpy()
        return torch.from_numpy(features), torch.from_numpy(labels)

    def file_name(self, id):
        return self.df_idx[id]


def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_sequences = pad_sequence(sequences, batch_first=True)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1).to(int)
    return padded_sequences, lengths, padded_labels