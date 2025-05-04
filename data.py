import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
import numpy as np
import torchaudio
import os

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, table_path, data_path, transforms=None):
        self.table = pd.read_csv(table_path)
        self.data_path = data_path
        self.transforms = transforms
    def __getitem__(self, idx):
        file_path = self.table.loc[idx, 'data_table.csv'] # TODO исправить data_table
        if isinstance(file_path, str):
            path = os.path.join(self.data_path, file_path)
            audio, _ = torchaudio.load(path, channels_first=True)
            if self.transforms:
                return self.transforms(audio[0:].squeeze())
            return audio[0:].squeeze()
        else:
            return None
    def __length__(self):
        return len(self.table) # TODO исправить data_table

    def get_dataloader(config, dataset_type, transforms):
        dataset = AudioDataset(
            config['dataset'][dataset_type]['table'],
            config['dataset'][dataset_type]['data'],
            transforms
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            config[dataset_type]['batch_size'],
            config[dataset_type]['shuffle'],
            pin_memory=config[dataset_type]['pin_memory'],
        )
        return dataloader
