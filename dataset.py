import torch
import numpy as np
import pandas as pd
import enum


class TimeSeriesDataset(torch.utils.data.Dataset):
    
    # Tasks enumeration
    class Task(enum.Enum):
        REGRESSION = 'regression'
        CLASSIFICATION = 'classification'
    
    @staticmethod
    #def build_windows(X:np.array, y:np.array, seq_len_x:int=10, seq_len_y:int=1) -> tuple:
    def build_windows(X:torch.tensor, y:torch.tensor, seq_len_x:int=10, seq_len_y:int=1) -> tuple:
        seq_len = max(seq_len_x, seq_len_y)
        tot_len = X.shape[0] - seq_len

        X_windows = X.unfold(0, seq_len_x, 1).permute(0,2,1)[0:tot_len]
        y_windows = y.unfold(0, seq_len_y, 1).permute(0,2,1)[0:tot_len]
        
        return X_windows, y_windows
        
    def __init__(
        self,
        X : torch.tensor,
        y : torch.tensor,
        seq_len_x : int = 10,
        seq_len_y : int = 1,
        offset : int = 0,
        channels: bool = False,
        task : Task = Task.REGRESSION
    ) -> None:
        # Store the parameters
        self.offset  = offset
        self.seq_len_x = seq_len_x
        self.seq_len_y = seq_len_y
        self.channels = channels
        self.task = task
        # Convert original time series into a windowed time series
        self.X, self.y = self.build_windows(X, y, seq_len_x, seq_len_y)
        # Compute the length of the dataset
        self.length  = self.X.shape[0] - self.offset
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx) -> tuple:
        x = self.X[idx]
        y = self.y[idx+self.offset]
        if self.channels:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        if self.task == self.Task.CLASSIFICATION:
            y = y.squeeze()
    
        return x, y            


class TimeSeriesLoader(torch.utils.data.DataLoader):
    
    def __init__(
        self,
        dataset: TimeSeriesDataset,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False
    ) -> None:
        super(TimeSeriesLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    def __iter__(self):
        for batch in super(TimeSeriesLoader, self).__iter__():
            yield batch
            