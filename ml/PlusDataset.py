import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class PlusDataset(Dataset):
    def __init__(self, csv_file):
        self.X_lines = pd.read_csv(csv_file, usecols=[0, 1])
        self.y_lines = pd.read_csv(csv_file, usecols=[2])

    def __len__(self):
        return len(self.X_lines)

    def __getitem__(self, idx):
        X_lines = self.X_lines.iloc[idx].to_numpy(dtype=np.float32)
        y_lines = self.y_lines.iloc[idx].to_numpy(dtype=np.float32)
        return X_lines, y_lines
