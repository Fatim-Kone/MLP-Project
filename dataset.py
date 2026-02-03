import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

class PendulumDataset(Dataset):
    def __init__(self, trajectories, seq_len=5, pred_len=0, mean=None, std=None):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.x =[]
        self.y = []
        
        if mean is None or std is None:
            all_data = np.concatenate(trajectories, axis=0)
            self.mean = all_data.mean(axis=0)
            self.std  = all_data.std(axis=0) + 1e-8
        else:
            self.mean = mean
            self.std  = std
        
        for traj in trajectories:
            traj = (traj - self.mean) / self.std
            for i in range(len(traj) - seq_len):
                self.x.append(traj[i:i+seq_len])
                self.y.append(traj[i+seq_len])

        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
