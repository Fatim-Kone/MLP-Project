import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

class PendulumDataset(Dataset):
    def __init__(self, file_paths, seq_len=5, pred_len=0):
        self.seq_len = seq_len
        self.trajectories = []
        
        for path in file_paths:
            traj = np.load(path)  # shape: (steps, 4) -> [theta1, theta2, omega1, omega2]
            self.trajectories.append(traj)
        
        all_data = np.concatenate(self.trajectories, axis=0)  # shape: (total_steps, 4)
        self.scaler = MinMaxScaler()
        self.scaler.fit(all_data)
        self.trajectories = [self.scaler.transform(traj) for traj in self.trajectories]
        
        self.create_seq():
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def create_seq(self):
        for traj in self.trajectories:
            for i in range(len(traj) - seq_len):
                self.x.append(traj[i:i+seq_len])
                self.y.append(traj[i+seq_len:i+seq_len+pred_len])
                
        self.x = torch.tensor(self.x, dtype=torch.float32) 
        self.y = torch.tensor(self.y, dtype=torch.float32) 
