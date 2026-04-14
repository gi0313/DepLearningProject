import torch
from torch.utils.data import Dataset
import numpy as np

class OptDigitsDataset(Dataset):
    def __init__(self, file_path):
        #Read file into 2D Numpy array
        #np.loadtxt automatically splits the spaces and converts everything to floats
        raw_data = np.loadtxt(file_path, dtype=np.float32)

        #Extract the features
        #Slice all rows and take the first 1024 columns
        self.features = torch.tensor(raw_data[:, :1024], dtype=torch.float32)

        #Extract the labels, slice all rows and take exactly the last column
        #Cast this to torch.long because they are integer class labels
        self.labels = torch.tensor(raw_data[:, 1024], dtype=torch.long)

    def __len__(self):
        #Returns total number of samples
        return len(self.labels)

    def __getitem__(self, idx):
        #Returns a single sample features and label
        return self.features[idx], self.labels[idx]