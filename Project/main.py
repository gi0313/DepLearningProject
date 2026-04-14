import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# 1. Inject your Kaggle API token directly into the environment
os.environ['KAGGLE_API_TOKEN'] = "KGAT_f04e9f69e859b8117d064171adfe8a10"

import kaggle

def setup_dataset():
    dataset_name = "abhikalpsrivastava15/space-images-category"
    target_dir = "./data"

    if not os.path.exists(target_dir) or not os.listdir(target_dir):
        print("Downloading Astronomy dataset from Kaggle...")
        os.makedirs(target_dir, exist_ok=True)

        # This will now use the KGAT token you provided above!
        kaggle.api.dataset_download_files(dataset_name, path=target_dir, unzip=True)

        print("Dataset successfully downloaded and extracted into /data!")
    else:
        print("Dataset already exists locally. Skipping download.")

if __name__ == "__main__":
    setup_dataset()
