"""
GTSRB Dataset Loader

Simple PyTorch Dataset class for loading the German Traffic Sign Recognition Benchmark.
Reads image paths and labels from CSV files.
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class GTSRBDataset(Dataset):
    """
    Custom Dataset for GTSRB traffic signs.
    
    Expects:
    - CSV file with columns: 'Path' (relative image path) and 'ClassId' (0-42)
    - Image directory containing the actual image files
    - Optional transform for data augmentation/normalization
    """
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rel_path = self.data.iloc[idx]['Path']
        if rel_path.startswith('Train/'):
            rel_path = rel_path[len('Train/'):]
        elif rel_path.startswith('Test/'):
            rel_path = rel_path[len('Test/'):]
        img_path = os.path.join(self.img_dir, rel_path)
        image = Image.open(img_path).convert('RGB')
        label = int(self.data.iloc[idx]['ClassId'])

        if self.transform:
            image = self.transform(image)
        return image, label
