from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os

class ImagePairsDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img1_path, img2_path, label = row["Image1"], row["Image2"], row["Label"]

        img1 = self.load_image(img1_path)
        img2 = self.load_image(img2_path)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

    def load_image(self, path):
        if os.path.exists(path):
            image = Image.open(path).convert("RGB")
            return image
        else:
            print(f"Image not found: {path}")
            return None

def create_dataframe_from_csv(csv_path):
    return pd.read_csv(csv_path)