from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch
from PIL import Image
from models.feature_extractor import FeatureExtractor
from submodlib.functions import FacilityLocationFunction
from submodlib.functions import DisparityMinFunction
from submodlib.functions import DisparitySumFunction
from submodlib.functions import LogDeterminantFunction
from tqdm import tqdm
import numpy as np
import random
import os
import pathlib

class CelebaDataset(Dataset):
    def __init__(self, subset = False, sub_selection_technique = None,  percentage = 0.1, 
                 transform = None, bs = 64):
        self.data = list(pathlib.Path('data/celeba/images/').glob('*.png'))
        if not subset:
            self.idxs = list(range(len(self.data)))
        else:
            # For storing idxs in a file and loading for fast retrieval
            if os.path.exists(f'celeba_{sub_selection_technique}.txt'):
                self.idxs = np.loadtxt(f'data/celeba_{sub_selection_technique}.txt').tolist()
            else:
                self.idxs = self.subset_selection(sub_selection_technique, percentage, bs)
                np.savetxt(f'data/celeba_{sub_selection_technique}.txt', np.array(self.idxs), fmt='%d')
        self.transform = transform

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        img_path = self.data[self.idxs[idx]]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img