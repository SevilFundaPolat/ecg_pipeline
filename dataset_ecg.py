import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class ECGSegDataset(Dataset):
    def __init__(self, img_paths, mask_paths):
        self.imgs = img_paths
        self.masks = mask_paths

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(self.imgs[idx])
        if mask is None:
            raise FileNotFoundError(self.masks[idx])

        img = cv2.resize(img, (512, 512))
        mask = cv2.resize(mask, (512, 512))

        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask
