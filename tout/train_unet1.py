# train_unet.py
import os, glob, random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.unet import UNet
import cv2
import numpy as np
from tqdm import tqdm

class ECGDataset(Dataset):
    def __init__(self, imgs, masks, size=(512,512)):
        self.imgs = imgs
        self.masks = masks
        self.size = size
        self.aug = A.Compose([
            A.Resize(size[0], size[1]),
            A.RandomRotate90(),
            A.HorizontalFlip(p=0.5),
            A.OneOf([A.RandomContrast(), A.RandomBrightness()], p=0.5),
        ])
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise FileNotFoundError(self.imgs[idx])
        augmented = self.aug(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        img = np.expand_dims(img/255.0,0).astype('float32')
        mask = (mask>127).astype('float32')[None,...]
        return torch.tensor(img), torch.tensor(mask)

def train(data_dir='data/train', epochs=40, batch_size=8, lr=1e-4, out='checkpoints/unet.pth'):
    imgs = sorted(glob.glob(os.path.join(data_dir,'images','*.png')) + glob.glob(os.path.join(data_dir,'images','*.jpg')))
    masks = sorted(glob.glob(os.path.join(data_dir,'masks','*.png')) + glob.glob(os.path.join(data_dir,'masks','*.jpg')))
    assert len(imgs)==len(masks), "images/masks count mismatch"
    dataset = ECGDataset(imgs, masks)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_ch=1, out_ch=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for imgs_t, masks_t in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs_t = imgs_t.to(device)
            masks_t = masks_t.to(device)
            preds = model(imgs_t)
            loss = criterion(preds, masks_t)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {epoch_loss/len(loader):.4f}")
        if (epoch+1)%5==0:
            torch.save(model.state_dict(), out.replace('.pth', f'_ep{epoch+1}.pth'))
    torch.save(model.state_dict(), out)
    print("Saved", out)

if __name__=='__main__':
    train()
