import torch
from torch.utils.data import DataLoader
from dataset_ecg import ECGSegDataset
from models.unet import UNet
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ECGSegDataset(train_imgs, train_masks)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = UNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(20):
    model.train()
    loss_sum = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

    print(f"Epoch {epoch+1} | Loss: {loss_sum/len(loader):.4f}")

torch.save(model.state_dict(), "unet_ecg.pth")
