import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import pandas as pd

LOGGER = logging.getLogger('CNN_Denoiser')
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler())
LOGGER.addHandler(logging.FileHandler('cnn.log'))

# ------------------------ Dataset ------------------------
class DenoisingDataset(Dataset):
    def __init__(self, dirty_dir, clean_dir=None, transform=None):
        self.dirty_dir = dirty_dir
        self.clean_dir = clean_dir
        self.image_names = sorted(os.listdir(dirty_dir))
        self.transform = transform or T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((420, 540)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        dirty_img_path = os.path.join(self.dirty_dir, name)
        dirty_img = Image.open(dirty_img_path).convert('RGB')
        original_size = dirty_img.size[::-1]  # PIL gives (W, H), we want (H, W)
        dirty_tensor = self.transform(dirty_img)

        if self.clean_dir:
            clean_img_path = os.path.join(self.clean_dir, name)
            clean_img = Image.open(clean_img_path).convert('RGB')
            clean_tensor = self.transform(clean_img)
            return dirty_tensor, clean_tensor
        else:
            return dirty_tensor, name, original_size


# ------------------------ Model ------------------------
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# ------------------------ Training ------------------------
def train_model(model, dataloader, device, epochs=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for dirty, clean in dataloader:
            dirty, clean = dirty.to(device), clean.to(device)
            optimizer.zero_grad()
            outputs = model(dirty)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        LOGGER.info(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(dataloader):.4f}")


# ------------------------ Resize to Original ------------------------
def resize_to_original(pred_tensor, original_size):
    pred_tensor = pred_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    resized = F.interpolate(pred_tensor, size=original_size, mode='bilinear', align_corners=False)
    return resized.squeeze()


# ------------------------ Prediction & CSV ------------------------
def predict_and_generate_submission(model, dataloader, device, output_csv):
    model.to(device)
    model.eval()
    rows = []

    with torch.no_grad():
        for dirty, names, sizes in dataloader:
            dirty = dirty.to(device)
            outputs = model(dirty).cpu()

            for i in range(len(names)):
                pred = outputs[i].squeeze()
                resized_pred = resize_to_original(pred, sizes[i]).numpy()

                h, w = resized_pred.shape
                for r in range(h):
                    for c in range(w):
                        pixel_id = f"{i+1}_{r+1}_{c+1}"
                        value = float(resized_pred[r, c])
                        rows.append((pixel_id, value))

    df = pd.DataFrame(rows, columns=["id", "value"])
    df.to_csv(output_csv, index=False)
    print(f"Submission saved to {output_csv}")


# ------------------------ Main Entry Point ------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict'], required=True)
    parser.add_argument('--dirty_dir', required=True)
    parser.add_argument('--clean_dir')
    parser.add_argument('--model_path', default='denoiser.pth')
    parser.add_argument('--submission_csv', default='submission.csv')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvAutoencoder()

    if args.mode == 'train':
        dataset = DenoisingDataset(args.dirty_dir, args.clean_dir)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        train_model(model, dataloader, device, args.epochs)
        torch.save(model.state_dict(), args.model_path)

    elif args.mode == 'predict':
        dataset = DenoisingDataset(args.dirty_dir)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        predict_and_generate_submission(model, dataloader, device, args.submission_csv)