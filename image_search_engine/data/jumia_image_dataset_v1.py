import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class JumiaImageDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.file_names = df["filepath"].values
        self.labels = df["class"].values
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = Image.open(img_path).convert("RGB")
        label = self.labels[index]

        if self.transforms:
            img = self.transforms(img)

        return {"image": img, "label": torch.tensor(label, dtype=torch.long)}
