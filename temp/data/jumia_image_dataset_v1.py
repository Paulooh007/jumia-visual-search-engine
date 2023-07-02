import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import load_config

CONFIG = load_config()


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


data_transforms = {
    "train": transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.PILToTensor(),
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.GaussianBlur(7),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # transforms.ToTensor()
        ]
    ),
    "valid": transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.PILToTensor(),
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # transforms.ToTensor()
        ]
    ),
}
