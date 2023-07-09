from image_search_engine.metadata import jumia_3650
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import joblib
from pathlib import Path
from torchvision import transforms

import joblib

PACKAGE_DIR = Path(__file__).parent.parent

# Load the pickled file
with open(PACKAGE_DIR / "artifacts/class_encoder_jumia_3650.pkl", "rb") as file:
    encoder = joblib.load(file)


class Jumia3650Dataset(Dataset):
    def __init__(self, data_filename, data_transforms=None, img_size=224):
        self.df = pd.read_csv(data_filename)
        self.file_paths = self.df["filepath"].values
        self.labels = encoder.transform(self.df["class"])
        self.classes = encoder.classes_
        self.class_to_idx = {l: i for i, l in enumerate(encoder.classes_)}
        if transforms is None:
            self.data_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((img_size, img_size)),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.data_transforms = data_transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = jumia_3650.PROCESSED_DATA_DIRNAME / self.file_paths[index]
        img = Image.open(img_path).convert("RGB")
        label = self.labels[index]

        img = self.data_transforms(img)

        return {"image": img, "label": torch.tensor(label, dtype=torch.long)}

    def create_dataloader(self, batch_size, shuffle=True, num_workers=0):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
