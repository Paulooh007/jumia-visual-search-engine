from image_search_engine.metadata import jumia_3650
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


TRAIN_FILENAME = jumia_3650.PROCESSED_DATA_DIRNAME / "train_df.csv"

encoder = LabelEncoder()


class JumiaImageDataset(Dataset):
    def __init__(self, data_filename, transforms=None):
        self.df = pd.read_csv(data_filename)
        self.file_paths = self.df["filepath"].values
        self.labels = encoder.fit_transform(self.df["class"])
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = jumia_3650.PROCESSED_DATA_DIRNAME / self.file_paths[index]
        img = Image.open(img_path).convert("RGB")
        label = self.labels[index]

        if self.transforms:
            img = self.transforms(img)

        return {"image": img, "label": torch.tensor(label, dtype=torch.long)}

    def create_dataloader(self, batch_size, shuffle=True, num_workers=0):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
