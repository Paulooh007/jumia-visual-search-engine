import copy
import gc
import math
import os
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

# Utils
import joblib
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn.functional as F

# import torch.nn as nn
import torch.optim as optim

# Sklearn Imports
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from image_search_engine.data import JumiaImageDataset
from image_search_engine.models import EfficientNet_b0_ns
from training.utils import load_config, set_seed

warnings.filterwarnings("ignore")

from colorama import Back, Fore, Style

b_ = Fore.BLUE
sr_ = Style.RESET_ALL

import time

# For descriptive error messages
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CONFIG = load_config()
PARENT_DIR = Path(__file__).parent

DATA_DIR = Path(PARENT_DIR) / "data"
TRAIN_DIR = DATA_DIR / "processed/train"
TEST_DIR = DATA_DIR / "processed/index"

TRAIN_DF = PARENT_DIR / "train_df.csv"
TEST_DF = PARENT_DIR / "test_df.csv"


WEIGHTS_DIR = PARENT_DIR / "artifacts/weights"


set_seed(CONFIG["seed"])


df_train = pd.read_csv(TRAIN_DF)
df_valid = pd.read_csv(TEST_DF)

encoder = LabelEncoder()
df_train["class"] = encoder.fit_transform(df_train["class"])
df_valid["class"] = encoder.fit_transform(df_valid["class"])

with open(PARENT_DIR / "artifacts/le.pkl", "wb") as fp:
    joblib.dump(encoder, fp)


model = EfficientNet_b0_ns()
model.to(DEVICE)
optimizer = optim.Adam(
    model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"]
)
from torchvision import transforms

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.GaussianBlur(7),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "valid": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}


def criterion(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data["image"].to(device, dtype=torch.float)
        labels = data["label"].to(device, dtype=torch.long)

        batch_size = images.size(0)

        outputs, emb = model(images, labels)
        loss = criterion(outputs, labels)
        loss = loss / CONFIG["n_accumulate"]
        if CONFIG["enable_amp_half_precision"] == True:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if (step + 1) % CONFIG["n_accumulate"] == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(
            Epoch=epoch, Train_Loss=epoch_loss, LR=optimizer.param_groups[0]["lr"]
        )
    gc.collect()

    return epoch_loss


@torch.inference_mode()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data["image"].to(device, dtype=torch.float)
        labels = data["label"].to(device, dtype=torch.long)

        batch_size = images.size(0)

        outputs, emb = model(images, labels)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(
            Epoch=epoch, Valid_Loss=epoch_loss, LR=optimizer.param_groups[0]["lr"]
        )

    gc.collect()

    return epoch_loss


def run_training(model, optimizer, scheduler, device, num_epochs):
    # To automatically log gradients

    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        train_epoch_loss = train_one_epoch(
            model,
            optimizer,
            scheduler,
            dataloader=train_loader,
            device=DEVICE,
            epoch=epoch,
        )

        val_epoch_loss = valid_one_epoch(
            model, valid_loader, device=DEVICE, epoch=epoch
        )

        history["Train Loss"].append(train_epoch_loss)
        history["Valid Loss"].append(val_epoch_loss)

        # deep copy the model
        if val_epoch_loss <= best_epoch_loss:
            print(
                f"{b_}Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})"
            )
            best_epoch_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = "Loss{:.4f}_epoch{:.0f}.bin".format(best_epoch_loss, epoch)
            torch.save(model.state_dict(), WEIGHTS_DIR / PATH)
            # Save a model file from the current directory
            print(f"Model Saved{sr_}")

        print()

    end = time.time()
    time_elapsed = end - start
    print(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            time_elapsed // 3600,
            (time_elapsed % 3600) // 60,
            (time_elapsed % 3600) % 60,
        )
    )
    print("Best Loss: {:.4f}".format(best_epoch_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history


def fetch_scheduler(optimizer):
    if CONFIG["scheduler"] == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG["T_max"], eta_min=CONFIG["min_lr"]
        )
    elif CONFIG["scheduler"] == "CosineAnnealingWarmRestarts":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=CONFIG["T_0"], eta_min=CONFIG["min_lr"]
        )

    elif CONFIG["scheduler"] == "OneCycleLR":
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=CONFIG["learning_rate"],
            total_steps=CONFIG["epochs"] * len(train_loader),
        )

    elif CONFIG["scheduler"] == None:
        return None

    return scheduler


print(WEIGHTS_DIR)


from image_search_engine.metadata import jumia_3650
from image_search_engine.data.base_data_module import JumiaImageDataset

TRAIN_FILENAME = jumia_3650.PROCESSED_DATA_DIRNAME / "train.csv"
TEST_FILENAME = jumia_3650.PROCESSED_DATA_DIRNAME / "test.csv"

train_dataset = JumiaImageDataset(TRAIN_FILENAME, data_transforms["train"])
train_loader = train_dataset.create_dataloader(CONFIG["train_batch_size"])

test_dataset = JumiaImageDataset(TEST_FILENAME, data_transforms["valid"])
valid_loader = test_dataset.create_dataloader(CONFIG["valid_batch_size"], shuffle=False)


# def main():
# train_dataset = JumiaImageDataset(df_train, transforms=data_transforms["train"])
# valid_dataset = JumiaImageDataset(df_valid, transforms=data_transforms["valid"])

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=CONFIG["train_batch_size"],
#     # num_workers=os.cpu_count(),
#     shuffle=True,
#     pin_memory=True,
#     drop_last=True,
# )
# valid_loader = DataLoader(
#     valid_dataset,
#     batch_size=CONFIG["valid_batch_size"],
#     # num_workers=os.cpu_count(),
#     shuffle=False,
#     pin_memory=True,
# )

scheduler = fetch_scheduler(optimizer)


model, history = run_training(model, optimizer, scheduler, device=DEVICE, num_epochs=3)


# if __name__ == "__main__":
#     main()
