import sys
from pathlib import Path
import numpy as np
import pandas as pd

import os

import gc
import math
import copy
import random

# Pytorch Imports
import torch

# import torch.nn as nn
import torch.optim as optim
from torch import nn

import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

# Utils
import joblib

from tqdm import tqdm

from collections import defaultdict

# Sklearn Imports
from sklearn.preprocessing import LabelEncoder

import warnings

warnings.filterwarnings("ignore")

from colorama import Fore, Back, Style

b_ = Fore.BLUE
sr_ = Style.RESET_ALL

import time

# For descriptive error messages
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

PARENT_DIR = str(Path(__file__).resolve().parents[1])
sys.path.append(PARENT_DIR)

from image_search_engine.data import JumiaImageDataset
from image_search_engine.models import EfficientNet_b0_ns

from training.utils import load_config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CONFIG = load_config()
CWD = Path.cwd()
DATA_DIR = Path(PARENT_DIR) / "data"

TRAIN_DF = "train_df.csv"
TEST_DF = "test_df.csv"


ROOT_DIR = "/artifacts/weights"

TRAIN_DIR = DATA_DIR / "processed/train"
TEST_DIR = DATA_DIR / "processed/index"

print(TRAIN_DIR, TEST_DIR)


def set_seed(seed=42):
    """Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(CONFIG["seed"])


df_train = pd.read_csv(TRAIN_DF)
df_valid = pd.read_csv(TEST_DF)

encoder = LabelEncoder()
df_train["class"] = encoder.fit_transform(df_train["class"])
df_valid["class"] = encoder.fit_transform(df_valid["class"])

with open(CWD / "artifacts/le.pkl", "wb") as fp:
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
            torch.save(model.state_dict(), PATH)
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


# def main():
train_dataset = JumiaImageDataset(df_train, transforms=data_transforms["train"])
valid_dataset = JumiaImageDataset(df_valid, transforms=data_transforms["valid"])

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["train_batch_size"],
    # num_workers=os.cpu_count(),
    shuffle=True,
    pin_memory=True,
    drop_last=True,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=CONFIG["valid_batch_size"],
    # num_workers=os.cpu_count(),
    shuffle=False,
    pin_memory=True,
)

scheduler = fetch_scheduler(optimizer)


model, history = run_training(model, optimizer, scheduler, device=DEVICE, num_epochs=3)


# if __name__ == "__main__":
#     main()
