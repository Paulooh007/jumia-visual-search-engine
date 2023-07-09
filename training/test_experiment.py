import os
import warnings
from pathlib import Path

# Pytorch Imports
import torch

# import torch.nn as nn
import torch.optim as optim
from colorama import Back, Fore, Style

# Sklearn Imports
from torch import nn
from torch.optim import lr_scheduler
from torchvision import transforms

from image_search_engine.data import Jumia3650Dataset
from image_search_engine.metadata import jumia_3650

# from image_search_engine.models import EfficientNet_b0_ns
# from image_search_engine.models.base import EfficientNet_b0_ns
from training.utils import load_config, set_seed

from image_search_engine.models import EfficientNet_b0_ns

warnings.filterwarnings("ignore")


b_ = Fore.BLUE
sr_ = Style.RESET_ALL

# For descriptive error messages
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CONFIG = load_config()
PARENT_DIR = Path(__file__).parent

TRAIN_FILENAME = jumia_3650.PROCESSED_DATA_DIRNAME / "train.csv"
TEST_FILENAME = jumia_3650.PROCESSED_DATA_DIRNAME / "test.csv"

WEIGHTS_DIR = PARENT_DIR / "artifacts/weights"

set_seed(CONFIG["seed"])


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


train_dataset = Jumia3650Dataset(TRAIN_FILENAME, data_transforms["train"])
train_loader = train_dataset.create_dataloader(CONFIG["train_batch_size"])

test_dataset = Jumia3650Dataset(TEST_FILENAME, data_transforms["valid"])
valid_loader = test_dataset.create_dataloader(CONFIG["valid_batch_size"], shuffle=False)


model = EfficientNet_b0_ns().to(DEVICE)
optimizer = optim.Adam(
    model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"]
)
scheduler = fetch_scheduler(optimizer)

history = model.run_training(
    train_loader,
    valid_loader,
    optimizer,
    scheduler,
    device=DEVICE,
    num_epochs=3,
    weights_dir=WEIGHTS_DIR,
)
