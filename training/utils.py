import json
import os
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.optim import lr_scheduler

TRAINING_DIR = Path(__file__).parent
PROJECT_DIR = TRAINING_DIR.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(
    file_path=TRAINING_DIR / "training_config.json",
):
    with open(file_path) as file:
        data = json.load(file)
    return data


def set_seed(seed=42):
    """Sets the seed of the entire notebook
    so results are the same every time we run.
    This is for REPRODUCIBILITY."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


CONFIG = load_config()


def fetch_scheduler(optimizer, dl_len):
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
            total_steps=CONFIG["epochs"] * dl_len,
        )

    elif CONFIG["scheduler"] is None:
        return None

    return scheduler


def serialize_object(obj, file_path):
    try:
        joblib.dump(obj, file_path)
        print(f"Object serialized and saved successfully: {file_path}")
    except Exception as e:
        print(f"Error serializing object: {str(e)}")


def load_serialized_object(file_path):
    try:
        obj = joblib.load(file_path)
        return obj
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error loading serialized object: {str(e)}")


if __name__ == "__main__":
    load_config()
