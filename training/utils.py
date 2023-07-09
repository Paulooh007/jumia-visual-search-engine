import json
import torch
import numpy as np
import os
from pathlib import Path

PARENT_DIR = Path(__file__).parent


def load_config(
    file_path=PARENT_DIR / "training_config.json",
):
    with open(file_path) as file:
        data = json.load(file)
    return data


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


# def fetch_scheduler(optimizer, scheduler):
#     if scheduler == "CosineAnnealingLR":
#         scheduler = lr_scheduler.CosineAnnealingLR(
#             optimizer, T_max=CONFIG["T_max"], eta_min=CONFIG["min_lr"]
#         )
#     elif scheduler == "CosineAnnealingWarmRestarts":
#         scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
#             optimizer, T_0=CONFIG["T_0"], eta_min=CONFIG["min_lr"]
#         )

#     elif scheduler == "OneCycleLR":
#         scheduler = lr_scheduler.OneCycleLR(
#             optimizer,
#             max_lr=CONFIG["learning_rate"],
#             total_steps=CONFIG["epochs"] * len(train_loader),
#         )

#     elif scheduler == None:
#         return None

#     return scheduler


if __name__ == "__main__":
    load_config()
