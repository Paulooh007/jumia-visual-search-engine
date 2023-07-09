import os
import warnings
from pathlib import Path

# Pytorch Imports
import torch

import torch.optim as optim

from image_search_engine.data import Jumia3650Dataset
from image_search_engine.metadata import jumia_3650
from image_search_engine.models import EfficientNet_b0_ns
from training.image_transforms import data_transforms
from training.utils import fetch_scheduler, load_config, set_seed

warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CONFIG = load_config()
TRAINING_DIR = Path(__file__).parent

TRAIN_FILENAME = jumia_3650.PROCESSED_DATA_DIRNAME / "train.csv"
TEST_FILENAME = jumia_3650.PROCESSED_DATA_DIRNAME / "test.csv"

WEIGHTS_DIR = TRAINING_DIR / "artifacts/weights"

set_seed(CONFIG["seed"])

train_dataset = Jumia3650Dataset(TRAIN_FILENAME, data_transforms["train"])
train_loader = train_dataset.create_dataloader(CONFIG["train_batch_size"])

test_dataset = Jumia3650Dataset(TEST_FILENAME)
valid_loader = test_dataset.create_dataloader(CONFIG["valid_batch_size"], shuffle=False)

print(train_dataset.class_to_idx)


# model = EfficientNet_b0_ns().to(DEVICE)
# optimizer = optim.Adam(
#     model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"]
# )
# scheduler = fetch_scheduler(optimizer, len(train_loader))

# history = model.run_training(
#     train_loader,
#     valid_loader,
#     optimizer,
#     scheduler,
#     device=DEVICE,
#     num_epochs=3,
#     weights_dir=WEIGHTS_DIR,
# )
