import os
from pathlib import Path


PARENT_DIR = str(Path(__file__).resolve().parents[1])

# PARENT_DIR = Path(__file__).parent
DATA_DIR = Path(PARENT_DIR) / "data"

TRAIN_DF = "train_df.csv"
TEST_DF = "test_df.csv"


ROOT_DIR = "/artifacts/weights"

TRAIN_DIR = DATA_DIR / "processed/train"
TEST_DIR = DATA_DIR / "processed/index"

print(TRAIN_DIR)
print(TEST_DIR)
print(PARENT_DIR, ROOT_DIR)
