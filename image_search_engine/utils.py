import json
from pathlib import Path

PACKAGE_DIR = Path(__file__).parent


def load_config(
    file_path=PACKAGE_DIR / "artifacts/config.json",
):
    with open(file_path) as file:
        data = json.load(file)
    return data


def load_json_file(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


if __name__ == "__main__":
    load_config()
