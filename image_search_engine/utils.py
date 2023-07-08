import json
from pathlib import Path

parent_dir = Path(__file__).parent


def load_config(
    file_path=parent_dir / "artifacts/config.json",
):
    with open(file_path) as file:
        data = json.load(file)
    return data


if __name__ == "__main__":
    load_config()
