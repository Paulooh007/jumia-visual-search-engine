import json
from pathlib import Path

CWD = Path.cwd()


def load_config(
    file_path=CWD / "artifacts/config.json",
):
    with open(file_path) as file:
        data = json.load(file)
    return data


if __name__ == "__main__":
    load_config()
