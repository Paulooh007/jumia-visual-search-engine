import json
from pathlib import Path

package_dir = Path(__file__).resolve().parents[1]


def load_config(
    file_path=package_dir / "artifacts/config.json",
):
    with open(file_path) as file:
        data = json.load(file)
    return data


if __name__ == "__main__":
    load_config()
