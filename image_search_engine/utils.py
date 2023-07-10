import json
from pathlib import Path
import joblib

PACKAGE_DIR = Path(__file__).parent
STAGED_MODEL_DIR = PACKAGE_DIR / "artifacts/model_staged"


def load_serialized_object(file_path):
    try:
        obj = joblib.load(file_path)
        return obj
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error loading serialized object: {str(e)}")


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
