import json
from pathlib import Path
import smart_open
import joblib
from typing import Union
from PIL import Image

PACKAGE_DIR = Path(__file__).parent
STAGED_MODEL_DIR = PACKAGE_DIR / "artifacts/model_staged"


def read_image_pil(image_uri: Union[Path, str]) -> Image:
    with smart_open.open(image_uri, "rb") as image_file:
        return read_image_pil_file(image_file)


def read_image_pil_file(image_file) -> Image:
    with Image.open(image_file) as image:
        image = image.convert(mode=image.mode)
    return image


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


def reduce_image_filesize(input_path, output_path, quality=85):
    # Open the image using Pillow with reduced quality
    image = Image.open(input_path)
    image.save(output_path, optimize=True, quality=quality)

    # Print the original and reduced file sizes
    original_size = image.size
    reduced_image = Image.open(output_path)
    reduced_size = reduced_image.size
    print(f"Original Size: {original_size} | Reduced Size: {reduced_size}")

    # Close the images
    image.close()
    reduced_image.close()


if __name__ == "__main__":
    load_config()
