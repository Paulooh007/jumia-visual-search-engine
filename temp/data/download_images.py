import os
import shutil
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm
from utils import zip_folder

project_dir = Path(__file__).resolve().parents[2]
data_dir = project_dir / "data"
raw_data_dir = data_dir / "raw"
interim_data_dir = data_dir / "interim"
raw_img_dir = interim_data_dir / "product_images_raw"


def move_images(filenames, new_dir):
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    for file in filenames:
        shutil.copy(raw_img_dir / file, os.path.join(new_dir, file))


def download_images(df, save_dir):
    """
    Downloads images from URLs in a pandas dataframe and saves them to disk.

    Args:
        df (pandas.DataFrame): A pandas dataframe with a column of image URLs.
        save_dir (str): The directory to save the downloaded images to.
    """
    # create the directory to save the images if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, row in tqdm(df.iterrows(), desc="Downloading Images..", total=df.shape[0]):
        urls = row["product_images_url"].split(",")

        prefix = row["product_id"]

        # loop through each URL and download the image
        for j, url in enumerate(urls):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
            except (
                requests.exceptions.RequestException,
                requests.exceptions.Timeout,
            ) as e:
                print(f"Error downloading image from URL: {url}")
                print(e)
                continue

            filename = f"{prefix}_{j}.jpg"  # assuming the images are all JPEGs
            save_path = os.path.join(save_dir, filename)
            with open(save_path, "wb") as f:
                try:
                    f.write(response.content)
                except IOError as e:
                    print(f"Error saving image to disk: {filename}")
                    print(e)
                    continue


if __name__ == "__main__":
    df = pd.read_csv(raw_data_dir / "raw_test.csv").dropna()
    download_images(df, raw_img_dir)
    print(f"{raw_img_dir}...")
    zip_folder(raw_img_dir, raw_data_dir / "product_images_raw.zip")
