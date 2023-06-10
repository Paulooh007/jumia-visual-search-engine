import pandas as pd
import requests
import os
import shutil
from pathlib import Path
import zipfile
from tqdm import tqdm

from sklearn.model_selection import train_test_split

project_dir = Path(__file__).resolve().parents[2]
data_dir = project_dir / "data"
raw_data_dir = data_dir / "raw"
interim_data_dir = data_dir / "interim"
processed_data_dir = data_dir / "processed"
raw_img_dir = interim_data_dir / "product_images_raw"


def zip_folder(source_dir, out_dir):
    directory = Path(f"{source_dir}/")

    with zipfile.ZipFile(out_dir, mode="w") as archive:
        for file_path in tqdm(directory.iterdir(), desc="Zipping..."):
            archive.write(file_path, arcname=file_path.name)


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


def prepare_train_test():
    img_list = os.listdir(raw_img_dir)
    product_list = set(["_".join(string.split("_")[:-1]) for string in img_list])
    categories = set(["_".join(string.split("_")[:-2]) for string in img_list])

    df = pd.read_csv(raw_data_dir / "raw.csv")
    df = df[df["product_id"].isin(product_list)]

    train, index = train_test_split(
        df, test_size=0.2, shuffle=True, stratify=df["product_category"]
    )

    train_filenames = []
    index_filenames = []
    train_product_id = train.product_id.tolist()
    for img in img_list:
        img_prod_id = "_".join(img.split("_")[:-1])
        if img_prod_id in train_product_id:
            train_filenames.append(img)
        else:
            index_filenames.append(img)

    train_summary = {
        category: sum(img.startswith(category) for img in train_filenames)
        for category in categories
    }

    test_summary = {
        category: sum(img.startswith(category) for img in index_filenames)
        for category in categories
    }

    train.to_csv(processed_data_dir / "train.csv", index=False)
    index.to_csv(processed_data_dir / "index.csv", index=False)

    move_images(train_filenames, processed_data_dir / "train")
    move_images(index_filenames, processed_data_dir / "index")

    for key in train_summary:
        print(f"{key}: \n  Train: {train_summary[key]}\n  Test: {test_summary[key]}")

    print(len(train_filenames), len(index_filenames))


if __name__ == "__main__":
    df = pd.read_csv(raw_data_dir / "raw.csv").dropna()
    download_images(df, raw_img_dir)
    print(f"{raw_img_dir}...")
    zip_folder(raw_img_dir, raw_data_dir / "product_images_raw.zip")
    prepare_train_test()
