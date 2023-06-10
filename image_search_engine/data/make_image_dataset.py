import pandas as pd
import os
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split


project_dir = Path(__file__).resolve().parents[2]
data_dir = project_dir / "data"
raw_data_dir = data_dir / "raw"
interim_data_dir = data_dir / "interim"
processed_data_dir = data_dir / "processed"
raw_img_dir = interim_data_dir / "product_images_raw"


def move_images(filenames, new_dir):
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    for file in filenames:
        shutil.copy(raw_img_dir / file, os.path.join(new_dir, file))


def prepare_train_test():
    img_list = os.listdir(raw_img_dir)
    product_list = set(["_".join(string.split("_")[:-1]) for string in img_list])
    categories = set(["_".join(string.split("_")[:-2]) for string in img_list])

    df = pd.read_csv(raw_data_dir / "raw_test.csv")
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
    prepare_train_test()
