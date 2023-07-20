from datetime import datetime

import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from image_search_engine.metadata import jumia_3650
from image_search_engine.models import EfficientNet_b0_ns
from training.utils import DEVICE, TRAINING_DIR, serialize_object

import random

import matplotlib.pyplot as plt

WEIGHTS_DIR = TRAINING_DIR / "artifacts/weights"
EMBEDDINGS_DIR = TRAINING_DIR / "artifacts/embeddings"
MODEL_DIR = TRAINING_DIR / "artifacts/models"
TRAIN_FILENAME = jumia_3650.PROCESSED_DATA_DIRNAME / "train.csv"

current_datetime = datetime.now()
# Format the datetime as a string
datetime_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")


df = pd.read_csv(TRAIN_FILENAME)

model = EfficientNet_b0_ns().to(DEVICE)
model.load_state_dict(
    torch.load(WEIGHTS_DIR / "Loss0.6555_epoch3.bin", map_location=DEVICE)
)

image_filenames = [
    str(jumia_3650.PROCESSED_DATA_DIRNAME) + f"/{path}" for path in list(df["filepath"])
]

embeddings = model.generate_embeddings(image_filenames)
serialize_object(embeddings, EMBEDDINGS_DIR / f"embed_{datetime_string}.pkl")


neigh = NearestNeighbors(n_neighbors=8, metric="cosine")
neigh.fit(embeddings)

serialize_object(neigh, MODEL_DIR / f"model_{datetime_string}.pkl")


def get_sim_img(img_path):
    img_emb = model.generate_embeddings(img_path)

    distances, idxs = neigh.kneighbors(img_emb, return_distance=True)
    conf = 1 - distances

    plt_path = []
    for id in idxs:
        plt_path += list(jumia_3650.PROCESSED_DATA_DIRNAME / df["filepath"][id])

    return plt_path


def plot_similar_images(query_image_path, similar_image_paths):
    num_similar_images = len(similar_image_paths)
    total_plots = num_similar_images + 1
    grid_size = (3, 3)

    # Check if the number of images is greater than the grid size
    if total_plots > grid_size[0] * grid_size[1]:
        print("Warning: Some images will not be displayed.")

    fig, axes = plt.subplots(*grid_size, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.4)

    # Plot the query image
    query_image = plt.imread(query_image_path)
    axes[0, 0].imshow(query_image)
    axes[0, 0].set_title("Query Image")
    axes[0, 0].axis("off")

    # Plot the similar images
    for i, image_path in enumerate(similar_image_paths, start=1):
        row = i // grid_size[1]
        col = i % grid_size[1]
        axes[row, col].imshow(plt.imread(image_path))
        axes[row, col].set_title(f"Similar Image {i}")
        axes[row, col].axis("off")

    # Hide any empty subplots
    for i in range(total_plots, grid_size[0] * grid_size[1]):
        row = i // grid_size[1]
        col = i % grid_size[1]
        axes[row, col].axis("off")

    plt.savefig(f"{random.randint(0,  100)}.png")
