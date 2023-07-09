from training.utils import load_serialized_object
import numpy as np
import pandas as pd
import time
from image_search_engine.metadata import jumia_3650

# Reduce dimensionality using t-SNE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from image_search_engine.metadata import jumia_3650

from sklearn.decomposition import PCA

TRAIN_FILENAME = jumia_3650.PROCESSED_DATA_DIRNAME / "train.csv"


df = pd.read_csv(TRAIN_FILENAME)
images = [
    str(jumia_3650.PROCESSED_DATA_DIRNAME) + f"/{path}" for path in list(df["filepath"])
]

embeddings = load_serialized_object(
    "training/artifacts/embeddings/embed_2023-07-09_15-17-45.pkl"
)

embeddings = np.array(embeddings)


num_feature_dimensions = 100  # Set the number of features
pca = PCA(n_components=num_feature_dimensions)
pca.fit(embeddings)
feature_list_compressed = pca.transform(embeddings)

selected_features = embeddings[:1000]
selected_class_ids = df["class"].map(jumia_3650.CLASS_DICT)[:1000]
selected_filenames = images[:1000]


n_components = 2
verbose = 1
perplexity = 5
n_iter = 1000
metric = "euclidean"

time_start = time.time()
tsne_results = TSNE(
    n_components=n_components,
    verbose=verbose,
    perplexity=perplexity,
    n_iter=n_iter,
    metric=metric,
).fit_transform(selected_features)

print("t-SNE done! Time elapsed: {} seconds".format(time.time() - time_start))


color_map = plt.cm.get_cmap("coolwarm")
scatter_plot = plt.scatter(
    tsne_results[:, 0], tsne_results[:, 1], c=selected_class_ids, cmap=color_map
)
plt.colorbar(scatter_plot)
plt.savefig("tsne.png")


from PIL import Image

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data


def plot_images_in_2d(x, y, image_paths, axis=None, zoom=1):
    if axis is None:
        axis = plt.gca()
    x, y = np.atleast_1d(x, y)
    for x0, y0, image_path in zip(x, y, image_paths):
        image = Image.open(image_path)
        image.thumbnail((100, 100), Image.ANTIALIAS)
        img = OffsetImage(image, zoom=zoom)
        anno_box = AnnotationBbox(img, (x0, y0), xycoords="data", frameon=False)
        axis.add_artist(anno_box)
    axis.update_datalim(np.column_stack([x, y]))
    axis.autoscale()


def show_tsne(x, y, selected_filenames):
    fig, axis = plt.subplots()
    fig.set_size_inches(22, 22, forward=True)
    plot_images_in_2d(x, y, selected_filenames, zoom=0.6, axis=axis)
    plt.savefig("image_cluster.png")


show_tsne(tsne_results[:, 0], tsne_results[:, 1], selected_filenames)


print("succes")
