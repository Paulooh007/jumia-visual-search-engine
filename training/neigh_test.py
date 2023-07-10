import gradio as gr
import numpy as np

from image_search_engine.metadata import jumia_3650
from image_search_engine.models import EfficientNet_b0_ns
from image_search_engine.utils import load_json_file
from training.utils import DEVICE, TRAINING_DIR, load_serialized_object

MODEL_DIR = TRAINING_DIR / "artifacts/models"
WEIGHTS_DIR = TRAINING_DIR / "artifacts/weights"
MODEL_FILENAME = MODEL_DIR / "model_2023-07-09_15-17-45.pkl"
product_json = jumia_3650.PROCESSED_DATA_DIRNAME / "jumia_3650.json"
PRODUCT_IMG_DIR = jumia_3650.PROCESSED_DATA_DIRNAME / "train"


json_data = load_json_file(product_json)

model = EfficientNet_b0_ns(load_weights=True).to(DEVICE)

neigh = load_serialized_object(MODEL_FILENAME)

test_img = "training/test_img/10.jpg"


def _search(file):
    query_embedding = model.generate_embeddings(file.name)
    query_embedding = np.array(query_embedding).astype("float32")
    query_embedding.reshape(1, -1)

    _, idxs = neigh.kneighbors(query_embedding, return_distance=True)

    images = [PRODUCT_IMG_DIR / json_data[id]["filename"] for id in idxs[0]]

    return images


with gr.Blocks() as demo:
    input_image = gr.File(label="Uplaod image file")
    outputs = gr.Gallery(preview=True)

    input_image.upload(_search, inputs=input_image, outputs=outputs)


demo.launch()

# print(_search(test_img))
