import os
from huggingface_hub import hf_hub_download
import joblib
import pinecone
from dotenv import load_dotenv
from pathlib import Path

from image_search_engine import utils
import gradio as gr

REPO_ID = "paulokewunmi/jumia_3650_v1.0.0"
FILENAME = "model.joblib"

PROJECT_DIR = Path(__file__).resolve().parents[1]
INDEX_NAME = "jumia-product-embeddings"

load_dotenv(PROJECT_DIR / ".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

model = joblib.load(hf_hub_download(repo_id=REPO_ID, filename=FILENAME))

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(INDEX_NAME)

print(index.describe_index_stats())
test_img = utils.PACKAGE_DIR / "tests/test_img/1.jpg"


def search(query):
    xq = model.generate_embeddings(query)
    res = index.query(xq[0].tolist(), top_k=10, include_metadata=True)
    images = []

    for i, record in enumerate(res["matches"]):
        metadata = record["metadata"]
        images.append(metadata["product_image_url"])
    return images


with gr.Blocks() as demo:
    input_image = gr.Image(label="Upload image file", type="filepath")
    outputs = gr.Gallery(preview=True).style(
        columns=[4], rows=[3], height="auto", width="auto"
    )
    input_image.upload(search, inputs=input_image, outputs=outputs)

demo.launch()
