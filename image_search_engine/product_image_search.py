import numpy as np
import torch
import os

from image_search_engine import utils
from image_search_engine.models import EfficientNet_b0_ns
from typing import Union
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
import pinecone

MODEL_FILE = "model.pt"
INDEX_FILE = "index.pkl"

PROJECT_DIR = utils.PACKAGE_DIR.parent
INDEX_NAME = "jumia-product-embeddings"

load_dotenv(PROJECT_DIR / ".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")


def load_pinecone_existing_index():
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    index = pinecone.Index(INDEX_NAME)
    return index


index = load_pinecone_existing_index()


class JumiaProductSearch:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = utils.STAGED_MODEL_DIR / MODEL_FILE
        self.model = EfficientNet_b0_ns()
        self.model.load_state_dict(torch.load(model_path))
        self.index = utils.load_serialized_object(utils.STAGED_MODEL_DIR / INDEX_FILE)

    def _encode(self, image: Union[str, Path, Image.Image]):
        image_pil = image
        if not isinstance(image, Image.Image):
            image_pil = utils.read_image_pil(image)

        query_embedding = self.model.generate_embeddings(image_pil)

        return query_embedding

    def search(self, image, k):
        xq = self._encode(image)
        result = index.query(xq, top_k=k, include_metadata=True)
        return result

    def search_nn(self, image):
        query_embedding = self.encode(image)
        distances, idxs = self.index.kneighbors(query_embedding, return_distance=True)
        return idxs


if __name__ == "__main__":
    search = JumiaProductSearch()
    test_img = utils.PACKAGE_DIR / "tests/test_img/1.jpg"
    idx = search.search(test_img)
    print(idx)
