import numpy as np
import torch

from image_search_engine import utils
from image_search_engine.models import EfficientNet_b0_ns

MODEL_FILE = "model.pt"
INDEX_FILE = "index.pkl"


class JumiaProductSearch:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = utils.STAGED_MODEL_DIR / MODEL_FILE
        self.model = EfficientNet_b0_ns()
        self.model.load_state_dict(torch.load(model_path))
        self.index = utils.load_serialized_object(utils.STAGED_MODEL_DIR / INDEX_FILE)

    def _encode(self, image):
        query_embedding = self.model.generate_embeddings(str(image))
        query_embedding = np.array(query_embedding).astype("float32")
        query_embedding.reshape(1, -1)
        return query_embedding

    def search(self, image):
        query_embedding = self._encode(image)
        distances, idxs = self.index.kneighbors(query_embedding, return_distance=True)
        return idxs


if __name__ == "__main__":
    search = JumiaProductSearch()
    test_img = utils.PACKAGE_DIR / "tests/test_img/1.jpg"
    idx = search.search(test_img)
    print(idx)
