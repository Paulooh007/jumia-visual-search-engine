import os
from pathlib import Path

import pinecone
from dotenv import load_dotenv
from tqdm.auto import tqdm

from image_search_engine.metadata import jumia_3650
from image_search_engine.utils import (
    PACKAGE_DIR,
    load_json_file,
    load_serialized_object,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
EMBEDDINGS_FILE = PACKAGE_DIR / "artifacts/embeddings/embed_2023-07-09_15-17-45.pkl"
INDEX_NAME = "jumia-product-search"
INDEX_DIMENSION = 512
BATCH_SIZE = 300
INDEX_NAME = "jumia-product-embeddings"

load_dotenv(PROJECT_DIR / ".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")


embeddings = load_serialized_object(EMBEDDINGS_FILE)
embeddings = [vector.tolist() for vector in embeddings]

metadata = load_json_file(jumia_3650.PROCESSED_DATA_DIRNAME / "jumia_3650.json")

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

if INDEX_NAME not in pinecone.list_indexes():
    # now create the new index
    pinecone.create_index(
        INDEX_NAME,
        dimension=INDEX_DIMENSION,
        metric="cosine",
        pod_type="s1",
        pods=1,
    )

# connect to index
index = pinecone.Index(INDEX_NAME)
print(index.describe_index_stats())


n_embeddings = len(embeddings)

for i in tqdm(range(0, n_embeddings, BATCH_SIZE)):
    ids = [f"JUMIA_3650.{idx}" for idx in range(i, i + BATCH_SIZE)]
    batch_embeddings = embeddings[i : i + BATCH_SIZE]
    batch_metadata = metadata[i : i + BATCH_SIZE]

    index.upsert(zip(ids, batch_embeddings, batch_metadata))
