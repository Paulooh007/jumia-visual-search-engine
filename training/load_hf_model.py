from huggingface_hub import hf_hub_download
import joblib

REPO_ID = "paulokewunmi/jumia_3650_v1.0.0"
FILENAME = "model.joblib"

model = joblib.load(hf_hub_download(repo_id=REPO_ID, filename=FILENAME))

idx = model.generate_embeddings("training/test_img/1.jpg")
print(idx[0][:5])
