import torch
from models import EfficientNet_b0_ns

from image_search_engine.utils import PACKAGE_DIR
from training.utils import DEVICE

model = EfficientNet_b0_ns(load_weights=True).to(DEVICE)
torch.save(model.state_dict(), PACKAGE_DIR / "artifacts/model_staged/model.pt")
