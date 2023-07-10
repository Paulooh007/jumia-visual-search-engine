from training.utils import DEVICE
from models import EfficientNet_b0_ns
from utils import PACKAGE_DIR
import torch


model = EfficientNet_b0_ns(load_weights=True).to(DEVICE)
torch.save(model.state_dict(), PACKAGE_DIR / "artifacts/model_staged/model.pt")
