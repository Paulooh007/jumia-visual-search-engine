import timm
import torch
import torch.nn as nn

from image_search_engine.models.arc_margin_product import ArcMarginProduct
from image_search_engine.models.base import BaseModel
from image_search_engine.models.gem_pooling import GeM
from image_search_engine.utils import PACKAGE_DIR

CLASSES = 8
SCALE = 10
MARGIN = 0.1
EMBEDING_SIZE = 512

WEIGHTS_DIR = PACKAGE_DIR / "artifacts/weights"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EfficientNet_b0_ns(BaseModel):
    def __init__(self, pretrained=True, load_weights=False):
        super(EfficientNet_b0_ns, self).__init__(predtrained=pretrained)
        self.model = timm.create_model("tf_efficientnet_b0_ns", pretrained=pretrained)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.drop = nn.Dropout(p=0.2, inplace=False)
        self.fc = nn.Linear(in_features, EMBEDING_SIZE)
        self.arc = ArcMarginProduct(
            EMBEDING_SIZE,
            CLASSES,
        )
        if load_weights:
            self.load_state_dict(
                torch.load(WEIGHTS_DIR / "Loss0.6555_epoch3.bin", map_location=DEVICE)
            )

    def forward(self, images, labels=None):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        pooled_drop = self.drop(pooled_features)
        emb = self.fc(pooled_drop)

        if labels is not None:
            output = self.arc(emb, labels)
            return output, emb
        else:
            return emb
