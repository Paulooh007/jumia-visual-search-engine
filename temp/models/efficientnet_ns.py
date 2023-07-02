import timm
import torch
import torch.nn as nn


class EfficientNet_b0_ns(nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNet_b0_ns, self).__init__()
        self.model = timm.create_model("tf_efficientnet_b0_ns", pretrained=pretrained)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.drop = nn.Dropout(p=0.2, inplace=False)
        self.fc = nn.Linear(in_features, 512)
        self.arc = ArcMarginProduct(
            512,
            CONFIG["num_classes"],
            s=CONFIG["s"],
            m=CONFIG["m"],
            easy_margin=CONFIG["ls_eps"],
            ls_eps=CONFIG["ls_eps"],
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


# model = JumiaModelV2(CONFIG["model_name"])
# model.to(CONFIG["device"])
# optimizer = optim.Adam(
#     model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"]
# )

# model = JumiaModel(CONFIG['model_name'])
# model.to(CONFIG['device']);
# optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'],
#                        weight_decay=CONFIG['weight_decay'])
