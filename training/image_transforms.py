from torchvision import transforms

from training.utils import load_config

CONFIG = load_config()

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
            ),
            transforms.GaussianBlur(7),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "valid": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}
