import torch
import torch.nn as nn


import copy
import gc
import warnings
from collections import defaultdict
import numpy as np

from colorama import Back, Fore, Style

from PIL import Image

# Utils
import numpy as np

# Pytorch Imports
import torch
from torchvision import transforms

# import torch.nn as nn

import time

# Sklearn Imports
from torch import nn
from tqdm import tqdm


from colorama import Back, Fore, Style

b_ = Fore.BLUE
sr_ = Style.RESET_ALL

warnings.filterwarnings("ignore")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseModel(nn.Module):
    def __init__(self, predtrained=True) -> None:
        super(BaseModel, self).__init__()

    def forward(self):
        pass

    def _criterion(self, outputs, labels):
        return nn.CrossEntropyLoss()(outputs, labels)

    def _train_one_epoch(self, dataloader, optimizer, scheduler, device, epoch):
        self.train()

        dataset_size = 0
        running_loss = 0.0

        bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for step, data in bar:
            images = data["image"].to(device, dtype=torch.float)
            labels = data["label"].to(device, dtype=torch.long)

            batch_size = images.size(0)

            outputs, _ = self(images, labels)
            loss = self._criterion(outputs, labels)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            epoch_loss = running_loss / dataset_size

            bar.set_postfix(
                Epoch=epoch, Train_Loss=epoch_loss, LR=optimizer.param_groups[0]["lr"]
            )
        gc.collect()

        return epoch_loss

    @torch.inference_mode()
    def _valid_one_epoch(self, dataloader, optimizer, device, epoch):
        self.eval()

        dataset_size = 0
        running_loss = 0.0

        bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for step, data in bar:
            images = data["image"].to(device, dtype=torch.float)
            labels = data["label"].to(device, dtype=torch.long)

            batch_size = images.size(0)

            outputs, _ = self(images, labels)
            loss = self._criterion(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            epoch_loss = running_loss / dataset_size

            bar.set_postfix(
                Epoch=epoch, Valid_Loss=epoch_loss, LR=optimizer.param_groups[0]["lr"]
            )

        gc.collect()

        return epoch_loss

    def run_training(
        self,
        train_loader,
        valid_loader,
        optimizer,
        scheduler,
        device,
        num_epochs,
        weights_dir,
    ):
        # To automatically log gradients

        if torch.cuda.is_available():
            print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

        start = time.time()
        best_model_wts = copy.deepcopy(self.state_dict())
        best_epoch_loss = np.inf
        history = defaultdict(list)

        for epoch in range(1, num_epochs + 1):
            gc.collect()

            # dataloader, optimizer, scheduler, criterion, device, epoch
            train_epoch_loss = self._train_one_epoch(
                optimizer=optimizer,
                scheduler=scheduler,
                dataloader=train_loader,
                device=device,
                epoch=epoch,
            )
            val_epoch_loss = self._valid_one_epoch(
                valid_loader, optimizer, device=device, epoch=epoch
            )

            history["Train Loss"].append(train_epoch_loss)
            history["Valid Loss"].append(val_epoch_loss)

            # deep copy the model
            if val_epoch_loss <= best_epoch_loss:
                print(
                    f"{b_}Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})"
                )
                best_epoch_loss = val_epoch_loss
                best_model_wts = copy.deepcopy(self.state_dict())
                PATH = "Loss{:.4f}_epoch{:.0f}.bin".format(best_epoch_loss, epoch)
                torch.save(self.state_dict(), weights_dir / PATH)
                # Save a model file from the current directory
                print(f"Model Saved{sr_}")

            print()

        end = time.time()
        time_elapsed = end - start
        print(
            "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
                time_elapsed // 3600,
                (time_elapsed % 3600) // 60,
                (time_elapsed % 3600) % 60,
            )
        )
        print("Best Loss: {:.4f}".format(best_epoch_loss))

        # load best model weights
        self.load_state_dict(best_model_wts)

        return history

    def generate_embeddings(self, image_filepath, data_transforms=None):
        if data_transforms is None:
            data_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((224, 224)),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        self.eval()

        if isinstance(image_filepath, str):
            # Single image path
            image_paths = [image_filepath]
        elif isinstance(image_filepath, list):
            # List of image paths
            image_paths = image_filepath
        else:
            raise ValueError("Invalid image_filepath type. Must be a string or a list.")

        embeddings = []

        for path in image_paths:
            image = Image.open(path).convert("RGB")
            image = data_transforms(image)
            image = image.unsqueeze(0)  # Add batch dimension
            image = image.to(DEVICE)

            # Generate embedding
            with torch.no_grad():
                embedding = self(image)

            embeddings.append(embedding.squeeze().cpu().numpy())

        return embeddings

    def generate_embeddings_(self, image_data, data_transforms=None):
        if data_transforms is None:
            data_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((224, 224)),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        self.eval()

        if isinstance(image_data, str):
            # Single image path
            image_paths = [image_data]
        elif isinstance(image_data, list):
            # List of image paths
            image_paths = image_data
        elif isinstance(image_data, np.ndarray):
            # Single numpy array or list of numpy arrays
            image_paths = None
            if len(image_data.shape) == 3:
                image_data = np.expand_dims(image_data, axis=0)
            elif len(image_data.shape) == 4:
                image_paths = None
            else:
                raise ValueError("Invalid image_data shape. Must be 3D or 4D array.")
        else:
            raise ValueError(
                "Invalid image_data type. Must be a string, list, or numpy array."
            )

        embeddings = []

        if image_paths is not None:
            for path in image_paths:
                image = Image.open(path).convert("RGB")
                image = data_transforms(image)
                image = image.unsqueeze(0)  # Add batch dimension
                image = image.to(DEVICE)
                with torch.no_grad():
                    embedding = self(image)
                embeddings.append(embedding.squeeze().cpu().numpy())
        else:
            images = torch.from_numpy(image_data).to(DEVICE)
            with torch.no_grad():
                embeddings = self(images).cpu().numpy().tolist()

        return embeddings
