import argparse
import os
import random
import sys

import albumentations as A
import cv2
import lightning as pl
import numpy as np
import pandas as pd
import torch
import wandb
from albumentations.pytorch import ToTensorV2
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from monai.losses import DiceCELoss, DiceLoss, FocalLoss
from monai.metrics import compute_hausdorff_distance
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import (
    BinaryPrecision,
    BinaryRecall,
    MulticlassAccuracy,
)
from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU

from models import SEGFORMER_ENCODER, segmentation_model_selector


# Set the same seed for all the generators
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    pl.seed_everything(SEED, workers=True)


NOISE_TYPES = ("gaussian", "poisson", "blur")

# Poisson: photons per pixel at full intensity, log-interpolated from 0% to 100% noise
POISSON_PEAK_MIN_NOISE = 1000.0  # if you want to train under poisson noise
POISSON_PEAK_MAX_NOISE = 1.0  # image maximum

# Blur: Gaussian blur std in pixels at 100% (for 512x512 images)
BLUR_SIGMA_MAX = 10.0


# Function that adds gaussian noise
def add_gaussian_noise(image, noise_level, rng):
    """Additive Gaussian noise with std = noise_level * full intensity range (0-1 → 0%-100%)."""
    if noise_level <= 0:
        return image
    img = image.astype(np.float32) / 255.0

    # Same noise for all channels
    noise = rng.normal(0.0, noise_level, size=(*img.shape[:2], 1)).astype(np.float32)
    img = np.clip(img + noise, 0.0, 1.0)
    return (img * 255.0).round().astype(np.uint8)


# Function that adds poisson noise
def add_poisson_noise(image, noise_level, rng):

    if noise_level <= 0:
        return image
    peak = (
        POISSON_PEAK_MIN_NOISE
        * (POISSON_PEAK_MAX_NOISE / POISSON_PEAK_MIN_NOISE) ** noise_level
    )
    img = image.astype(np.float32) / 255.0
    if np.array_equal(image[..., 0], image[..., 1]) and np.array_equal(
        image[..., 0], image[..., 2]
    ):
        # Grayscale stored as 3 identical channels (ARCADE): same noise on all channels
        noisy = rng.poisson(img[..., :1] * peak) / peak
        noisy = np.repeat(noisy, 3, axis=2)
    else:
        noisy = rng.poisson(img * peak) / peak
    img = np.clip(noisy, 0.0, 1.0)
    return (img * 255.0).round().astype(np.uint8)


# Adds blur to the image
def add_blur(image, noise_level, rng=None):
    """Gaussian blur (defocus / motion-like loss of detail), std = noise_level * BLUR_SIGMA_MAX px."""
    if noise_level <= 0:
        return image
    sigma = noise_level * BLUR_SIGMA_MAX
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)


def add_noise(image, noise_type, noise_level, rng):
    if noise_type == "gaussian":
        return add_gaussian_noise(image, noise_level, rng)
    elif noise_type == "poisson":
        return add_poisson_noise(image, noise_level, rng)
    elif noise_type == "blur":
        return add_blur(image, noise_level, rng)
    raise ValueError(f"Unknown noise type: {noise_type}. Choose from {NOISE_TYPES}")


# Dataset class for Segmentation task
class Single_Target_Segmentation(Dataset):
    def __init__(
        self,
        df_dataset,
        classes=2,
        mode="training",
        dataset_type="artifact",
        get_names=False,
        noise_level=0.0,  # 0.0 = clean image, 1.0 = 100% noise
        noise_seed=0,  # base seed for the fixed validation/test noise
        noise_type="gaussian",  # "gaussian", "poisson" or "blur"
    ):

        self.database = df_dataset
        self.mode = mode
        self.dataset_type = dataset_type
        self.get_names = get_names
        self.noise_level = noise_level
        self.noise_seed = noise_seed
        self.noise_type = noise_type

        # Value to fill the empty space after transformations (e.g., rotation, padding).
        fill_value = 0

        # Normalize images
        self.train_transform = A.Compose(
            [
                # Spatial transformations
                A.OneOf(
                    [
                        A.HorizontalFlip(p=1.0),
                        A.VerticalFlip(p=1.0),
                    ],
                    p=0.5,
                ),
                A.Rotate(limit=90, border_mode=0, fill_mask=fill_value, p=0.7),
                # ensure crop always works
                A.PadIfNeeded(
                    min_height=512, min_width=512, border_mode=0, fill_mask=fill_value
                ),
                A.RandomCrop(height=512, width=512, p=1.0),
                # Color & Stain Variability
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.3
                ),
                # Albumentations normalization following imagenet values
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )

        # Normalize validation images
        self.val_transform = A.Compose(
            [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )
        # This code requires the following organization for the image, pairs
        # Column 0 = mask paths
        # Column 1 = tile paths
        self.mask_path = df_dataset["mask_path"]
        self.tile_path = df_dataset["tile_path"]

        # Number of classes in total on masks
        self.classes = classes

    def transform(self, image, mask):

        # Convert PIL to numpy
        image = np.array(image)
        mask = np.array(mask)

        # ------------------------------
        # Apply augmentations
        # ------------------------------
        if self.mode == "training":
            augmented = self.train_transform(image=image, mask=mask)
        else:
            augmented = self.val_transform(image=image, mask=mask)

        image = augmented["image"]  # torch tensor CHW float32 normalized
        mask = augmented["mask"]  # torch tensor H×W integer mask (0–255)

        mask = mask.long()

        return image, mask

    # Get number of samples of the dataset
    def __len__(self):
        return len(self.mask_path)

    # Get a random sample of the dataset
    def __getitem__(self, idx):

        image = cv2.imread(self.tile_path[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2 loads in BGR, we need RGB

        mask = cv2.imread(self.mask_path[idx], cv2.IMREAD_GRAYSCALE)

        # Training: new noise each time; val/test: fixed noise per image (same for every model)
        if self.mode == "training":
            rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        else:
            rng = np.random.default_rng(self.noise_seed + idx)
        image = add_noise(image, self.noise_type, self.noise_level, rng)

        # Applies transformations and checks inside if everything is ok
        image, mask = self.transform(image, mask)

        # If we use this code to mask datasets, we need the names of the files
        if self.get_names == False:
            return image, mask
        else:
            return image, mask, self.tile_path[idx], self.mask_path[idx]


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df,
        val_df,
        test_df,
        classes=2,
        dataset_type="kvasir",
        batch_size=8,
        num_workers=4,
        prefetch_factor=2,
        get_names=False,
        noise_level=0.0,
        noise_type="gaussian",
    ):
        super().__init__()
        self.noise_level = noise_level
        self.noise_type = noise_type
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.classes = classes
        self.dataset_type = dataset_type

        self.train_df = self.train_df.reset_index(drop=True)
        self.val_df = self.val_df.reset_index(drop=True)
        self.test_df = self.test_df.reset_index(drop=True)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor if self.num_workers > 0 else None

        self.get_names = get_names

    def setup(self, stage=None):
        self.train_dataset = Single_Target_Segmentation(
            self.train_df,
            mode="training",
            classes=self.classes,
            dataset_type=self.dataset_type,
            get_names=self.get_names,
            noise_level=self.noise_level,
            noise_type=self.noise_type,
        )
        self.val_dataset = Single_Target_Segmentation(
            self.val_df,
            mode="validation",
            classes=self.classes,
            dataset_type=self.dataset_type,
            get_names=self.get_names,
            noise_level=self.noise_level,
            noise_type=self.noise_type,
        )
        self.test_dataset = Single_Target_Segmentation(
            self.test_df,
            mode="test",  # no augmentation, same as validation
            classes=self.classes,
            dataset_type=self.dataset_type,
            get_names=self.get_names,
            noise_level=self.noise_level,
            noise_type=self.noise_type,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=False,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=False,
        )


# Training loop class using PyTorch Lightning
class LitSegModel(pl.LightningModule):
    def __init__(
        self,
        model,
        num_classes=7,
        lr=1e-3,
        loss_fn="cross_entropy",
        weight_decay=1e-3,
        wandb_active=False,
        dataset_type="kvasir",
        deep_supervision=False,
    ):

        super().__init__()
        self.model = model
        self.lr = lr
        self.num_classes = num_classes
        # Recurrent models: loss on the output of every step (0 = base model, ..., n_iter),
        # so every number of iterations is trained and later steps must do better
        self.deep_supervision = deep_supervision and hasattr(model, "n_iter")

        self.wandb_active = wandb_active
        self.loss_fn_name = loss_fn
        self.weight_decay = weight_decay

        self.class_names = ["Background", "Target"]

        # Define loss function
        if loss_fn == "cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss()

        elif loss_fn == "focal_loss":
            self.loss_fn = FocalLoss(
                to_onehot_y=True,
                gamma=2.0,
                # weight=torch.tensor([0.1, 0.6, 0.6, 0.6, 0.6, 0.6, 0.2])
            )
        elif loss_fn == "dice_loss":
            self.loss_fn = DiceLoss(
                to_onehot_y=True,  # converts (B,1,H,W) class indices → one-hot
                softmax=True,  # applies softmax to logits
            )
        elif loss_fn == "dice_ce_loss":
            self.loss_fn = DiceCELoss(
                to_onehot_y=True,
                softmax=True,
                lambda_dice=0.6,
                lambda_ce=0.4,
                include_background=False,
            )
        else:
            raise ValueError(
                "Invalid loss function name. Choose 'cross_entropy' or 'focal_loss'."
            )

        # Per-class outputs, averaged over metric_keep_classes at epoch end
        self.val_acc = MulticlassAccuracy(num_classes=self.num_classes, average=None)
        self.val_iou = MeanIoU(
            num_classes=self.num_classes, input_format="index", per_class=True
        )
        self.val_dice = GeneralizedDiceScore(
            num_classes=self.num_classes,
            input_format="index",
            weight_type="simple",
            per_class=True,
        )
        # Precision and recall of the target class (pixels with label 1)
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()

        # HD95 per test image, computed only in the test loop (CPU-heavy)
        self.test_hd95_values = []

    def update_metrics(self, preds, mask):
        self.val_acc.update(preds, mask)
        self.val_iou.update(preds, mask)
        self.val_dice.update(preds, mask)
        self.val_precision.update(preds, mask)
        self.val_recall.update(preds, mask)

    def compute_and_reset_metrics(self):
        # Metrics of the target class only (index 1), background is excluded
        metrics = {
            "acc": self.val_acc.compute()[1],
            "iou": self.val_iou.compute()[1],
            "dice": self.val_dice.compute()[1],
            "precision": self.val_precision.compute(),
            "recall": self.val_recall.compute(),
        }
        self.val_acc.reset()
        self.val_iou.reset()
        self.val_dice.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        return metrics

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, logits, mask):
        if self.loss_fn_name == "cross_entropy":
            return self.loss_fn(logits, mask.long())
        return self.loss_fn(
            logits, mask.unsqueeze(1).long()
        )  # Monai requires (B,1,H,W)

    def training_step(self, batch, batch_idx):

        img, mask = batch  # mask shape (B, H, W)

        if self.deep_supervision:
            # Weighted mean of the step losses, weight t+1 for step t: later steps count more
            outputs = self.model(img, return_all=True)
            step_losses = [self.compute_loss(logits, mask) for logits in outputs]
            # Models with supervise_unary = False get no loss on step 0 (the base model's output)
            first = 0 if getattr(self.model, "supervise_unary", True) else 1
            weights = [t + 1 if t >= first else 0 for t in range(len(step_losses))]
            loss = sum(w * l for w, l in zip(weights, step_losses)) / sum(weights)
            self.log("train_loss_step0", step_losses[0], on_step=False, on_epoch=True)
            self.log("train_loss_last", step_losses[-1], on_step=False, on_epoch=True)
        else:
            logits = self.forward(img)  # shape (B, num_classes, H, W)
            loss = self.compute_loss(logits, mask)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):

        img, mask = batch  # mask shape (B, H, W)
        logits = self.forward(img)  # shape (B, num_classes, H, W)

        if self.loss_fn_name == "cross_entropy":
            loss = self.loss_fn(logits, mask.long())
        else:
            loss = self.loss_fn(
                logits, mask.unsqueeze(1).long()
            )  # Monai requires (B,1,H,W)

        # Compute accuracy, iou, dice, precision and recall
        preds = logits.argmax(dim=1)
        self.update_metrics(preds, mask)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    # Test loop, to use as Trainer.test(model, datamodule)
    def test_step(self, batch, batch_idx):

        img, mask = batch  # mask shape (B, H, W)
        logits = self.forward(img)  # shape (B, num_classes, H, W)

        # Compute accuracy, iou, dice, precision and recall
        preds = logits.argmax(dim=1)
        self.update_metrics(preds, mask)

        # HD95 per image on the target class. An empty prediction (or mask) gives
        # nan/inf, which is replaced by the image diagonal (worst possible distance)
        H, W = mask.shape[-2:]
        hd95 = compute_hausdorff_distance(
            F.one_hot(preds, self.num_classes).permute(0, 3, 1, 2),
            F.one_hot(mask, self.num_classes).permute(0, 3, 1, 2),
            include_background=False,
            percentile=95,
        )
        diagonal = float((H**2 + W**2) ** 0.5)
        hd95 = torch.nan_to_num(hd95, nan=diagonal, posinf=diagonal)
        self.test_hd95_values.append(hd95.flatten().cpu())

    # Just for wandb to get the metrics at the end of each epoch
    def on_validation_epoch_end(self):

        print(f"\n--- Epoch {self.current_epoch} Target-Class Metrics ---")

        metrics = self.compute_and_reset_metrics()
        for name, value in metrics.items():
            self.log(f"val_{name}_epoch", value, prog_bar=name in ("iou", "dice"))

        print("  " + ", ".join(f"{k}: {v.item():.4f}" for k, v in metrics.items()))

    def on_test_epoch_end(self):

        print("\n--- Test Target-Class Metrics ---")

        metrics = self.compute_and_reset_metrics()
        metrics["hd95"] = torch.cat(self.test_hd95_values).mean()
        self.test_hd95_values = []

        for name, value in metrics.items():
            self.log(f"test_{name}", value)

        print("  " + ", ".join(f"{k}: {v.item():.4f}" for k, v in metrics.items()))

    # PRedict the mask for every
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        img, mask = batch

        logits = self.forward(img)
        preds = logits.argmax(dim=1)

        return {
            "image": img.detach().cpu(),
            "mask": mask.detach().cpu(),
            "preds": preds.detach().cpu(),
        }

    def configure_optimizers(self):
        # Define different lr for the base network and for the recurrent block
        block_params = [p for n, p in self.named_parameters() if ".block." in n]
        other_params = [p for n, p in self.named_parameters() if ".block." not in n]
        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "weight_decay": self.weight_decay},
                {"params": block_params, "lr": self.lr * 3, "weight_decay": 0.0},
            ],
            lr=self.lr,
        )

        # Cosine decay of the learning rate over all epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# MAIN CODE
if __name__ == "__main__":
    ########################################
    # Setting the variables
    ########################################
    # Init code
    SEED = 4

    DATA_SPLIT_SEED = 44
    WANDB_KEY = "YOUR_WANDB_KEY"
    WANDB = True

    # Hyperparameters
    BATCH_SIZE = 8
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4  # L2 regularization strength

    # Command line arguments (used by launch_runs.py); defaults are used when run directly
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arg_model",
        default="unet_crf",
        choices=[
            "unet",
            "unet_crf",
            "unet_hopfield",
            "unet_gru",
            "unet_gru_local",
            "segformer",
            "segformer_crf",
            "segformer_hopfield",
            "segformer_gru",
            "segformer_gru_local",
        ],
    )
    parser.add_argument(
        "--dataset_type", default="kvasir", choices=["kvasir", "arcade"]
    )
    parser.add_argument("--n_iter", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--deep_supervision",
        type=int,
        default=1,
        help="1: loss on the output of every recurrent step (recurrent models only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Training seed (the data split always uses DATA_SPLIT_SEED)",
    )
    args = parser.parse_args()
    SEED = args.seed

    # Number of recurrent iterations for the CRF or Hopfield block
    N_ITER = args.n_iter
    DEEP_SUPERVISION = bool(args.deep_supervision)
    MAX_EPOCHS = args.max_epochs
    LEARNING_RATE = args.lr
    # Noise added to train/val/test images
    NOISE_TYPE = "gaussian"  # Choose between "gaussian", "poisson" or "blur"
    NOISE_LEVEL = 0.0  # 0.0 = clean, 0.10 = 10%, 1.0 = 100%
    # Run name tag: "noise" for Gaussian (keeps old run names), otherwise the noise type
    NOISE_TAG = "noise" if NOISE_TYPE == "gaussian" else NOISE_TYPE

    # Model Related variables
    DATASET_TYPE = args.dataset_type  # Choose between kvasir and arcade

    arg_MODEL = args.arg_model  # Choose between "unet", "unet_crf" or "unet_hopfield"

    # Setting the seed
    set_seed(SEED)

    model = segmentation_model_selector(
        arg_model=arg_MODEL,
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=2,
        n_iter=N_ITER,
    )

    # Training related variables
    LOSS_FUNC = "dice_ce_loss"  # Choose between "cross_entropy", "focal_loss", "dice_loss", or "dice_ce_loss"
    PREFETCH_FACTOR = 2  # Number of batches to prefetch for DataLoader
    NUM_WORKERS = 8
    # Only models with a recurrent block get the number of iterations in the name
    # Deep supervision runs get "_ds" after the loss name (keeps them apart from older runs)
    if arg_MODEL.endswith(("_crf", "_hopfield", "_gru", "_gru_local")):
        LOSS_TAG = f"{LOSS_FUNC}_ds" if DEEP_SUPERVISION else LOSS_FUNC
        RUN_NAME = f"{DATASET_TYPE}_{arg_MODEL}_{LOSS_TAG}_{NOISE_TAG}{round(NOISE_LEVEL * 100)}_seed{SEED}_niter{N_ITER}"
    else:
        RUN_NAME = f"{DATASET_TYPE}_{arg_MODEL}_{LOSS_FUNC}_{NOISE_TAG}{round(NOISE_LEVEL * 100)}_seed{SEED}"
    DEVICE = [args.device]
    SAVE_MODEL = False

    # C
    if DATASET_TYPE == "kvasir":
        # Read the csv containing the image and mask paths for each example
        full_database = pd.read_csv(
            "/home/guests/jsm/isbi_paper/kvasir_seg_dataset/kvasir_pairs.csv",
            delimiter=",",
        )

        # Split the dataset into training (80%) and 20% for test and val sets
        training_df, temp_df = train_test_split(
            full_database,
            test_size=0.2,  # 20% held out for validation + test
            random_state=DATA_SPLIT_SEED,  # ensures reproducibility
            shuffle=True,
        )
        validation_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,  # half of the 20% → 10% test, 10% validation
            random_state=DATA_SPLIT_SEED,
            shuffle=True,
        )

    # Dataset type for arcade dataset
    elif DATASET_TYPE == "arcade":
        training_df = pd.read_csv(
            "/home/guests/jsm/isbi_paper/arcade_dataset/syntax/train/arcade_train.csv",
            delimiter=",",
        )
        validation_df = pd.read_csv(
            "/home/guests/jsm/isbi_paper/arcade_dataset/syntax/val/arcade_val.csv",
            delimiter=",",
        )
        test_df = pd.read_csv(
            "/home/guests/jsm/isbi_paper/arcade_dataset/syntax/test/arcade_test.csv",
            delimiter=",",
        )

    else:
        raise ValueError("Invalid dataset type. Choose 'kvasir' or 'arcade'.")

    print(
        f"Training samples: {len(training_df)}, Validation samples: {len(validation_df)}, Test samples: {len(test_df)}"
    )

    # Wandb INITIALIZATION
    wandb_logger = None  # Default to None if WANDB is False

    if WANDB:
        print("Activating wandb")
        wandb_key = WANDB_KEY
        wandb.login(key=wandb_key)

    # Define wandb project name
    PROJECT_NAME = "ISBI Recurrent segmentation in the middle of unet"  # + DATASET_TYPE

    # Let Lightning handle the initialization
    wandb_logger = WandbLogger(
        project=PROJECT_NAME,
        name=f"{RUN_NAME}",
        tags=[arg_MODEL],
        offline=False
        if WANDB
        else True,  # If WANDB is true, sends it all to wandb online
        config={
            "model": arg_MODEL,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "max_epochs": MAX_EPOCHS,
            "lr": LEARNING_RATE,
            "loss_function": LOSS_FUNC,
            "noise_type": NOISE_TYPE,
            "noise_level": NOISE_LEVEL,
            "n_iter": N_ITER,
            "deep_supervision": DEEP_SUPERVISION,
            "seed": SEED,
            "dataset_split_seed": DATA_SPLIT_SEED,
        },
    )

    # Build the data module
    data_module = SegmentationDataModule(
        train_df=training_df,
        val_df=validation_df,
        test_df=test_df,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        classes=2,
        dataset_type=DATASET_TYPE,
        noise_level=NOISE_LEVEL,
        noise_type=NOISE_TYPE,
    )

    # Build the training loop
    lit_model = LitSegModel(
        model=model,
        num_classes=2,
        lr=LEARNING_RATE,
        loss_fn=LOSS_FUNC,
        weight_decay=WEIGHT_DECAY,
        wandb_active=WANDB,
        dataset_type=DATASET_TYPE,
        deep_supervision=DEEP_SUPERVISION,
    )

    # Keep the checkpoint with the best target Dice on the validation set
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{RUN_NAME}",
        filename="best-{epoch:02d}-{val_dice_epoch:.4f}",
        monitor="val_dice_epoch",
        mode="max",
        save_top_k=1,
    )

    # Fit the model to the data
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=DEVICE,  # Use GPU
        max_epochs=MAX_EPOCHS,
        log_every_n_steps=25,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(lit_model, datamodule=data_module)

    # Evaluate the best checkpoint on the held-out test set (also loads its weights into lit_model)
    trainer.test(lit_model, datamodule=data_module, ckpt_path="best")

    # Save after training (weights of the best checkpoint)
    if SAVE_MODEL:
        base_path = "/media/general_storage6/jsmstorage/artifact_detection/saved_models"

        # Define the path to save the models
        if DATASET_TYPE == "kvasir":
            save_path = base_path + f"/kvasir_segmentation_models/{RUN_NAME}.pth"

        elif DATASET_TYPE == "arcade":
            save_path = base_path + f"/arcade_segmentation_models/{RUN_NAME}.pth"

        else:
            raise ValueError("Dataset_type not implemented. Choose target or artifact.")

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_type": arg_MODEL,
                "num_classes": 2,
                # SegFormer always uses SEGFORMER_ENCODER (see models.py)
                "encoder_name": (
                    SEGFORMER_ENCODER
                    if arg_MODEL.startswith("segformer")
                    else "efficientnet-b4"
                ),
                "n_iter": N_ITER,
            },
            save_path,
        )

    if WANDB:
        wandb.finish()
