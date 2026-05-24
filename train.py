from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_loader import build_dataloaders
from model import AttentionUNet3D


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = BASE_DIR.parent / "veriler" / "karaciger_3d"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    class_id: int,
    eps: float = 1e-6,
) -> float:
    pred_mask = preds == class_id
    target_mask = targets == class_id
    intersection = (pred_mask & target_mask).sum().item()
    pred_sum = pred_mask.sum().item()
    target_sum = target_mask.sum().item()
    return (2.0 * intersection + eps) / (pred_sum + target_sum + eps)


class TverskyLoss3D(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        class_weights: Tuple[float, float, float] = (1.0, 1.0, 2.0),
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.register_buffer(
            "class_weights", torch.tensor(class_weights, dtype=torch.float32)
        )
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes)
        targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)
        tp = (probs * targets_onehot).sum(dims)
        fp = (probs * (1 - targets_onehot)).sum(dims)
        fn = ((1 - probs) * targets_onehot).sum(dims)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        weights = self.class_weights[:num_classes].to(logits.device)
        loss = 1.0 - (tversky * weights).sum() / weights.sum()
        return loss


def run_epoch(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    optimizer: AdamW | None,
    scaler: GradScaler | None,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float, float]:
    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    num_batches = 0
    liver_sum = 0.0
    tumor_sum = 0.0
    for volumes, masks in loader:
        print("-> Batch isleniyor...", flush=True)
        volumes = volumes.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=use_amp):
            logits = model(volumes)
            loss = criterion(logits, masks)

        if optimizer is not None and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            liver_sum += dice_score(preds, masks, class_id=1)
            tumor_sum += dice_score(preds, masks, class_id=2)

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0, 0.0
    return (
        total_loss / num_batches,
        liver_sum / num_batches,
        tumor_sum / num_batches,
    )


def evaluate_tumor_metrics(
    model: nn.Module, loader: Iterable, device: torch.device
) -> Tuple[float, float, float]:
    model.eval()
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    with torch.no_grad():
        for volumes, masks in loader:
            volumes = volumes.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(volumes)
            preds = torch.argmax(logits, dim=1)

            pred_pos = preds == 2
            true_pos = masks == 2

            tp += (pred_pos & true_pos).sum().item()
            fp += (pred_pos & ~true_pos).sum().item()
            fn += (~pred_pos & true_pos).sum().item()
            tn += (~pred_pos & ~true_pos).sum().item()

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return accuracy, precision, recall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LiTS 3D Attention U-Net Training")
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="Dataset root containing volumes/ and segmentations/",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Checkpoint path to resume training",
    )
    return parser.parse_args()


def resume_training(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: ReduceLROnPlateau,
) -> tuple[int, float]:
    if not checkpoint_path:
        return 1, 0.0

    path = Path(checkpoint_path)
    if not path.exists():
        return 1, 0.0

    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint.get("model_state_dict", {}))
    optimizer.load_state_dict(checkpoint.get("optimizer_state_dict", {}))
    scheduler.load_state_dict(checkpoint.get("scheduler_state_dict", {}))
    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_tumor_dice = float(checkpoint.get("best_tumor_dice", 0.0))
    return start_epoch, best_tumor_dice


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    train_loader, val_loader = build_dataloaders(
        data_root=Path(args.data_root),
        batch_size=args.batch_size,
        num_workers=0,
    )

    model = AttentionUNet3D(
        in_channels=1,
        num_classes=3,
        base_channels=args.base_channels,
        dropout=args.dropout,
        use_residual=True,
    ).to(device)

    criterion = TverskyLoss3D(alpha=0.3, beta=0.7, class_weights=(1.0, 1.0, 2.0))
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    scaler = GradScaler("cuda", enabled=use_amp)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_liver_dice": [],
        "val_liver_dice": [],
        "train_tumor_dice": [],
        "val_tumor_dice": [],
    }

    start_epoch, best_tumor_dice = resume_training(
        args.resume,
        model,
        optimizer,
        scheduler,
    )
    if args.resume and Path(args.resume).exists():
        print(
            "Resume: "
            f"checkpoint={args.resume} | "
            f"start_epoch={start_epoch} | "
            f"best_tumor_dice={best_tumor_dice:.4f}"
        )
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_liver_dice, train_tumor_dice = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp,
        )
        val_loss, val_liver_dice, val_tumor_dice = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            scaler=None,
            device=device,
            use_amp=use_amp,
        )
        scheduler.step(val_loss)
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"train_liver_dice={train_liver_dice:.4f} | "
            f"val_liver_dice={val_liver_dice:.4f} | "
            f"train_tumor_dice={train_tumor_dice:.4f} | "
            f"val_tumor_dice={val_tumor_dice:.4f}"
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_liver_dice"].append(train_liver_dice)
        history["val_liver_dice"].append(val_liver_dice)
        history["train_tumor_dice"].append(train_tumor_dice)
        history["val_tumor_dice"].append(val_tumor_dice)

        if val_tumor_dice > best_tumor_dice:
            best_tumor_dice = val_tumor_dice
            torch.save(
                {
                    "epoch": epoch,
                    "best_tumor_dice": best_tumor_dice,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                "best_model.pth",
            )

    tumor_acc, tumor_prec, tumor_rec = evaluate_tumor_metrics(
        model, val_loader, device
    )
    print(
        "Validation tumor metrics | "
        f"accuracy={tumor_acc:.4f} | "
        f"precision={tumor_prec:.4f} | "
        f"recall={tumor_rec:.4f}"
    )

    output_dir = Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics_report.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "train_liver_dice",
                "val_liver_dice",
                "train_tumor_dice",
                "val_tumor_dice",
            ]
        )
        for i in range(len(history["epoch"])):
            writer.writerow(
                [
                    history["epoch"][i],
                    history["train_loss"][i],
                    history["val_loss"][i],
                    history["train_liver_dice"][i],
                    history["val_liver_dice"][i],
                    history["train_tumor_dice"][i],
                    history["val_tumor_dice"][i],
                ]
            )
        writer.writerow([])
        writer.writerow(["summary", "", "", "", "", "", ""])
        writer.writerow(["tumor_accuracy", tumor_acc])
        writer.writerow(["tumor_precision", tumor_prec])
        writer.writerow(["tumor_recall", tumor_rec])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(
        history["epoch"], history["train_liver_dice"], label="Train Liver Dice"
    )
    axes[1].plot(
        history["epoch"], history["val_liver_dice"], label="Val Liver Dice"
    )
    axes[1].plot(
        history["epoch"], history["train_tumor_dice"], label="Train Tumor Dice"
    )
    axes[1].plot(
        history["epoch"], history["val_tumor_dice"], label="Val Tumor Dice"
    )
    axes[1].set_title("Dice Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_plots.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
