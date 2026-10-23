from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "ml" / "checkpoints"
RESULTS_DIR = ROOT / "ml" / "results"

MODEL_NAMES = ("densenet121", "convnext_tiny", "efficientnetv2_s")
EXPECTED_CLASSES = {"glioma", "meningioma", "notumor", "pituitary"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(name: str, num_classes: int, pretrained: bool):
    """Build a torchvision classifier with a four-class output layer."""
    weight_sets = {
        "densenet121": models.DenseNet121_Weights.DEFAULT,
        "convnext_tiny": models.ConvNeXt_Tiny_Weights.DEFAULT,
        "efficientnetv2_s": models.EfficientNet_V2_S_Weights.DEFAULT,
    }
    weights = weight_sets[name] if pretrained else None

    if name == "densenet121":
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == "convnext_tiny":
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif name == "efficientnetv2_s":
        model = models.efficientnet_v2_s(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model


def get_transforms(image_size: int):
    # ImageNet normalization is appropriate when using ImageNet-pretrained weights.
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.08, contrast=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_transform, eval_transform


def make_loaders(image_size: int, batch_size: int, val_fraction: float, seed: int, workers: int):
    train_dir = DATA_DIR / "Training"
    test_dir = DATA_DIR / "Testing"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"Expected dataset folders:\n  {train_dir}\n  {test_dir}"
        )

    # Two copies allow training augmentation and deterministic validation transforms.
    train_tf, eval_tf = get_transforms(image_size)
    train_full = datasets.ImageFolder(train_dir, transform=train_tf)
    val_full = datasets.ImageFolder(train_dir, transform=eval_tf)
    test_set = datasets.ImageFolder(test_dir, transform=eval_tf)

    classes = train_full.classes
    if set(classes) != EXPECTED_CLASSES:
        raise ValueError(
            f"Unexpected class folders: {classes}. Expected these four names: "
            f"{sorted(EXPECTED_CLASSES)}"
        )
    if test_set.classes != classes:
        raise ValueError(f"Training class order {classes} != Testing class order {test_set.classes}")

    targets = np.asarray(train_full.targets)
    indices = np.arange(len(targets))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_fraction, random_state=seed, stratify=targets
    )
    train_set = Subset(train_full, train_idx.tolist())
    val_set = Subset(val_full, val_idx.tolist())

    # Windows-safe default: num_workers=0. User can increase later if needed.
    loader_args = dict(batch_size=batch_size, num_workers=workers, pin_memory=False)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)
    return train_loader, val_loader, test_loader, classes, len(train_idx), len(val_idx), len(test_set)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_items = 0.0, 0
    all_y, all_pred = [], []

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_n = labels.size(0)
            total_loss += loss.item() * batch_n
            total_items += batch_n
            preds = outputs.argmax(dim=1)
            all_y.extend(labels.detach().cpu().tolist())
            all_pred.extend(preds.detach().cpu().tolist())

    avg_loss = total_loss / max(total_items, 1)
    acc = accuracy_score(all_y, all_pred)
    return avg_loss, acc, all_y, all_pred


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train a selectable brain MRI classifier.")
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Do not download/use ImageNet pretrained weights.")
    parser.add_argument("--test-after-training", action="store_true",
                        help="Run final test evaluation after validation-based checkpoint selection.")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 64:
        parser.error("epochs/batch-size must be >= 1 and image-size must be >= 64.")
    if not 0 < args.val_fraction < 0.5:
        parser.error("--val-fraction must be between 0 and 0.5.")

    seed_everything(args.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device("cpu")  # This project environment is CPU-only.
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, classes, n_train, n_val, n_test = make_loaders(
        args.image_size, args.batch_size, args.val_fraction, args.seed, args.workers
    )
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Classes (index order): {classes}")
    print(f"Split sizes: train={n_train}, validation={n_val}, untouched test={n_test}")
    print(f"Pretrained weights: {not args.no_pretrained}")
    if not args.no_pretrained:
        print("If first-time pretrained-weight download fails, rerun with --no-pretrained (accuracy may differ).")

    model = build_model(args.model, len(classes), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    checkpoint_path = CHECKPOINT_DIR / f"{args.model}_best.pth"
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc, _, _ = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_acc, 6),
            "validation_loss": round(val_loss, 6),
            "validation_accuracy": round(val_acc, 6),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": round(time.time() - epoch_start, 2),
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.4f} | "
            f"{row['epoch_seconds']:.1f}s"
        )

        # Select checkpoint using validation loss only, never the test set.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "model_name": args.model,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "class_names": classes,
                "image_size": args.image_size,
                "epoch": epoch,
                "validation_loss": val_loss,
                "validation_accuracy": val_acc,
                "seed": args.seed,
            }, checkpoint_path)
            print(f"  Saved best checkpoint: {checkpoint_path}")
        else:
            stale_epochs += 1

        if stale_epochs >= args.patience:
            print(f"Early stopping: validation loss did not improve for {args.patience} epochs.")
            break

    history_path = RESULTS_DIR / f"{args.model}_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "model": args.model,
        "class_names": classes,
        "train_images": n_train,
        "validation_images": n_val,
        "test_images": n_test,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "checkpoint": str(checkpoint_path),
        "history_csv": str(history_path),
        "total_training_seconds": round(time.time() - start_time, 2),
        "test_evaluated": False,
    }

    if args.test_after_training:
        print("\nFinal test evaluation (run only after choosing the model using validation results).")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        _, test_acc, y_true, y_pred = run_epoch(model, test_loader, criterion, device)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(classes))), zero_division=0
        )
        report = classification_report(
            y_true, y_pred, labels=list(range(len(classes))),
            target_names=classes, output_dict=True, zero_division=0
        )
        matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
        test_summary = {
            "model": args.model,
            "test_accuracy": float(test_acc),
            "macro_precision": float(np.mean(precision)),
            "macro_recall": float(np.mean(recall)),
            "macro_f1": float(np.mean(f1)),
            "class_metrics": {
                name: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                } for i, name in enumerate(classes)
            },
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
        }
        save_json(RESULTS_DIR / f"{args.model}_test_metrics.json", test_summary)
        summary["test_evaluated"] = True
        summary["test_accuracy"] = float(test_acc)
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Macro precision: {test_summary['macro_precision']:.4f}")
        print(f"Macro recall: {test_summary['macro_recall']:.4f}")
        print(f"Macro F1: {test_summary['macro_f1']:.4f}")
        print("Confusion matrix (rows=true class, columns=predicted class):")
        print(matrix)
        print(f"Saved metrics: {RESULTS_DIR / f'{args.model}_test_metrics.json'}")

    save_json(RESULTS_DIR / f"{args.model}_summary.json", summary)
    print("\nTraining finished.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")
    print(f"Summary: {RESULTS_DIR / f'{args.model}_summary.json'}")


if __name__ == "__main__":
    import os
    main()
