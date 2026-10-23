
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from train import build_model

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "data" / "Testing"
CHECKPOINT_DIR = ROOT / "ml" / "checkpoints"
RESULTS_DIR = ROOT / "ml" / "results"

MODELS = ("densenet121", "convnext_tiny", "efficientnetv2_s")
BATCH_SIZE = 8
DEVICE = torch.device("cpu")


def evaluate_model(model_name: str) -> dict:
    checkpoint_path = CHECKPOINT_DIR / f"{model_name}_best.pth"

    if not checkpoint_path.is_file():
        print(f"\nSkipping {model_name}: checkpoint not found.")
        return {}

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    class_names = checkpoint["class_names"]
    image_size = int(checkpoint["image_size"])

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ])

    dataset = datasets.ImageFolder(
        root=str(TEST_DIR),
        transform=transform,
    )

    if dataset.classes != class_names:
        raise ValueError(
            f"Class mismatch for {model_name}!\n"
            f"Checkpoint classes: {class_names}\n"
            f"Test folder classes: {dataset.classes}"
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(
        model_name,
        num_classes=len(class_names),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []

    print(f"\nEvaluating {model_name}...")
    print(f"Test images: {len(dataset)}")

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.tolist())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    print(f"\n{'=' * 55}")
    print(f"MODEL: {model_name}")
    print(f"{'=' * 55}")
    print(f"Test Accuracy : {accuracy:.4%}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall   : {recall:.4f}")
    print(f"Macro F1-score : {macro_f1:.4f}")

    print("\nConfusion Matrix")
    print("Rows = Actual class | Columns = Predicted class")
    print("Class order:", class_names)
    print(cm)

    print("\nClassification Report")
    print(classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        digits=4,
    ))

    return {
        "model": model_name,
        "test_images": len(dataset),
        "test_accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "checkpoint": str(checkpoint_path),
    }


def main():
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(
            f"Test dataset folder not found: {TEST_DIR}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for model_name in MODELS:
        result = evaluate_model(model_name)
        if result:
            results.append(result)

    if not results:
        raise RuntimeError("No trained model checkpoints were found.")

    summary_path = RESULTS_DIR / "test_evaluation_summary.json"
    summary_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("FINAL TEST-SET COMPARISON")
    print("=" * 65)
    print(
        f"{'Model':<20} {'Accuracy':>12} "
        f"{'Macro F1':>12} {'Images':>10}"
    )

    for result in sorted(
        results,
        key=lambda item: item["test_accuracy"],
        reverse=True,
    ):
        print(
            f"{result['model']:<20} "
            f"{result['test_accuracy']:>11.2%} "
            f"{result['macro_f1']:>12.4f} "
            f"{result['test_images']:>10}"
        )

    print(f"\nDetailed results saved to:\n{summary_path}")


if __name__ == "__main__":
    main()
