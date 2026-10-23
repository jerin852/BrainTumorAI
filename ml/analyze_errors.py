
from pathlib import Path
import csv

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

from train import build_model

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "data" / "Testing"
CHECKPOINT = ROOT / "ml" / "checkpoints" / "convnext_tiny_best.pth"
OUTPUT_DIR = ROOT / "ml" / "results" / "convnext_tiny_errors"
CSV_PATH = ROOT / "ml" / "results" / "convnext_tiny_errors.csv"

BATCH_SIZE = 8
DEVICE = torch.device("cpu")


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    checkpoint = torch.load(
        CHECKPOINT, map_location=DEVICE, weights_only=True
    )
    classes = checkpoint["class_names"]
    image_size = int(checkpoint["image_size"])

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ])

    dataset = datasets.ImageFolder(TEST_DIR, transform=transform)

    if dataset.classes != classes:
        raise ValueError(
            f"Class mismatch: checkpoint={classes}, dataset={dataset.classes}"
        )

    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = build_model("convnext_tiny", len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    total_correct = 0

    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(DEVICE))
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            for i, (actual_idx, pred_idx) in enumerate(
                zip(labels.tolist(), predictions.tolist())
            ):
                image_path, _ = dataset.samples[
                    len(rows) + i if False else 0
                ] if False else ("", 0)

            # Paths for this batch are obtained from the dataset's fixed order.
            start = len(rows)
            batch_paths = [
                dataset.samples[j][0]
                for j in range(start, start + len(labels))
            ]

            for image_path, actual_idx, pred_idx, scores in zip(
                batch_paths,
                labels.tolist(),
                predictions.tolist(),
                probabilities.tolist(),
            ):
                actual = classes[actual_idx]
                predicted = classes[pred_idx]
                confidence = float(scores[pred_idx])
                correct = actual_idx == pred_idx

                if correct:
                    total_correct += 1
                else:
                    class_dir = OUTPUT_DIR / actual
                    class_dir.mkdir(parents=True, exist_ok=True)

                    source = Path(image_path)
                    destination = class_dir / (
                        f"pred_{predicted}__{source.name}"
                    )
                    if not destination.exists():
                        with Image.open(source) as image:
                            image.convert("RGB").save(destination)

                rows.append({
                    "image_path": image_path,
                    "actual_class": actual,
                    "predicted_class": predicted,
                    "correct": correct,
                    "predicted_class_score": round(confidence, 6),
                })

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    print(f"Total test images: {total}")
    print(f"Correct predictions: {total_correct}")
    print(f"Wrong predictions: {total - total_correct}")
    print(f"Accuracy: {total_correct / total:.4%}")
    print(f"Misclassified images: {OUTPUT_DIR}")
    print(f"Prediction report: {CSV_PATH}")


if __name__ == "__main__":
    main()
