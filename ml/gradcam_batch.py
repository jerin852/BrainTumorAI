
from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "ml" / "checkpoints" / "convnext_tiny_best.pth"
TEST_DIR = ROOT / "data" / "Testing"
OUTPUT_DIR = ROOT / "ml" / "results" / "gradcam_batch"
REPORT = ROOT / "ml" / "results" / "gradcam_batch_report.csv"

DEVICE = torch.device("cpu")
MAX_IMAGES_PER_CLASS = 10
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    classes = checkpoint["class_names"]
    image_size = int(checkpoint.get("image_size", 160))

    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(
        model.classifier[2].in_features,
        len(classes),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()

    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return model, classes, image_size, preprocess


def main():
    model, classes, image_size, preprocess = load_model()

    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    target_layers = [model.features[-1][-1]]
    rows = []

    for actual_class in classes:
        class_dir = TEST_DIR / actual_class

        if not class_dir.is_dir():
            print(f"Skipping missing folder: {class_dir}")
            continue

        images = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )[:MAX_IMAGES_PER_CLASS]

        print(f"\nProcessing {actual_class}: {len(images)} images")

        for image_path in images:
            try:
                pil_image = Image.open(image_path).convert("RGB")
                resized = pil_image.resize((image_size, image_size))
                rgb = np.asarray(resized, dtype=np.float32) / 255.0

                tensor = preprocess(pil_image).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    predicted_idx = int(probs.argmax().item())

                predicted_class = classes[predicted_idx]
                confidence = float(probs[predicted_idx].item() * 100)
                correct = predicted_class == actual_class

                targets = [ClassifierOutputTarget(predicted_idx)]

                with GradCAM(
                    model=model,
                    target_layers=target_layers,
                ) as cam:
                    heatmap = cam(
                        input_tensor=tensor,
                        targets=targets,
                    )[0]

                overlay = show_cam_on_image(
                    rgb,
                    heatmap,
                    use_rgb=True,
                )

                output_path = (
                    OUTPUT_DIR
                    / actual_class
                    / f"{image_path.stem}_gradcam.jpg"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(overlay).save(output_path)

                rows.append({
                    "image": str(image_path),
                    "actual_class": actual_class,
                    "predicted_class": predicted_class,
                    "confidence_percent": round(confidence, 2),
                    "correct": correct,
                    "gradcam_output": str(output_path),
                })

                print(
                    f"  {image_path.name}: "
                    f"{predicted_class} ({confidence:.2f}%) "
                    f"{'CORRECT' if correct else 'WRONG'}"
                )

            except Exception as exc:
                print(f"  ERROR processing {image_path.name}: {exc}")

    with REPORT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "actual_class",
                "predicted_class",
                "confidence_percent",
                "correct",
                "gradcam_output",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        correct_count = sum(row["correct"] for row in rows)
        print("\n" + "=" * 55)
        print(f"Images processed : {len(rows)}")
        print(f"Correct          : {correct_count}")
        print(f"Incorrect        : {len(rows) - correct_count}")
        print(f"Accuracy         : {100 * correct_count / len(rows):.2f}%")
        print(f"Grad-CAM folder  : {OUTPUT_DIR}")
        print(f"CSV report       : {REPORT}")
        print("=" * 55)
    else:
        print("No images processed. Check dataset folders and checkpoint classes.")


if __name__ == "__main__":
    main()

