
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ==================================================
# 1. Configuration
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_DIR / "ml" / "checkpoints" / "convnext_tiny_best.pth"
)

# Updated input: original glioma test image
DEFAULT_IMAGE_PATH = (
    PROJECT_DIR / "data" / "Testing" / "glioma" / "Te-gl_1.jpg"
)

OUTPUT_DIR = PROJECT_DIR / "ml" / "results" / "gradcam"

DEVICE = torch.device("cpu")


# ==================================================
# 2. Load the trained ConvNeXt-Tiny checkpoint
# ==================================================

def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}\n\n"
            "Check available files with:\n"
            "Get-ChildItem .\\ml\\checkpoints"
        )

    # Only load checkpoints you trust, since pickle-based
    # checkpoints can execute code when deserialized.
    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not load the checkpoint with weights_only=True. "
            "Check that it contains only supported tensors and "
            "basic metadata, and verify the checkpoint source."
        ) from exc

    class_names = checkpoint["class_names"]
    image_size = int(checkpoint.get("image_size", 160))

    model = models.convnext_tiny(weights=None)

    model.classifier[2] = nn.Linear(
        model.classifier[2].in_features,
        len(class_names),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names, image_size


# ==================================================
# 3. Generate Grad-CAM
# ==================================================

def generate_gradcam(image_path: Path):
    if not image_path.exists():
        raise FileNotFoundError(
            f"Input MRI image not found:\n{image_path}"
        )

    model, class_names, image_size = load_model()

    # Load original MRI image
    original_image = Image.open(image_path).convert("RGB")

    # Resize for heatmap overlay
    resized_image = original_image.resize(
        (image_size, image_size)
    )

    rgb_image = (
        np.asarray(resized_image, dtype=np.float32) / 255.0
    )

    # Same ImageNet normalization used during training
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    input_tensor = preprocess(original_image).unsqueeze(0)
    input_tensor = input_tensor.to(DEVICE)

    # Predict class. Grad-CAM itself must run with gradients enabled.
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_idx = int(probabilities.argmax().item())

    predicted_class = class_names[predicted_idx]
    confidence = float(probabilities[predicted_idx].item() * 100)

    # Ask Grad-CAM to explain the predicted class
    targets = [ClassifierOutputTarget(predicted_idx)]

    # Last block of the final ConvNeXt feature stage
    target_layers = [model.features[-1][-1]]

    with GradCAM(
        model=model,
        target_layers=target_layers,
    ) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets,
        )[0]

    # Combine MRI and heatmap
    cam_image = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True,
    )

    # Save heatmap overlay
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / (
        f"{image_path.stem}_gradcam.jpg"
    )

    Image.fromarray(cam_image).save(output_path)

    return predicted_class, confidence, output_path


# ==================================================
# 4. Main
# ==================================================

def main():
    print("=" * 55)
    print("Brain Tumor AI - Grad-CAM Visualization")
    print("=" * 55)

    print(f"Checkpoint : {CHECKPOINT_PATH}")
    print(f"Input MRI  : {DEFAULT_IMAGE_PATH}")

    predicted_class, confidence, output_path = generate_gradcam(
        DEFAULT_IMAGE_PATH
    )

    print("\nPrediction completed")
    print(f"Actual class     : glioma")
    print(f"Predicted class  : {predicted_class}")
    print(f"Confidence       : {confidence:.2f}%")
    print(f"Grad-CAM output  : {output_path}")

    if predicted_class.lower() == "glioma":
        print("Result           : Correct class prediction")
    else:
        print("Result           : Incorrect class prediction")

    print("\nNote: Grad-CAM highlights regions contributing to")
    print("the model prediction; it does not prove medical accuracy.")
    print("=" * 55)


if __name__ == "__main__":
    main()
