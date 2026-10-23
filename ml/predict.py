from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from train import build_model

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "ml" / "checkpoints"

def main():
    parser = argparse.ArgumentParser(description="Predict the class of one brain MRI image.")
    parser.add_argument("--model", choices=("densenet121", "convnext_tiny", "efficientnetv2_s"), required=True)
    parser.add_argument("--image", required=True, help="Path to an MRI image.")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    checkpoint_path = CHECKPOINT_DIR / f"{args.model}_best.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train this model first.")

    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    classes = checkpoint["class_names"]
    image_size = int(checkpoint["image_size"])
    model = build_model(args.model, len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with Image.open(image_path) as im:
        tensor = tf(im.convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        probs = F.softmax(model(tensor), dim=1)[0]
    best_idx = int(probs.argmax())
    print(f"Predicted class: {classes[best_idx]}")
    print(f"Model score (not a calibrated clinical probability): {float(probs[best_idx]):.4f}")
    print("All class scores:")
    for name, score in sorted(zip(classes, probs.tolist()), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {score:.4f}")

if __name__ == "__main__":
    main()
