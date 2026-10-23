
from pathlib import Path
import base64
import io

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import numpy as np


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# IMPORTANT:
# Change this to your existing trained checkpoint's actual path.
CHECKPOINT = PROJECT_ROOT / "ml" / "models" / "convnext_tiny_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title="BrainTumorAI API",
    description="Brain MRI classification with Grad-CAM visualization",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 2. GLOBAL MODEL VARIABLES
# ============================================================

model = None
class_names = []
image_size = 160
preprocess = None
target_layers = None


# ============================================================
# 3. LOAD TRAINED CONVNEXT-TINY CHECKPOINT
# ============================================================

@app.on_event("startup")
def load_checkpoint():
    global model, class_names, image_size, preprocess, target_layers

    if not CHECKPOINT.is_file():
        raise RuntimeError(
            f"Checkpoint not found: {CHECKPOINT}. "
            "Update CHECKPOINT to the actual trained model path."
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    required_keys = {"class_names", "model_state_dict"}
    missing_keys = required_keys - set(checkpoint.keys())

    if missing_keys:
        raise RuntimeError(
            f"Checkpoint is missing required keys: {sorted(missing_keys)}"
        )

    class_names = checkpoint["class_names"]
    image_size = int(checkpoint.get("image_size", 160))

    if not isinstance(class_names, (list, tuple)) or not class_names:
        raise RuntimeError("Checkpoint class_names is invalid.")

    # Re-create the same architecture used during training.
    loaded_model = models.convnext_tiny(weights=None)

    loaded_model.classifier[2] = nn.Linear(
        loaded_model.classifier[2].in_features,
        len(class_names),
    )

    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.to(DEVICE)
    loaded_model.eval()

    model = loaded_model

    # Final ConvNeXt feature block for Grad-CAM.
    target_layers = [model.features[-1][-1]]

    # Must match the preprocessing used during model training.
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    print(f"Model loaded successfully on {DEVICE}")
    print(f"Classes: {class_names}")
    print(f"Image size: {image_size}")


# ============================================================
# 4. ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "BrainTumorAI API is running",
        "model": "ConvNeXt-Tiny",
        "device": str(DEVICE),
    }


# ============================================================
# 5. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "device": str(DEVICE),
    }


# ============================================================
# 6. PREDICTION + GRAD-CAM
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if model is None or preprocess is None or target_layers is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready. Check backend startup logs.",
        )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Please upload an image file.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # --------------------------------------------------------
    # 6A. READ AND PREPROCESS MRI
    # --------------------------------------------------------

    try:
        with Image.open(io.BytesIO(contents)) as source_image:
            rgb_pil = source_image.convert("RGB")

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or unreadable image.",
        )

    resized_image = rgb_pil.resize(
        (image_size, image_size),
        Image.Resampling.BILINEAR,
    )

    rgb_image = np.asarray(
        resized_image,
        dtype=np.float32,
    ) / 255.0

    input_tensor = preprocess(rgb_pil).unsqueeze(0).to(DEVICE)

    # --------------------------------------------------------
    # 6B. PREDICT CLASS
    # --------------------------------------------------------

    try:
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        predicted_idx = int(probabilities.argmax().item())
        predicted_class = class_names[predicted_idx]

        confidence = float(
            probabilities[predicted_idx].item() * 100
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(exc).__name__}",
        ) from exc

    # --------------------------------------------------------
    # 6C. GENERATE GRAD-CAM
    # --------------------------------------------------------

    try:
        targets = [ClassifierOutputTarget(predicted_idx)]

        with GradCAM(
            model=model,
            target_layers=target_layers,
        ) as cam:
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=targets,
            )[0]

        overlay = show_cam_on_image(
            rgb_image,
            grayscale_cam,
            use_rgb=True,
        )

        buffer = io.BytesIO()

        Image.fromarray(overlay).save(
            buffer,
            format="JPEG",
            quality=90,
        )

        heatmap_base64 = base64.b64encode(
            buffer.getvalue()
        ).decode("ascii")

        gradcam_data_url = (
            f"data:image/jpeg;base64,{heatmap_base64}"
        )

    except Exception as exc:
        # Keep the error in the backend log for debugging.
        import logging
        logging.exception("Grad-CAM generation failed")

        raise HTTPException(
            status_code=500,
            detail=f"Grad-CAM generation failed: {type(exc).__name__}",
        ) from exc

    # --------------------------------------------------------
    # 6D. RETURN STRUCTURED RESPONSE
    # --------------------------------------------------------

    return {
        "filename": file.filename,
        "predicted_class": predicted_class,
        "confidence_percent": round(confidence, 2),
        "probabilities": {
            name: round(float(probabilities[i].item() * 100), 2)
            for i, name in enumerate(class_names)
        },
        "gradcam_image": gradcam_data_url,
        "model": "ConvNeXt-Tiny",
        "disclaimer": (
            "Research demonstration only. "
            "This output is not a medical diagnosis."
        ),
    }


# ============================================================
# 7. RUN WITH UVICORN
# ============================================================

# Start from PowerShell using:
# python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
