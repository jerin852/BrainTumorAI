
import base64
import io
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    ROOT / "ml" / "checkpoints" / "convnext_tiny_best.pth"
)

EVALUATION_FILE = (
    ROOT / "ml" / "results" / "test_evaluation_summary.json"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BrainTumorAI API",
    description="Brain MRI classification with Grad-CAM visualization",
    version="2.0.0",
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

model = None
target_layer = None
image_size = 160

preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

def load_model():
    global model, target_layer, image_size, preprocess

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
        )
        saved_classes = checkpoint.get("class_names")
        saved_image_size = checkpoint.get("image_size", 160)
    else:
        state_dict = checkpoint
        saved_classes = None
        saved_image_size = 160

    if state_dict is None:
        raise ValueError(
            "Checkpoint does not contain model_state_dict or state_dict."
        )

    if saved_classes:
        CLASS_NAMES.clear()
        CLASS_NAMES.extend(saved_classes)

    image_size = int(saved_image_size)

    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Build the same ConvNeXt-Tiny architecture used for training.
    net = models.convnext_tiny(weights=None)

    input_features = net.classifier[2].in_features
    net.classifier[2] = torch.nn.Linear(
        input_features,
        len(CLASS_NAMES),
    )

    # Handle checkpoints saved from DataParallel.
    cleaned_state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

    net.load_state_dict(cleaned_state_dict, strict=True)

    net.to(DEVICE)
    net.eval()

    model = net

    # Final ConvNeXt feature block for Grad-CAM.
    target_layer = model.features[-1][-1]

    logger.info("Model loaded: ConvNeXt-Tiny")
    logger.info("Checkpoint: %s", CHECKPOINT)
    logger.info("Device: %s", DEVICE)
    logger.info("Classes: %s", CLASS_NAMES)
    logger.info("Image size: %s", image_size)


@app.on_event("startup")
async def startup_event():
    try:
        load_model()
    except Exception:
        logger.exception("Model loading failed")


# --------------------------------------------------
# GRAD-CAM: JET COLOUR MAP
# --------------------------------------------------

def generate_gradcam(
    input_tensor,
    original_image,
    class_index,
):
    """
    Generate a Jet-colormap Grad-CAM overlay.

    Blue -> Cyan -> Green -> Yellow -> Red
    Lower activation -> Higher activation

    Returns a base64 PNG data URL.
    """

    if model is None or target_layer is None:
        raise RuntimeError("Model is not loaded.")

    activations = []
    gradients = []

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    try:
        model.eval()
        model.zero_grad(set_to_none=True)

        # Grad-CAM requires gradients.
        with torch.enable_grad():
            x = input_tensor.detach().clone().to(DEVICE)
            x.requires_grad_(True)

            logits = model(x)
            score = logits[0, class_index]
            score.backward()

        if not activations or not gradients:
            raise RuntimeError(
                "Grad-CAM hooks did not capture model features."
            )

        feature_maps = activations[0]
        grads = gradients[0]

        # Average gradients spatially to obtain channel weights.
        weights = grads.mean(
            dim=(2, 3),
            keepdim=True,
        )

        # Weighted sum of feature maps.
        cam = (weights * feature_maps).sum(
            dim=1,
            keepdim=True,
        )

        cam = F.relu(cam)

        width, height = original_image.size

        cam = F.interpolate(
            cam,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0].detach().cpu().numpy()

        # Normalize activation values to 0..1.
        cam_min = float(cam.min())
        cam_max = float(cam.max())

        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)

        heatmap_uint8 = np.uint8(
            np.clip(cam * 255.0, 0, 255)
        )

        # OpenCV JET colour map:
        # blue -> cyan -> green -> yellow -> red.
        heatmap_bgr = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET,
        )

        # Original MRI image in BGR format.
        original_rgb = np.array(
            original_image.convert("RGB")
        )

        original_bgr = cv2.cvtColor(
            original_rgb,
            cv2.COLOR_RGB2BGR,
        )

        # Blend: 60% original MRI + 40% heatmap.
        overlay_bgr = cv2.addWeighted(
            original_bgr,
            0.60,
            heatmap_bgr,
            0.40,
            0,
        )

        overlay_rgb = cv2.cvtColor(
            overlay_bgr,
            cv2.COLOR_BGR2RGB,
        )

        output_image = Image.fromarray(overlay_rgb)

        buffer = io.BytesIO()
        output_image.save(buffer, format="PNG")

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        return f"data:image/png;base64,{encoded}"

    finally:
        forward_handle.remove()
        backward_handle.remove()
        model.zero_grad(set_to_none=True)


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "BrainTumorAI API is running",
        "docs": "/docs",
        "health": "/health",
        "evaluation": "/evaluation",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "model_name": "convnext_tiny",
        "device": str(DEVICE),
        "checkpoint_exists": CHECKPOINT.exists(),
        "classes": CLASS_NAMES,
    }


# --------------------------------------------------
# MRI PREDICTION
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check the backend logs.",
        )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Please upload an MRI image.",
        )

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        original_image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image. Upload a valid JPG or PNG file.",
        )

    try:
        input_tensor = preprocess(original_image).unsqueeze(0)
        input_tensor = input_tensor.to(DEVICE)

        # Run prediction without building a gradient graph.
        with torch.inference_mode():
            logits = model(input_tensor)
            probabilities = torch.softmax(
                logits,
                dim=1,
            )[0]

            predicted_index = int(
                torch.argmax(probabilities).item()
            )

            confidence = float(
                probabilities[predicted_index].item() * 100
            )

        predicted_class = CLASS_NAMES[predicted_index]

        # Grad-CAM is calculated separately with gradients enabled.
        gradcam_image = generate_gradcam(
            input_tensor=input_tensor,
            original_image=original_image,
            class_index=predicted_index,
        )

        probability_dict = {
            CLASS_NAMES[i]: round(
                float(probabilities[i].item()) * 100,
                2,
            )
            for i in range(len(CLASS_NAMES))
        }

        return {
            "success": True,
            "filename": file.filename,
            "model": "convnext_tiny",
            "device": str(DEVICE),
            "predicted_class": predicted_class,
            "confidence_percent": round(confidence, 2),
            "probabilities": probability_dict,
            "gradcam_image": gradcam_image,
            "disclaimer": (
                "Research demonstration only. "
                "This AI prediction is not a medical diagnosis."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )


# --------------------------------------------------
# MODEL EVALUATION METRICS
# --------------------------------------------------

def read_evaluation_results():
    if not EVALUATION_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Evaluation file not found. Run ml/evaluate.py first."
            ),
        )

    try:
        with EVALUATION_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Evaluation JSON file is invalid.",
        )

    # Support either a list of model results or one result object.
    if isinstance(data, dict):
        if isinstance(data.get("models"), list):
            results = data["models"]
        else:
            results = [data]
    elif isinstance(data, list):
        results = data
    else:
        raise HTTPException(
            status_code=500,
            detail="Unexpected evaluation JSON structure.",
        )

    return [
        item for item in results
        if isinstance(item, dict)
    ]


@app.get("/evaluation")
def evaluation():
    results = read_evaluation_results()

    if not results:
        return {
            "success": True,
            "models_count": 0,
            "best_model": None,
            "models": [],
        }

    def accuracy_value(item):
        value = item.get("test_accuracy", 0)

        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0

        # Accept either fraction (0.95) or percentage (95).
        return value / 100 if value > 1 else value

    best_model = max(
        results,
        key=accuracy_value,
    )

    return {
        "success": True,
        "test_dataset": "data/Testing",
        "models_count": len(results),
        "best_model": best_model.get(
            "model_name",
            best_model.get("model", "unknown"),
        ),
        "models": results,
    }


@app.get("/evaluation/{model_name}")
def evaluation_by_model(model_name: str):
    results = read_evaluation_results()

    for item in results:
        saved_name = item.get(
            "model_name",
            item.get("model", ""),
        )

        if str(saved_name).lower() == model_name.lower():
            return {
                "success": True,
                "model": item,
            }

    raise HTTPException(
        status_code=404,
        detail=f"No evaluation found for model '{model_name}'.",
    )
