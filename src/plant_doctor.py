
import json
import time
from pathlib import Path
from functools import lru_cache

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Updated 15-class model
MODEL_PATH = ROOT / "models" / "plant_disease_model_15class.keras"
LABELS_PATH = ROOT / "models" / "labels_15class.json"

IMAGE_SIZE = 160


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        return None

    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)


def _load_labels():
    if not LABELS_PATH.exists():
        return None

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_crop(label):
    if label.startswith("Tomato"):
        return "Tomato"
    elif label.startswith("Potato"):
        return "Potato"
    elif label.startswith("Pepper"):
        return "Pepper"
    return "Unknown"


def _format_disease(label):
    label = label.replace("___", "_").replace("__", "_")
    label = label.replace("_", " ")

    replacements = {
        "Pepper bell Bacterial spot": "Bacterial Spot",
        "Pepper bell healthy": "Healthy",
        "Potato Early blight": "Early Blight",
        "Potato Late blight": "Late Blight",
        "Potato healthy": "Healthy",
        "Tomato Bacterial spot": "Bacterial Spot",
        "Tomato Early blight": "Early Blight",
        "Tomato healthy": "Healthy",
        "Tomato Late blight": "Late Blight",
        "Tomato Leaf Mold": "Leaf Mold",
        "Tomato Septoria leaf spot": "Septoria Leaf Spot",
        "Tomato Spider mites Two spotted spider mite": "Spider Mites",
        "Tomato Target Spot": "Target Spot",
        "Tomato Tomato mosaic virus": "Mosaic Virus",
        "Tomato Tomato YellowLeaf Curl Virus": "Yellow Leaf Curl Virus",
    }

    return replacements.get(label, label)


def model_status():
    ready = MODEL_PATH.exists() and LABELS_PATH.exists()

    size_mb = (
        MODEL_PATH.stat().st_size / (1024 * 1024)
        if MODEL_PATH.exists() else 0
    )

    return {
        "ready": ready,
        "size_mb": size_mb,
        "model_path": str(MODEL_PATH),
        "labels_path": str(LABELS_PATH)
    }


def predict_leaf_disease(image):
    start = time.perf_counter()

    try:
        import tensorflow as tf
        from PIL import Image

        model = _load_model()
        labels = _load_labels()

        if model is None or labels is None:
            return {"error": "15-class model or labels are missing."}

        image = image.convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        array = np.asarray(image, dtype=np.float32)

        array = tf.keras.applications.mobilenet_v2.preprocess_input(
            array
        )

        batch = np.expand_dims(array, axis=0)

        output = model.predict(batch, verbose=0)
        scores = np.asarray(output).reshape(-1)

        if len(scores) != len(labels):
            return {"error": "Model and labels do not match."}

        order = np.argsort(scores)[::-1][:3]

        top = []

        for i in order:
            raw_label = str(labels[int(i)])
            top.append({
                "label": raw_label,
                "display_label": _format_disease(raw_label),
                "confidence": float(scores[int(i)])
            })

        predicted_label = top[0]["label"]
        confidence = top[0]["confidence"]

        return{
            "label": predicted_label,
            "display_label": top[0]["display_label"],
            "crop": _get_crop(predicted_label),
            "confidence": confidence,
            "top_predictions": top,

            # Confidence warning
            "low_confidence": confidence < 0.70,

            # Warning message
            "confidence_message": (
                "Low confidence! Please upload a clearer leaf image "
                "or consult an agriculture expert."
                if confidence < 0.70
                else "The model has relatively high confidence "
                     "in this prediction."
            ),

            "latency_ms": (
                time.perf_counter() - start
            ) * 1000
        }
    except Exception as exc:
        return {"error": f"Prediction failed: {exc}"}
