"""AgriSense AI Plant Doctor.

Supported crops (Tomato, Potato, Pepper): trained 15-class CNN diagnosis.
Other crops: Gemini visual observations ONLY, clearly separated from the dataset model.
Gemini unavailable: the CNN can still run, but its crop identity is NOT verified.

The API cannot guarantee correctness on internet photos; check the label mapping
and training preprocessing before interpreting model scores.
"""

import io
import json
import logging
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import ImageOps

ROOT = Path(__file__).resolve().parents[1]
TFLITE_PATH = ROOT / "models" / "plant_disease_model_15class.tflite"
MODEL_PATH = ROOT / "models" / "plant_disease_model_15class.keras"
LABELS_PATH = ROOT / "models" / "labels_15class.json"
IMAGE_SIZE = 160
SUPPORTED_CROPS = {"Tomato", "Potato", "Pepper"}
LOG = logging.getLogger(__name__)
_INFERENCE_LOCK = Lock()


@lru_cache(maxsize=1)
def _load_labels():
    with LABELS_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        labels = raw
    elif isinstance(raw, dict):
        if all(str(k).isdigit() for k in raw):
            by_index = {int(k): v for k, v in raw.items()}
        elif all(isinstance(v, int) and not isinstance(v, bool) for v in raw.values()):
            by_index = {v: k for k, v in raw.items()}
        else:
            raise ValueError("Unsupported labels_15class.json mapping")
        if sorted(by_index) != list(range(len(by_index))):
            raise ValueError("Label indexes must be consecutive from zero")
        labels = [by_index[i] for i in range(len(by_index))]
    else:
        raise ValueError("Labels must be a list or index mapping")
    if len(labels) != 15 or not all(isinstance(x, str) for x in labels):
        raise ValueError("Exactly 15 string labels are required")
    if len(set(labels)) != 15:
        raise ValueError("Duplicate labels found")
    return labels


def _get_crop(label):
    value = str(label).lower()
    if value.startswith("tomato"):
        return "Tomato"
    if value.startswith("potato"):
        return "Potato"
    if value.startswith("pepper"):
        return "Pepper"
    return "Unknown"


def _format_disease(label):
    text = str(label).replace("___", "_").replace("__", "_").replace("_", " ")
    mapping = {
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
    return mapping.get(text, text)


@lru_cache(maxsize=1)
def _load_tflite():
    if not TFLITE_PATH.is_file():
        return None
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    obj = Interpreter(model_path=str(TFLITE_PATH))
    obj.allocate_tensors()
    return obj


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.is_file():
        return None
    import tensorflow as tf
    return tf.keras.models.load_model(str(MODEL_PATH), compile=False)


def model_status():
    available = TFLITE_PATH.is_file() or MODEL_PATH.is_file()
    model_file = TFLITE_PATH if TFLITE_PATH.is_file() else MODEL_PATH
    try:
        _load_labels()
        labels_ok = True
    except Exception:
        labels_ok = False
    return {
        "ready": available and labels_ok,
        "size_mb": model_file.stat().st_size / (1024 * 1024) if model_file.is_file() else 0.0,
        "model_path": str(model_file),
        "labels_path": str(LABELS_PATH),
        "inference_engine": "TFLite" if TFLITE_PATH.is_file() else ("Keras" if MODEL_PATH.is_file() else "Unavailable"),
    }


def _normalize_crop(value):
    value = re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", str(value or "").lower())).strip()
    aliases = {
        "tomato plant": "Tomato", "tomato leaf": "Tomato",
        "potato plant": "Potato", "potato leaf": "Potato",
        "pepper plant": "Pepper", "pepper leaf": "Pepper",
        "capsicum": "Pepper", "bell pepper": "Pepper", "chilli pepper": "Pepper",
        "chili pepper": "Pepper", "sweet pepper": "Pepper",
        "unknown": "Unknown", "not sure": "Unknown", "": "Unknown",
    }
    return aliases.get(value, value.title())


def identify_crop(image):
    """Crop screening only; other-crop health notes are NOT CNN predictions."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing")

    from google import genai
    from google.genai import types

    buf = io.BytesIO()
    picture = ImageOps.exif_transpose(image).convert("RGB")
    picture.thumbnail((1024, 1024))
    picture.save(buf, format="JPEG", quality=80)

    prompt = (
        "You are screening a plant image. Identify the crop without assuming it is "
        "tomato, potato, or pepper. It may be any plant, or no plant. "
        "If you are unsure say Unknown. Return JSON with keys: "
        "is_plant (boolean), crop (common name, or Unknown), "
        "confidence (number 0..1), visible_issue (short observation, NOT a "
        "confirmed diagnosis), advice (one short conservative next step). "
        "Do not invent symptoms. Never give chemical/pesticide dosages. "
        "A single leaf image cannot establish a confirmed disease."
    )

    model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    client = genai.Client(api_key=api_key, http_options={"timeout": 9000})

    result = client.models.generate_content(
        model=model_name,
        contents=[
            prompt,
            types.Part.from_bytes(
                data=buf.getvalue(),
                mime_type="image/jpeg"
            )
        ],
        config={
            "response_mime_type": "application/json",
            "max_output_tokens": 450
        },
    )

    if not getattr(result, "text", None):
        raise RuntimeError("Gemini returned no text")

    parsed = json.loads(
        result.text.strip()
        .removeprefix("```json")
        .removesuffix("```")
        .strip()
    )

    if not isinstance(parsed, dict):
        raise ValueError("Gemini returned invalid JSON structure")

    try:
        confidence = float(parsed.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    return {
        "is_plant": parsed.get("is_plant") is True,
        "crop": _normalize_crop(parsed.get("crop")),
        "confidence": confidence,
        "visible_issue": str(
            parsed.get("visible_issue") or "No reliable issue identified"
        ),
        "advice": str(
            parsed.get("advice") or "Consult a local agriculture expert"
        ),
    }


def _prepare_image(image):
    # Keep the original MobileNetV2 preprocessing (160x160, [-1, 1]).
    picture = ImageOps.exif_transpose(image).convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )
    array = np.asarray(picture, dtype=np.float32)
    return np.expand_dims(array / 127.5 - 1.0, axis=0)


def _run_tflite(batch):
    interpreter = _load_tflite()

    if interpreter is None:
        raise FileNotFoundError("TFLite model unavailable")

    with _INFERENCE_LOCK:
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]

        dtype = np.dtype(inp["dtype"])

        if np.issubdtype(dtype, np.integer):
            scale, offset = inp["quantization"]

            if not scale:
                raise ValueError("Missing quantization scale")

            limits = np.iinfo(dtype)
            value = np.clip(
                np.rint(batch / scale + offset),
                limits.min,
                limits.max
            ).astype(dtype)
        else:
            value = batch.astype(dtype)

        interpreter.set_tensor(inp["index"], value)
        interpreter.invoke()

        result = interpreter.get_tensor(out["index"])

        if np.issubdtype(np.dtype(out["dtype"]), np.integer):
            scale, offset = out["quantization"]

            if not scale:
                raise ValueError("Missing output quantization scale")

            result = (
                result.astype(np.float32) - offset
            ) * scale

        return result


def _scores(output, count):
    values = np.asarray(output, dtype=np.float64).reshape(-1)

    if len(values) != count or not np.all(np.isfinite(values)):
        raise ValueError("Model outputs and labels do not match")

    if (
        np.all((values >= 0) & (values <= 1))
        and np.isclose(values.sum(), 1, atol=0.01)
    ):
        return values

    # Only valid if non-probability model outputs are logits.
    exps = np.exp(values - np.max(values))
    return exps / np.sum(exps)


def _cnn_prediction(image):
    labels = _load_labels()
    batch = _prepare_image(image)

    try:
        raw = _run_tflite(batch)
        engine = "TFLite"

    except Exception as exc:
        LOG.warning(
            "TFLite failed; using Keras fallback: %s", exc
        )

        model = _load_model()

        if model is None:
            raise RuntimeError(
                "Both TFLite and Keras models unavailable"
            ) from exc

        raw = model.predict(batch, verbose=0)
        engine = "Keras"

    scores = _scores(raw, len(labels))
    order = np.argsort(scores)[::-1][:3]

    top = [{
        "label": labels[int(i)],
        "display_label": _format_disease(labels[int(i)]),
        "confidence": float(scores[int(i)])
    } for i in order]

    return top, engine


def predict_leaf_disease(image, verify_with_gemini=True):
    """Route supported crops to CNN; other crops to Gemini-only observations.

    If verification is disabled or unavailable, CNN predictions are explicitly
    unverified and cannot reliably reject crops outside the training set.
    """

    started = time.perf_counter()
    crop_info = None
    verification_problem = None

    if verify_with_gemini:
        try:
            crop_info = identify_crop(image)

        except Exception as exc:
            LOG.warning(
                "Gemini crop check unavailable: %s: %s",
                type(exc).__name__,
                exc
            )
            verification_problem = type(exc).__name__

    if crop_info:
        crop = crop_info["crop"]
        score = crop_info["confidence"]

        if not crop_info["is_plant"]:
            return {
                "status": "no_plant",
                "error": "Gemini did not recognize a plant in this image.",
                "latency_ms": (time.perf_counter() - started) * 1000
            }

        if crop == "Unknown" or score < 0.60:
            return {
                "status": "uncertain_crop",
                "error": "Crop identity is uncertain. Upload a clearer leaf image.",
                "latency_ms": (time.perf_counter() - started) * 1000
            }

        if crop not in SUPPORTED_CROPS:
            summary = (
                f"Gemini Vision: {crop} (AI crop estimate {score:.0%}). "
                f"Visible observation: {crop_info['visible_issue']}. "
                f"Suggested next step: {crop_info['advice']}. "
                "This crop is not in our 15-class dataset; "
                "no CNN disease diagnosis was made."
            )

            return {
                "status": "other_crop",
                "error": summary,
                "crop": crop,
                "crop_confidence": score,
                "source": "Gemini Vision (online)",
                "visible_issue": crop_info["visible_issue"],
                "advice": crop_info["advice"],
                "latency_ms": (time.perf_counter() - started) * 1000
            }

    try:
        top, engine = _cnn_prediction(image)

    except Exception as exc:
        LOG.exception("CNN plant diagnosis failed")

        return {
            "status": "error",
            "error": (
                f"Local disease model error: "
                f"{type(exc).__name__}. Check Render logs."
            ),
            "latency_ms": (time.perf_counter() - started) * 1000
        }

    label = top[0]["label"]
    cnn_crop = _get_crop(label)
    confidence = top[0]["confidence"]
    margin = confidence - top[1]["confidence"]

    if crop_info and crop_info["crop"] != cnn_crop:
        return {
            "status": "crop_mismatch",
            "error": (
                f"Crop disagreement: Gemini identified {crop_info['crop']}; "
                f"CNN predicted {cnn_crop}. Disease result withheld. "
                "Try another clear image."
            ),
            "crop": crop_info["crop"],
            "classifier_crop": cnn_crop,
            "top_predictions": top,
            "latency_ms": (time.perf_counter() - started) * 1000
        }

    unverified = crop_info is None
    uncertain = confidence < 0.70 or margin < 0.10

    if unverified:
        reason = (
            f"Gemini unavailable ({verification_problem}). "
            if verification_problem
            else "Gemini verification disabled. "
        )

        message = (
            reason +
            "UNVERIFIED CNN result. Unsupported crops might be "
            "incorrectly labeled as Tomato/Potato/Pepper; confirm "
            "the crop before relying on this diagnosis."
        )

    elif uncertain:
        message = (
            "The CNN is uncertain; visually confirm "
            "the symptoms before treatment."
        )

    else:
        message = (
            "Gemini crop and CNN crop agree. Disease prediction "
            "is still not a confirmed diagnosis."
        )

    return {
        "status": "success",
        "label": label,
        "display_label": top[0]["display_label"],
        "crop": cnn_crop,
        "confidence": confidence,
        "top_predictions": top,
        "low_confidence": uncertain or unverified,
        "confidence_message": message,
        "inference_engine": engine,
        "crop_verification": (
            "verified_by_gemini"
            if crop_info
            else "unverified"
        ),
        "crop_confidence": (
            crop_info["confidence"]
            if crop_info
            else None
        ),
        "latency_ms": (
            time.perf_counter() - started
        ) * 1000,
    }
