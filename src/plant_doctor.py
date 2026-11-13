"""AgriSense AI Plant Doctor.

Crop identification is optional: when Gemini is unavailable, the local
MobileNetV2 classifier can still run, but its result is clearly marked
UNVERIFIED. It cannot reliably reject unsupported crops without a verifier.
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

ROOT = Path(__file__).resolve().parents[1]
TFLITE_PATH = ROOT / "models" / "plant_disease_model_15class.tflite"
MODEL_PATH = ROOT / "models" / "plant_disease_model_15class.keras"
LABELS_PATH = ROOT / "models" / "labels_15class.json"
IMAGE_SIZE = 160
SUPPORTED_CROPS = {"Tomato", "Potato", "Pepper"}

LOGGER = logging.getLogger(__name__)
_TFLITE_LOCK = Lock()


def _get_crop(label):
    """Extract a crop from one of the trained PlantVillage class labels."""
    label = str(label)
    for crop in ("Tomato", "Potato", "Pepper"):
        if label.lower().startswith(crop.lower()):
            return crop
    return "Unknown"


def _format_disease(label):
    label = str(label)
    # Matches the original project's DISEASE_GUIDE class names.
    normalized = re.sub(r"_+", " ", label).strip()
    names = {
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
    return names.get(normalized, normalized)


@lru_cache(maxsize=1)
def _load_labels():
    with LABELS_PATH.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    if isinstance(raw, list):
        labels = raw
    elif isinstance(raw, dict):
        if all(str(key).isdigit() for key in raw):
            ordered = {int(key): value for key, value in raw.items()}
        elif all(type(value) is int for value in raw.values()):
            ordered = {value: key for key, value in raw.items()}
        else:
            raise ValueError("Unknown labels JSON format.")

        if sorted(ordered) != list(range(len(ordered))):
            raise ValueError("Labels must use consecutive indexes from zero.")
        labels = [ordered[i] for i in range(len(ordered))]
    else:
        raise ValueError("Labels JSON must be a list or dictionary.")

    if len(labels) != 15 or len(set(labels)) != 15:
        raise ValueError("Expected 15 distinct PlantVillage class labels.")
    if not all(isinstance(label, str) for label in labels):
        raise ValueError("All class labels must be strings.")
    counts = {crop: sum(_get_crop(label) == crop for label in labels)
              for crop in SUPPORTED_CROPS}
    if counts != {"Tomato": 10, "Potato": 3, "Pepper": 2}:
        raise ValueError(f"Unexpected crop labels: {counts}")
    return labels


@lru_cache(maxsize=1)
def _load_tflite():
    if not TFLITE_PATH.is_file():
        return None
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    interpreter = Interpreter(model_path=str(TFLITE_PATH))
    interpreter.allocate_tensors()
    return interpreter


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.is_file():
        return None
    import tensorflow as tf
    return tf.keras.models.load_model(str(MODEL_PATH), compile=False)


def model_status():
    tflite_exists = TFLITE_PATH.is_file()
    keras_exists = MODEL_PATH.is_file()
    active = TFLITE_PATH if tflite_exists else MODEL_PATH
    try:
        _load_labels()
        labels_valid = True
        labels_error = ""
    except Exception as exc:
        labels_valid = False
        labels_error = str(exc)

    return {
        "ready": bool((tflite_exists or keras_exists) and labels_valid),
        "size_mb": active.stat().st_size / (1024 * 1024)
                   if active.is_file() else 0.0,
        "model_path": str(active),
        "labels_path": str(LABELS_PATH),
        "inference_engine": "TFLite" if tflite_exists else
                            "Keras" if keras_exists else "Unavailable",
        "labels_error": labels_error,
    }


def _canonical_crop(name):
    value = re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ",
                   str(name or "").lower())).strip()
    if value in {"capsicum", "bell pepper", "sweet pepper", "pepper",
                 "bell pepper leaf", "pepper plant"}:
        return "Pepper"
    for crop in ("Tomato", "Potato", "Pepper"):
        if value in {crop.lower(), crop.lower() + " leaf",
                     crop.lower() + " plant"}:
            return crop
    if value in {"", "unknown", "unidentified", "unsure", "uncertain"}:
        return "Unknown"
    return value.title()


def identify_crop(image):
    """Ask Gemini which crop is shown; raises if the API is unavailable."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured on the service.")

    from PIL import ImageOps
    from google import genai
    from google.genai import types

    buffer = io.BytesIO()
    ImageOps.exif_transpose(image).convert("RGB").save(
        buffer, format="JPEG", quality=90
    )
    prompt = (
        "Identify the crop or plant in this photograph. Do NOT diagnose "
        "disease. Do NOT assume it is Tomato, Potato or Pepper: it could "
        "be Apple, Mango, another plant, or not a plant. If uncertain, "
        "return Unknown. Reply ONLY as JSON with keys: "
        "is_plant (boolean), crop (common crop name or Unknown), "
        "confidence (number between 0 and 1), reason (short text). "
        "Confidence is an approximate, uncalibrated estimate."
    )

    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        # This model was suggested in the user's Render 404 error.
        # Override via Render's GEMINI_MODEL environment variable if needed.
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        contents=[prompt, types.Part.from_bytes(
            data=buffer.getvalue(), mime_type="image/jpeg"
        )],
        config={"response_mime_type": "application/json",
                "max_output_tokens": 500},
    )
    raw = getattr(response, "text", None)
    if not raw:
        raise RuntimeError("Gemini returned an empty response.")
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("Gemini response is not a JSON object.")

    try:
        confidence = float(result.get("confidence", 0))
    except (ValueError, TypeError):
        confidence = 0.0
    if not 0 <= confidence <= 1:
        confidence = 0.0
    return {
        "is_plant": result.get("is_plant") is True,
        "crop": _canonical_crop(result.get("crop")),
        "confidence": confidence,
        "reason": str(result.get("reason", "")),
    }


def _prepare_image(image):
    import numpy as np
    from PIL import ImageOps
    resized = ImageOps.exif_transpose(image).convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )
    data = np.asarray(resized, dtype=np.float32)
    # Equivalent to tf.keras.applications.mobilenet_v2.preprocess_input.
    # This must be the same preprocessing that the model used in training.
    return np.expand_dims(data / 127.5 - 1.0, axis=0)


def _run_tflite(batch):
    import numpy as np
    interpreter = _load_tflite()
    if interpreter is None:
        raise FileNotFoundError("TFLite model file not found.")
    with _TFLITE_LOCK:
        input_info = interpreter.get_input_details()[0]
        output_info = interpreter.get_output_details()[0]
        expected = tuple(int(x) for x in input_info["shape"])
        if expected != tuple(batch.shape):
            raise ValueError(f"Model input {expected}, image batch {batch.shape}.")

        dtype = np.dtype(input_info["dtype"])
        if np.issubdtype(dtype, np.integer):
            scale, zero = input_info["quantization"]
            if scale <= 0:
                raise ValueError("Invalid TFLite input quantization.")
            lim = np.iinfo(dtype)
            data = np.clip(np.rint(batch / scale + zero),
                           lim.min, lim.max).astype(dtype)
        else:
            data = batch.astype(dtype)
        interpreter.set_tensor(input_info["index"], data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_info["index"])

        out_dtype = np.dtype(output_info["dtype"])
        if np.issubdtype(out_dtype, np.integer):
            scale, zero = output_info["quantization"]
            if scale <= 0:
                raise ValueError("Invalid TFLite output quantization.")
            output = (output.astype(np.float32) - zero) * scale
    return output


def _run_keras(batch):
    model = _load_model()
    if model is None:
        raise FileNotFoundError("Keras model file not found.")
    return model.predict(batch, verbose=0)


def _normalize_scores(raw):
    import numpy as np
    scores = np.asarray(raw, dtype=np.float64).reshape(-1)
    if len(scores) != 15 or not np.all(np.isfinite(scores)):
        raise ValueError("Model must return 15 finite class scores.")
    if (np.all(scores >= 0) and np.all(scores <= 1) and
            np.isclose(scores.sum(), 1.0, atol=1e-3)):
        return scores
    # Assumes non-probability output is logits; training must confirm this.
    exps = np.exp(scores - scores.max())
    return exps / exps.sum()


def _error(status, message, started, **details):
    return {"status": status, "error": message,
            "latency_ms": (time.perf_counter() - started) * 1000,
            **details}


def predict_leaf_disease(image, verify_with_gemini=True):
    """Return a UI-compatible prediction with explicit verification status."""
    started = time.perf_counter()
    crop_check = None
    verification_state = "not_requested"
    verification_warning = ""

    try:
        # An optional *online* crop check: a 503 must not disable local AI.
        if verify_with_gemini:
            try:
                crop_check = identify_crop(image)
                verification_state = "verified"
            except Exception as exc:
                LOGGER.exception("Gemini crop verification unavailable")
                verification_state = "unavailable"
                verification_warning = (
                    "Gemini crop verification is unavailable "
                    f"({type(exc).__name__}). Showing an UNVERIFIED local "
                    "model prediction. Confirm the crop yourself; unsupported "
                    "plants such as Apple or Mango can be misclassified."
                )

        if crop_check is not None:
            identified_crop = crop_check["crop"]
            crop_confidence = crop_check["confidence"]
            if not crop_check["is_plant"]:
                return _error("not_plant",
                              "No recognizable plant detected. Upload a clear leaf image.",
                              started)
            if identified_crop == "Unknown" or crop_confidence < 0.60:
                return _error("uncertain_crop",
                              "Crop identification is uncertain. Upload a clearer image.",
                              started, crop=identified_crop,
                              crop_confidence=crop_confidence)
            if identified_crop not in SUPPORTED_CROPS:
                return _error("unsupported_crop",
                              f"Identified crop: {identified_crop}. Disease detection "
                              "is supported only for Tomato, Potato and Pepper.",
                              started, crop=identified_crop,
                              crop_confidence=crop_confidence)

        labels = _load_labels()
        batch = _prepare_image(image)
        engine = "TFLite"
        fallback_note = None
        if TFLITE_PATH.is_file():
            try:
                output = _run_tflite(batch)
            except Exception as exc:
                LOGGER.exception("TFLite inference failed; trying Keras")
                if not MODEL_PATH.is_file():
                    raise RuntimeError("TFLite failed and no Keras fallback exists.") from exc
                engine = "Keras"
                output = _run_keras(batch)
                fallback_note = "Used Keras because TFLite inference failed."
        elif MODEL_PATH.is_file():
            engine = "Keras"
            output = _run_keras(batch)
        else:
            return _error("model_missing",
                          "Both TFLite and Keras model files are missing.", started)

        scores = _normalize_scores(output)
        import numpy as np
        order = np.argsort(scores)[::-1][:3]
        top = []
        for index in order:
            i = int(index)
            label = labels[i]
            top.append({
                "label": label,
                "display_label": _format_disease(label),
                "confidence": float(scores[i]),
            })

        label = top[0]["label"]
        predicted_crop = _get_crop(label)
        if predicted_crop == "Unknown":
            raise ValueError(f"Unknown class label: {label}")

        if crop_check is not None and crop_check["crop"] != predicted_crop:
            return _error(
                "crop_mismatch",
                f"Crop mismatch: Gemini identified {crop_check['crop']}, "
                f"but the disease classifier predicted {predicted_crop}. "
                "No disease diagnosis was shown.",
                started, crop=crop_check["crop"],
                classifier_crop=predicted_crop,
                crop_confidence=crop_check["confidence"],
            )

        confidence = float(top[0]["confidence"])
        margin = confidence - float(top[1]["confidence"])
        uncertain = confidence < 0.70 or margin < 0.10
        unverified = crop_check is None

        if unverified:
            confidence_message = verification_warning or (
                "Crop was not independently verified. This is an UNVERIFIED "
                "local prediction; unsupported crops may be misclassified."
            )
        elif uncertain:
            confidence_message = (
                "The model is uncertain. Confirm symptoms with an agriculture expert."
            )
        else:
            confidence_message = (
                "Gemini and the classifier agree on the crop, but the "
                "disease and prediction score are not guaranteed to be correct."
            )

        return {
            "status": "success",
            "label": label,
            "display_label": top[0]["display_label"],
            "crop": predicted_crop,
            "crop_confidence": crop_check["confidence"] if crop_check else None,
            "confidence": confidence,
            "top_predictions": top,
            "low_confidence": bool(uncertain or unverified),
            "confidence_message": confidence_message,
            "crop_verification": (
                "match" if crop_check is not None else verification_state
            ),
            "verification_warning": verification_warning,
            "inference_engine": engine,
            "fallback_note": fallback_note,
            "top_two_margin": margin,
            "latency_ms": (time.perf_counter() - started) * 1000,
        }

    except Exception:
        LOGGER.exception("Plant Doctor prediction failed")
        return _error(
            "error",
            "Plant Doctor could not run the disease model. Check the "
            "model files, labels, dependencies and Render Logs.",
            started,
        )
