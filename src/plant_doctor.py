
"""AgriSense AI - Plant Disease Screening."""

import io
import json
import logging
import os
import re
import time

from functools import lru_cache
from pathlib import Path
from threading import Lock


# ======================================================
# AGRISENSE AI - PLANT DOCTOR
# MobileNetV2 + TFLite + Gemini 3.8 Flash
# ======================================================

LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

TFLITE_PATH = (
    ROOT / "models" / "plant_disease_model_15class.tflite"
)

KERAS_PATH = (
    ROOT / "models" / "plant_disease_model_15class.keras"
)

LABELS_PATH = ROOT / "models" / "labels_15class.json"

IMAGE_SIZE = (160, 160)

SUPPORTED_CROPS = {"Tomato", "Potato", "Pepper"}

_TFLITE_LOCK = Lock()


# ======================================================
# 1. CROP NAME FROM MODEL LABEL
# ======================================================

def _crop_from_label(label):
    match = re.match(
        r"^(Tomato|Potato|Pepper)(?:\W|_|$)",
        str(label),
        re.I
    )

    return (
        match.group(1).capitalize()
        if match
        else "Unknown"
    )


# ======================================================
# 2. FORMAT DISEASE NAME
# ======================================================

def _display_disease(label):
    return (
        str(label)
        .split("___", 1)[-1]
        .replace("_", " ")
        .strip()
    )


# ======================================================
# 3. LOAD LABELS
# ======================================================

@lru_cache(maxsize=1)
def _load_labels():

    with LABELS_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    if isinstance(data, list):
        labels = data

    elif isinstance(data, dict):

        if all(str(k).isdigit() for k in data):
            by_index = {
                int(k): v
                for k, v in data.items()
            }

        elif all(type(v) is int for v in data.values()):
            by_index = {
                v: k
                for k, v in data.items()
            }

        else:
            raise ValueError(
                "Unsupported labels JSON format"
            )

        if sorted(by_index) != list(range(len(by_index))):
            raise ValueError(
                "Label indexes must start at 0"
            )

        labels = [
            by_index[i]
            for i in range(len(by_index))
        ]

    else:
        raise ValueError(
            "Labels must be a list or dictionary"
        )

    if len(labels) != 15:
        raise ValueError(
            "Expected exactly 15 class labels"
        )

    if not all(isinstance(x, str) for x in labels):
        raise ValueError(
            "Every label must be a string"
        )

    if len(set(labels)) != 15:
        raise ValueError(
            "Duplicate labels detected"
        )

    counts = {
        crop: sum(
            _crop_from_label(x) == crop
            for x in labels
        )
        for crop in SUPPORTED_CROPS
    }

    expected = {
        "Tomato": 10,
        "Potato": 3,
        "Pepper": 2
    }

    if counts != expected:
        raise ValueError(
            f"Unexpected crop counts: {counts}"
        )

    return labels


# ======================================================
# 4. LOAD TFLITE MODEL
# ======================================================

@lru_cache(maxsize=1)
def _load_tflite():

    if not TFLITE_PATH.is_file():
        return None

    try:
        from tflite_runtime.interpreter import Interpreter

    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

    interpreter = Interpreter(
        model_path=str(TFLITE_PATH)
    )

    interpreter.allocate_tensors()

    return interpreter


# ======================================================
# 5. LOAD KERAS MODEL
# ======================================================

@lru_cache(maxsize=1)
def _load_keras():

    if not KERAS_PATH.is_file():
        return None

    import tensorflow as tf

    return tf.keras.models.load_model(
        str(KERAS_PATH),
        compile=False
    )


# ======================================================
# 6. MODEL STATUS
# ======================================================

def model_status():

    has_tflite = TFLITE_PATH.is_file()
    has_keras = KERAS_PATH.is_file()

    active = (
        TFLITE_PATH
        if has_tflite
        else KERAS_PATH
    )

    try:
        labels_ok = len(_load_labels()) == 15
        labels_error = ""

    except Exception as exc:
        labels_ok = False
        labels_error = str(exc)

    return {
        "ready": (
            (has_tflite or has_keras)
            and labels_ok
        ),

        "size_mb": (
            active.stat().st_size / (1024 ** 2)
            if active.is_file()
            else 0.0
        ),

        "model_path": str(active),

        "labels_path": str(LABELS_PATH),

        "inference_engine": (
            "TFLite"
            if has_tflite
            else "Keras"
            if has_keras
            else "Unavailable"
        ),

        "labels_error": labels_error
    }


# ======================================================
# 7. NORMALIZE CROP NAMES
# ======================================================

def _canonical_crop(name):

    text = re.sub(
        r"\s+",
        " ",
        re.sub(
            r"[^a-z ]",
            " ",
            str(name or "").lower()
        )
    ).strip()

    names = {
        "tomato": "Tomato",
        "tomato plant": "Tomato",
        "tomato leaf": "Tomato",

        "potato": "Potato",
        "potato plant": "Potato",
        "potato leaf": "Potato",

        "pepper": "Pepper",
        "pepper plant": "Pepper",
        "pepper leaf": "Pepper",

        "bell pepper": "Pepper",
        "bell pepper plant": "Pepper",
        "capsicum": "Pepper",

        "apple": "Apple",
        "apple tree": "Apple",
        "apple leaf": "Apple",

        "mango": "Mango",
        "mango tree": "Mango",
        "mango leaf": "Mango",

        "unknown": "Unknown",
        "unidentified": "Unknown",
        "": "Unknown"
    }

    return names.get(
        text,
        text.title()
    )


# ======================================================
# 8. GEMINI CROP IDENTIFICATION
# ======================================================

def identify_crop(image):

    key = os.getenv(
        "GEMINI_API_KEY",
        ""
    ).strip()

    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing"
        )

    from PIL import ImageOps
    from google import genai
    from google.genai import types

    buffer = io.BytesIO()

    ImageOps.exif_transpose(
        image
    ).convert("RGB").save(
        buffer,
        format="JPEG",
        quality=90
    )

    prompt = (
        "Identify the plant or crop in this image. "
        "Do NOT diagnose disease. "
        "Do not assume Tomato, Potato or Pepper. "
        "The image may contain Apple, Mango or "
        "another unsupported crop. "
        "If the crop cannot be identified, "
        "return Unknown. "
        "Return only a JSON object with fields: "
        "is_plant (boolean), crop (short common name), "
        "confidence (number from 0 to 1), "
        "reason (short text). "
        "Confidence is an uncalibrated estimate."
    )

    client = genai.Client(
        api_key=key
    )

    # Updated model from Render error.
    response = client.models.generate_content(
        model="gemini-3.8-flash",

        contents=[
            prompt,

            types.Part.from_bytes(
                data=buffer.getvalue(),
                mime_type="image/jpeg"
            )
        ],

        config={
            "response_mime_type": "application/json",
            "max_output_tokens": 400
        }
    )

    raw = getattr(
        response,
        "text",
        None
    )

    if not raw:
        raise RuntimeError(
            "Gemini returned an empty response"
        )

    raw = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        raw.strip(),
        flags=re.I
    )

    result = json.loads(raw)

    if not isinstance(result, dict):
        raise ValueError(
            "Unexpected Gemini JSON response"
        )

    try:
        confidence = float(
            result.get("confidence", 0)
        )

    except (ValueError, TypeError):
        confidence = 0.0

    if not (0 <= confidence <= 1):
        confidence = 0.0

    return {
        "is_plant": (
            result.get("is_plant") is True
        ),

        "crop": _canonical_crop(
            result.get("crop")
        ),

        "confidence": confidence,

        "reason": str(
            result.get("reason", "")
        )
    }


# ======================================================
# 9. IMAGE PREPROCESSING
# ======================================================

def _prepare_image(image):

    import numpy as np
    from PIL import ImageOps

    image = ImageOps.exif_transpose(
        image
    )

    image = image.convert("RGB")

    image = image.resize(
        IMAGE_SIZE
    )

    pixels = np.asarray(
        image,
        dtype=np.float32
    )

    # MobileNetV2 preprocessing.
    # Must match the original training pipeline.
    pixels = pixels / 127.5 - 1.0

    batch = np.expand_dims(
        pixels,
        axis=0
    )

    return batch


# ======================================================
# 10. TFLITE INFERENCE
# ======================================================

def _run_tflite(batch):

    import numpy as np

    interpreter = _load_tflite()

    if interpreter is None:
        raise FileNotFoundError(
            "TFLite model missing"
        )

    with _TFLITE_LOCK:

        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]

        expected = tuple(
            int(x)
            for x in inp["shape"]
        )

        if expected != tuple(batch.shape):
            raise ValueError(
                f"Expected {expected}, got {batch.shape}"
            )

        dtype = np.dtype(inp["dtype"])

        if np.issubdtype(dtype, np.integer):

            scale, zero = inp["quantization"]

            if scale <= 0:
                raise ValueError(
                    "Invalid input quantization"
                )

            limits = np.iinfo(dtype)

            data = np.clip(
                np.rint(batch / scale + zero),
                limits.min,
                limits.max
            ).astype(dtype)

        else:
            data = batch.astype(dtype)

        interpreter.set_tensor(
            inp["index"],
            data
        )

        interpreter.invoke()

        result = interpreter.get_tensor(
            out["index"]
        )

        output_dtype = np.dtype(
            out["dtype"]
        )

        if np.issubdtype(
            output_dtype,
            np.integer
        ):

            scale, zero = out["quantization"]

            if scale <= 0:
                raise ValueError(
                    "Invalid output quantization"
                )

            result = (
                result.astype(np.float32) - zero
            ) * scale

    return result


# ======================================================
# 11. KERAS INFERENCE
# ======================================================

def _run_keras(batch):

    model = _load_keras()

    if model is None:
        raise FileNotFoundError(
            "Keras model missing"
        )

    return model.predict(
        batch,
        verbose=0
    )


# ======================================================
# 12. PROCESS MODEL SCORES
# ======================================================

def _format_scores(raw_output):

    import numpy as np

    values = np.asarray(
        raw_output,
        dtype=np.float64
    ).reshape(-1)

    if (
        len(values) != 15
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            "Expected 15 valid model scores"
        )

    if (
        np.all(values >= 0)
        and np.all(values <= 1)
        and np.isclose(
            values.sum(),
            1.0,
            atol=1e-3
        )
    ):
        return (
            values,
            "model output (not calibrated)"
        )

    # Assumes unnormalized outputs are logits.
    # Verify this against training code.
    exp = np.exp(
        values - np.max(values)
    )

    scores = exp / exp.sum()

    return (
        scores,
        "softmax-style score (not calibrated)"
    )


# ======================================================
# 13. BLOCK INVALID RESULTS
# ======================================================

def _block(
    status,
    message,
    started,
    **details
):

    return {
        "status": status,
        "error": message,

        "latency_ms": (
            time.perf_counter() - started
        ) * 1000,

        **details
    }


# ======================================================
# 14. MAIN PLANT PREDICTION
# ======================================================

def predict_leaf_disease(
    image,
    verify_with_gemini=True
):

    started = time.perf_counter()

    try:
        crop_check = None

        # STEP 1: VERIFY CROP ONLINE
        if verify_with_gemini:

            try:
                crop_check = identify_crop(image)

            except Exception:
                LOGGER.exception(
                    "Gemini crop verification failed"
                )

                return _block(
                    "verification_unavailable",

                    "Online crop verification failed. "
                    "Check Render Logs and Gemini API. "
                    "No disease diagnosis was shown.",

                    started
                )

            if not crop_check["is_plant"]:
                return _block(
                    "no_plant",
                    "No recognizable plant was identified.",
                    started
                )

            if (
                crop_check["crop"] == "Unknown"
                or crop_check["confidence"] < 0.60
            ):
                return _block(
                    "uncertain_crop",

                    "Crop could not be identified reliably.",

                    started,

                    crop=crop_check["crop"],

                    crop_confidence=(
                        crop_check["confidence"]
                    )
                )

            if crop_check["crop"] not in SUPPORTED_CROPS:

                return _block(
                    "unsupported_crop",

                    f"{crop_check['crop']} is not supported. "
                    "Only Tomato, Potato and Pepper "
                    "have disease classes in this model. "
                    "No diagnosis was shown.",

                    started,

                    crop=crop_check["crop"],

                    crop_confidence=(
                        crop_check["confidence"]
                    )
                )

        # STEP 2: LOAD LABELS
        labels = _load_labels()

        # STEP 3: PREPROCESS IMAGE
        batch = _prepare_image(image)

        fallback_note = None

        # STEP 4: TFLITE OR KERAS
        if TFLITE_PATH.is_file():

            try:
                output = _run_tflite(batch)
                engine = "TFLite"

            except Exception as exc:
                LOGGER.exception(
                    "TFLite inference failed"
                )

                if not KERAS_PATH.is_file():
                    return _block(
                        "inference_failed",

                        "TFLite inference failed and "
                        "Keras fallback is missing.",

                        started
                    )

                output = _run_keras(batch)
                engine = "Keras"

                fallback_note = (
                    "TFLite failed: "
                    f"{type(exc).__name__}; "
                    "used Keras fallback."
                )

        else:
            output = _run_keras(batch)
            engine = "Keras"

        # STEP 5: TOP PREDICTIONS
        scores, score_type = _format_scores(
            output
        )

        import numpy as np

        indices = np.argsort(
            scores
        )[::-1][:3]

        top = []

        for i in indices:
            i = int(i)
            label = labels[i]

            top.append({
                "class_index": i,

                "label": label,

                "display_label": (
                    _display_disease(label)
                ),

                "confidence": float(
                    scores[i]
                )
            })

        predicted_crop = _crop_from_label(
            top[0]["label"]
        )

        if predicted_crop == "Unknown":
            raise ValueError(
                "Unexpected crop label"
            )

        # STEP 6: COMPARE CNN WITH GEMINI
        if (
            crop_check
            and predicted_crop != crop_check["crop"]
        ):
            return _block(
                "crop_mismatch",

                f"Crop mismatch: Gemini identified "
                f"{crop_check['crop']}, but CNN "
                f"predicted {predicted_crop}. "
                "Diagnosis withheld.",

                started,

                crop=crop_check["crop"],

                classifier_crop=predicted_crop,

                crop_confidence=(
                    crop_check["confidence"]
                ),

                top_predictions=top,

                inference_engine=engine
            )

        # STEP 7: UNCERTAINTY
        confidence = top[0]["confidence"]

        margin = (
            confidence -
            top[1]["confidence"]
        )

        uncertain = (
            confidence < 0.70
            or margin < 0.10
        )

        offline = not verify_with_gemini

        if offline:
            message = (
                "Offline model output. Unsupported "
                "crops such as Apple and Mango "
                "may be classified incorrectly."
            )

        elif uncertain:
            message = (
                "Model prediction is uncertain. "
                "Confirm symptoms with an expert."
            )

        else:
            message = (
                "Both systems agree on crop. "
                "However, disease classification "
                "is not guaranteed to be correct."
            )

        # STEP 8: FINAL RESULT
        return {
            "status": "success",

            "label": top[0]["label"],

            "display_label": (
                top[0]["display_label"]
            ),

            "crop": predicted_crop,

            "crop_confidence": (
                crop_check["confidence"]
                if crop_check
                else None
            ),

            "confidence": confidence,

            "top_predictions": top,

            "low_confidence": (
                uncertain or offline
            ),

            "confidence_message": message,

            "crop_verification": (
                "match"
                if crop_check
                else "not_requested"
            ),

            "inference_engine": engine,

            "fallback_note": fallback_note,

            "score_type": score_type,

            "top_two_margin": float(margin),

            "latency_ms": (
                time.perf_counter() - started
            ) * 1000
        }

    except Exception:
        LOGGER.exception(
            "Plant Doctor prediction failed"
        )

        return _block(
            "error",

            "Plant Doctor failed. "
            "Check model files and Render Logs.",

            started
        )
