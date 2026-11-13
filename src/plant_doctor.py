
import json
import time
import os
import io
import re
from pathlib import Path
from functools import lru_cache
from threading import Lock

import numpy as np


# ============================================================
# AGRISENSE AI - PLANT DOCTOR
# MobileNetV2 + TensorFlow Lite + Optional Gemini Verification
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TFLITE_PATH = (
    ROOT / "models" / "plant_disease_model_15class.tflite"
)

MODEL_PATH = (
    ROOT / "models" / "plant_disease_model_15class.keras"
)

LABELS_PATH = ROOT / "models" / "labels_15class.json"

IMAGE_SIZE = 160

SUPPORTED_CROPS = {"Tomato", "Potato", "Pepper"}

# A shared TFLite interpreter is not thread-safe.
_TFLITE_LOCK = Lock()


# ============================================================
# 1. LOAD TFLITE MODEL
# ============================================================

@lru_cache(maxsize=1)
def _load_tflite():

    if not TFLITE_PATH.exists():
        return None

    import tensorflow as tf

    interpreter = tf.lite.Interpreter(
        model_path=str(TFLITE_PATH)
    )

    interpreter.allocate_tensors()

    return interpreter


# ============================================================
# 2. LOAD KERAS MODEL
# ============================================================

@lru_cache(maxsize=1)
def _load_model():

    if not MODEL_PATH.exists():
        return None

    import tensorflow as tf

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# ============================================================
# 3. LOAD AND VALIDATE LABELS
# ============================================================

@lru_cache(maxsize=1)
def _load_labels():

    if not LABELS_PATH.exists():
        return None

    with open(
        LABELS_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    if isinstance(data, list):
        labels = data

    elif isinstance(data, dict) and all(
        str(i) in data for i in range(len(data))
    ):
        labels = [
            data[str(i)]
            for i in range(len(data))
        ]

    elif (
        isinstance(data, dict)
        and all(isinstance(v, int) for v in data.values())
        and sorted(data.values()) == list(range(len(data)))
    ):

        labels = [None] * len(data)

        for name, index in data.items():
            labels[index] = name

    else:
        raise ValueError(
            "Invalid label mapping in labels_15class.json"
        )

    if len(labels) != 15:
        raise ValueError(
            "Expected exactly 15 class labels."
        )

    if not all(isinstance(x, str) for x in labels):
        raise ValueError(
            "All class labels must be strings."
        )

    if len(set(labels)) != len(labels):
        raise ValueError(
            "Duplicate class labels detected."
        )

    return labels


# ============================================================
# 4. IDENTIFY CROP FROM CLASS LABEL
# ============================================================

def _get_crop(label):

    label = str(label)

    if label.startswith("Tomato"):
        return "Tomato"

    elif label.startswith("Potato"):
        return "Potato"

    elif label.startswith("Pepper"):
        return "Pepper"

    return "Unknown"


# ============================================================
# 5. FORMAT DISEASE NAME
# ============================================================

def _format_disease(label):

    label = label.replace("___", "_")
    label = label.replace("__", "_")
    label = label.replace("_", " ")

    replacements = {

        "Pepper bell Bacterial spot":
            "Bacterial Spot",

        "Pepper bell healthy":
            "Healthy",

        "Potato Early blight":
            "Early Blight",

        "Potato Late blight":
            "Late Blight",

        "Potato healthy":
            "Healthy",

        "Tomato Bacterial spot":
            "Bacterial Spot",

        "Tomato Early blight":
            "Early Blight",

        "Tomato healthy":
            "Healthy",

        "Tomato Late blight":
            "Late Blight",

        "Tomato Leaf Mold":
            "Leaf Mold",

        "Tomato Septoria leaf spot":
            "Septoria Leaf Spot",

        "Tomato Spider mites Two spotted spider mite":
            "Spider Mites",

        "Tomato Target Spot":
            "Target Spot",

        "Tomato Tomato mosaic virus":
            "Mosaic Virus",

        "Tomato Tomato YellowLeaf Curl Virus":
            "Yellow Leaf Curl Virus"
    }

    return replacements.get(label, label)


# ============================================================
# 6. OPTIONAL GEMINI CROP IDENTIFICATION
# ============================================================

def identify_crop(image):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    image_buffer = io.BytesIO()

    image.convert("RGB").save(
        image_buffer,
        format="JPEG"
    )

    image_part = types.Part.from_bytes(
        data=image_buffer.getvalue(),
        mime_type="image/jpeg"
    )

    prompt = """
    You are a plant identification assistant.

    Examine the uploaded image carefully.

    Identify the most likely crop or plant.

    Crops may include Tomato, Potato, Pepper,
    Apple, Rice, Wheat, Maize, Cotton, Grape,
    Mango, Banana, Soybean, or other plants.

    Rules:
    1. Do not assume the crop is Tomato,
       Potato, or Pepper.
    2. Identify the plant, not the disease.
    3. If the image is unclear, return Unknown.
    4. If no recognizable plant is visible,
       set is_plant to false.
    5. Return valid JSON only.

    Required JSON:

    {
        "is_plant": true,
        "crop": "Tomato",
        "confidence": 0.90,
        "reason": "Visible leaf characteristics"
    }

    Confidence is an approximate model estimate,
    not a calibrated probability.
    """

    response = client.models.generate_content(
        model="gemini-flash-lite-latest",
        contents=[prompt, image_part],
        config={
            "max_output_tokens": 300,
            "response_mime_type": "application/json"
        }
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    raw = response.text.strip()

    raw = re.sub(
        r"^```(?:json)?\s*",
        "",
        raw
    )

    raw = re.sub(
        r"\s*```$",
        "",
        raw
    )

    result = json.loads(raw)

    crop = str(
        result.get("crop", "Unknown")
    ).strip()

    is_plant = bool(
        result.get("is_plant", False)
    )

    try:
        confidence = float(
            result.get("confidence", 0)
        )

    except (ValueError, TypeError):
        confidence = 0.0

    return {
        "is_plant": is_plant,
        "crop": crop,
        "confidence": max(
            0.0,
            min(1.0, confidence)
        ),
        "reason": str(
            result.get("reason", "")
        )
    }


# ============================================================
# 7. MODEL STATUS
# ============================================================

def model_status():

    tflite_exists = TFLITE_PATH.exists()
    keras_exists = MODEL_PATH.exists()

    active_path = (
        TFLITE_PATH
        if tflite_exists
        else MODEL_PATH
    )

    size_mb = (
        active_path.stat().st_size / (1024 * 1024)
        if active_path.exists()
        else 0
    )

    return {
        "ready": (
            (tflite_exists or keras_exists)
            and LABELS_PATH.exists()
        ),
        "size_mb": size_mb,
        "model_path": str(active_path),
        "labels_path": str(LABELS_PATH),
        "inference_engine": (
            "TFLite"
            if tflite_exists
            else "Keras"
        )
    }


# ============================================================
# 8. NORMALIZE GEMINI CROP NAME
# ============================================================

def _canonical_crop(name):

    words = set(
        re.findall(
            r"[a-z]+",
            str(name).lower()
        )
    )

    matches = {
        crop
        for crop in SUPPORTED_CROPS
        if crop.lower() in words
    }

    if len(matches) == 1:
        return next(iter(matches))

    if not matches:
        return str(name).strip() or "Unknown"

    return "Unknown"


# ============================================================
# 9. IMAGE PREPROCESSING
# ============================================================

def _prepare_image(image):

    import tensorflow as tf
    from PIL import ImageOps

    # Correct camera orientation.
    image = ImageOps.exif_transpose(image)

    # Convert to RGB.
    image = image.convert("RGB")

    # Resize to model input size.
    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Convert image to float32 array.
    array = np.asarray(
        image,
        dtype=np.float32
    )

    # IMPORTANT:
    # Must match preprocessing used in training.
    array = (
        tf.keras.applications.mobilenet_v2
        .preprocess_input(array)
    )

    # Add batch dimension.
    batch = np.expand_dims(
        array,
        axis=0
    )

    return batch


# ============================================================
# 10. TFLITE INFERENCE
# ============================================================

def _run_tflite(batch):

    interpreter = _load_tflite()

    if interpreter is None:
        raise FileNotFoundError(
            "TFLite model is unavailable."
        )

    with _TFLITE_LOCK:

        input_details = (
            interpreter.get_input_details()[0]
        )

        output_details = (
            interpreter.get_output_details()[0]
        )

        expected_shape = tuple(
            int(v)
            for v in input_details["shape"]
        )

        if expected_shape != tuple(batch.shape):
            raise ValueError(
                f"Expected input shape {expected_shape}, "
                f"received {tuple(batch.shape)}"
            )

        input_dtype = input_details["dtype"]

        # Handle quantized input if applicable.
        if np.issubdtype(input_dtype, np.integer):

            scale, zero_point = (
                input_details["quantization"]
            )

            if scale <= 0:
                raise ValueError(
                    "Invalid input quantization scale."
                )

            limits = np.iinfo(input_dtype)

            values = np.clip(
                np.rint(batch / scale + zero_point),
                limits.min,
                limits.max
            )

            input_batch = values.astype(input_dtype)

        else:
            input_batch = batch.astype(input_dtype)

        interpreter.set_tensor(
            input_details["index"],
            input_batch
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_details["index"]
        )

    # Convert quantized output to floating-point.
    output_dtype = output_details["dtype"]

    if np.issubdtype(output_dtype, np.integer):

        scale, zero_point = (
            output_details["quantization"]
        )

        if scale <= 0:
            raise ValueError(
                "Invalid output quantization scale."
            )

        output = (
            output.astype(np.float32) - zero_point
        ) * scale

    return output


# ============================================================
# 11. KERAS FALLBACK
# ============================================================

def _run_keras(batch):

    model = _load_model()

    if model is None:
        raise FileNotFoundError(
            "Keras model is unavailable."
        )

    return model.predict(
        batch,
        verbose=0
    )


# ============================================================
# 12. VALIDATE MODEL OUTPUT SCORES
# ============================================================

def _display_scores(output):

    values = np.asarray(
        output,
        dtype=np.float64
    ).reshape(-1)

    if len(values) != 15:
        raise ValueError(
            "Model must output exactly 15 class scores."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Model returned invalid prediction values."
        )

    # Already normalized, probability-like outputs.
    if (
        np.all(values >= 0)
        and np.all(values <= 1)
        and np.isclose(
            values.sum(),
            1,
            atol=0.01
        )
    ):

        return (
            values,
            "model-output score (not calibrated)"
        )

    # Convert arbitrary logits into relative softmax scores.
    shifted = values - np.max(values)

    exp_values = np.exp(shifted)

    scores = exp_values / exp_values.sum()

    return (
        scores,
        "relative softmax score (not calibrated)"
    )


# ============================================================
# 13. MAIN PLANT DISEASE PREDICTION
# ============================================================

def predict_leaf_disease(
    image,
    verify_with_gemini=False
):

    start = time.perf_counter()

    try:

        # ----------------------------------------------------
        # STEP 1: LOAD LABELS
        # ----------------------------------------------------

        labels = _load_labels()

        if labels is None:
            return {
                "status": "error",
                "error": "labels_15class.json is missing."
            }

        # ----------------------------------------------------
        # STEP 2: PREPROCESS IMAGE
        # ----------------------------------------------------

        batch = _prepare_image(image)

        # ----------------------------------------------------
        # STEP 3: RUN LOCAL CNN
        # ----------------------------------------------------

        fallback_note = None

        try:

            raw_output = _run_tflite(batch)

            engine = "TFLite"

        except Exception as tflite_error:

            fallback_note = (
                "TFLite failed "
                f"({type(tflite_error).__name__}: "
                f"{tflite_error}); used Keras."
            )

            if _load_model() is None:
                return {
                    "status": "error",
                    "error": (
                        fallback_note
                        + " Keras model unavailable."
                    )
                }

            raw_output = _run_keras(batch)

            engine = "Keras"

        # ----------------------------------------------------
        # STEP 4: PROCESS PREDICTIONS
        # ----------------------------------------------------

        scores, score_type = _display_scores(
            raw_output
        )

        if len(scores) != len(labels):
            return {
                "status": "error",
                "error": "Model and label count mismatch."
            }

        order = np.argsort(scores)[::-1][:3]

        top_predictions = []

        for index in order:

            label = labels[int(index)]

            top_predictions.append({
                "label": label,
                "display_label": _format_disease(label),
                "confidence": float(scores[int(index)])
            })

        # ----------------------------------------------------
        # STEP 5: EXTRACT MAIN PREDICTION
        # ----------------------------------------------------

        predicted_label = (
            top_predictions[0]["label"]
        )

        predicted_crop = _get_crop(
            predicted_label
        )

        confidence = (
            top_predictions[0]["confidence"]
        )

        second_score = (
            top_predictions[1]["confidence"]
        )

        margin = confidence - second_score

        # Heuristic uncertainty checks.
        low_confidence = (
            confidence < 0.70
            or margin < 0.10
        )

        result = {
            "status": "success",
            "label": predicted_label,
            "display_label": (
                top_predictions[0]["display_label"]
            ),
            "crop": predicted_crop,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "low_confidence": low_confidence,
            "confidence_message": (
                "Uncertain prediction. Review alternative "
                "classes and consult an agriculture expert."
                if low_confidence
                else
                "The model returned a high relative score. "
                "This does not guarantee correctness."
            ),
            "score_type": score_type,
            "top_two_margin": float(margin),
            "inference_engine": engine,
            "fallback_note": fallback_note,
            "crop_verification": "not_requested",
            "latency_ms": (
                time.perf_counter() - start
            ) * 1000
        }

        # ----------------------------------------------------
        # STEP 6: OPTIONAL GEMINI CROP VERIFICATION
        # ----------------------------------------------------

        if verify_with_gemini:

            try:

                gemini_result = identify_crop(image)

                gemini_crop = _canonical_crop(
                    gemini_result.get("crop", "Unknown")
                )

                result["gemini_crop"] = gemini_crop

                result["gemini_crop_score"] = (
                    gemini_result.get("confidence")
                )

                if not gemini_result.get(
                    "is_plant", False
                ):

                    result["crop_verification"] = (
                        "no_plant_detected"
                    )

                    result["status"] = "crop_mismatch"

                    result["error"] = (
                        "Gemini could not identify a plant. "
                        "No disease diagnosis is shown."
                    )

                elif gemini_crop == "Unknown":

                    result["crop_verification"] = (
                        "uncertain"
                    )

                    result["status"] = "crop_mismatch"

                    result["error"] = (
                        "Gemini crop identification is "
                        "uncertain. No disease diagnosis "
                        "is shown."
                    )

                elif gemini_crop not in SUPPORTED_CROPS:

                    result["crop_verification"] = (
                        "unsupported_crop"
                    )

                    result["status"] = "crop_mismatch"

                    result["error"] = (
                        f"Gemini identified {gemini_crop}, "
                        "which is not supported by the "
                        "15-class disease classifier. "
                        "No disease diagnosis is shown."
                    )

                elif gemini_crop != predicted_crop:

                    result["crop_verification"] = (
                        "mismatch"
                    )

                    result["status"] = "crop_mismatch"

                    result["error"] = (
                        "Crop disagreement: "
                        f"Gemini identified {gemini_crop}, "
                        "but the CNN predicted "
                        f"{predicted_crop} "
                        f"({predicted_label}). "
                        "Diagnosis withheld. Review "
                        "preprocessing, label order "
                        "and the image."
                    )

                else:

                    result["crop_verification"] = (
                        "match"
                    )

            except Exception as exc:

                result["crop_verification"] = (
                    "unavailable"
                )

                result["verification_note"] = (
                    "Gemini verification unavailable: "
                    f"{type(exc).__name__}"
                )

                result["low_confidence"] = True

                result["confidence_message"] = (
                    "Online crop verification unavailable. "
                    "CNN output is preliminary."
                )

        # ----------------------------------------------------
        # STEP 7: RETURN FINAL RESULT
        # ----------------------------------------------------

        result["latency_ms"] = (
            time.perf_counter() - start
        ) * 1000

        return result

    except Exception as exc:

        return {
            "status": "error",
            "error": (
                "Prediction failed: "
                f"{type(exc).__name__}: {exc}"
            )
        }
