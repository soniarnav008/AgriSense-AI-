
import io
import json
import logging
import math
import os
import re
import time

from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import ImageOps


# ==========================================================
# AGRISENSE AI - PLANT DOCTOR
# MobileNetV2 + TFLite + Gemini Crop Verification
# ==========================================================

LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

TFLITE_PATH = (
    ROOT / "models" / "plant_disease_model_15class.tflite"
)

KERAS_PATH = (
    ROOT / "models" / "plant_disease_model_15class.keras"
)

LABELS_PATH = ROOT / "models" / "labels_15class.json"

IMAGE_SIZE = 160

SUPPORTED_CROPS = {
    "Tomato",
    "Potato",
    "Pepper"
}

TFLITE_LOCK = Lock()


# ==========================================================
# 1. LOAD LABELS
# ==========================================================

@lru_cache(maxsize=1)
def _load_labels():

    with LABELS_PATH.open(
        "r", encoding="utf-8"
    ) as f:
        data = json.load(f)

    if isinstance(data, list):
        labels = data

    elif isinstance(data, dict):

        if all(str(key).isdigit() for key in data):

            mapping = {
                int(key): name
                for key, name in data.items()
            }

        elif all(
            type(value) is int
            for value in data.values()
        ):

            mapping = {
                index: name
                for name, index in data.items()
            }

        else:
            raise ValueError(
                "Unsupported labels JSON format"
            )

        if sorted(mapping) != list(range(len(mapping))):
            raise ValueError(
                "Label indices must start from zero."
            )

        labels = [
            mapping[i]
            for i in range(len(mapping))
        ]

    else:
        raise ValueError(
            "Expected labels JSON list or dictionary."
        )

    if len(labels) != 15:
        raise ValueError(
            "Exactly 15 labels are required."
        )

    if not all(isinstance(x, str) for x in labels):
        raise ValueError(
            "All labels must be strings."
        )

    if len(set(labels)) != 15:
        raise ValueError(
            "Duplicate labels detected."
        )

    counts = {
        crop: sum(
            _get_crop(label) == crop
            for label in labels
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
            f"Unexpected crop distribution: {counts}"
        )

    return labels


# ==========================================================
# 2. GET CROP NAME FROM MODEL LABEL
# ==========================================================

def _get_crop(label):

    found = re.match(
        r"^(Tomato|Potato|Pepper)(?:\b|_)",
        str(label),
        re.I
    )

    if found:
        return found.group(1).capitalize()

    return "Unknown"


# ==========================================================
# 3. FORMAT DISEASE NAME
# ==========================================================

def _format_disease(label):

    text = str(label).split("___", 1)[-1]

    text = text.replace("_", " ")

    text = re.sub(
        r"\s+", " ", text
    ).strip()

    return text or "Unknown condition"


# ==========================================================
# 4. NORMALIZE GEMINI CROP NAMES
# ==========================================================

def _canonical_crop(value):

    words = re.sub(
        r"[^a-z ]",
        " ",
        str(value or "").casefold()
    )

    words = " ".join(words.split())

    aliases = {
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

    return aliases.get(
        words,
        words.title()
    )


# ==========================================================
# 5. LOAD TFLITE MODEL
# ==========================================================

@lru_cache(maxsize=1)
def _load_tflite():

    if not TFLITE_PATH.is_file():
        return None

    import tensorflow as tf

    interpreter = tf.lite.Interpreter(
        model_path=str(TFLITE_PATH)
    )

    interpreter.allocate_tensors()

    return interpreter


# ==========================================================
# 6. LOAD KERAS MODEL
# ==========================================================

@lru_cache(maxsize=1)
def _load_keras():

    if not KERAS_PATH.is_file():
        return None

    import tensorflow as tf

    return tf.keras.models.load_model(
        str(KERAS_PATH),
        compile=False
    )


# ==========================================================
# 7. MODEL STATUS
# ==========================================================

def model_status():

    tflite_exists = TFLITE_PATH.is_file()
    keras_exists = KERAS_PATH.is_file()

    active = (
        TFLITE_PATH
        if tflite_exists
        else KERAS_PATH
    )

    try:
        labels_ok = len(_load_labels()) == 15

    except Exception:
        labels_ok = False

    return {
        "ready": (
            (tflite_exists or keras_exists)
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
            if tflite_exists
            else "Keras"
            if keras_exists
            else "Unavailable"
        )
    }


# ==========================================================
# 8. IDENTIFY CROP USING GEMINI
# ==========================================================

def identify_crop(image):

    key = os.getenv("GEMINI_API_KEY")

    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    from google import genai
    from google.genai import types

    buffer = io.BytesIO()

    ImageOps.exif_transpose(image).convert("RGB").save(
        buffer,
        format="JPEG",
        quality=90
    )

    prompt = """
    Identify the plant or crop in this image.

    Do not identify the disease.

    It may be Tomato, Potato, Pepper,
    Apple, Mango, or another crop.

    Do not assume the crop is supported.

    If identification is uncertain,
    return Unknown.

    Return JSON only:

    {
        "is_plant": true,
        "crop": "Tomato",
        "confidence": 0.9,
        "reason": "short reason"
    }

    The confidence is an approximate,
    uncalibrated model estimate.

    Return a short common crop name,
    not a sentence.
    """

    client = genai.Client(
        api_key=key
    )

    
    response = client.models.generate_content(
    model="gemini-2.5-flash",

    contents=[
        prompt,
        types.Part.from_bytes(
            data=buffer.getvalue(),
            mime_type="image/jpeg"
        )
    ],

    config={
        "response_mime_type": "application/json",
        "max_output_tokens": 350
    }
)


    contents=[
            prompt,

            types.Part.from_bytes(
                data=buffer.getvalue(),
                mime_type="image/jpeg"
            )
        ],

        config={
            "response_mime_type": "application/json",
            "max_output_tokens": 350
        }
    )

    if not response.text:
        raise RuntimeError(
            "Empty Gemini response."
        )

    data = json.loads(
        response.text.strip()
    )

    if not isinstance(data, dict):
        raise ValueError(
            "Gemini returned invalid JSON."
        )

    crop = _canonical_crop(
        data.get("crop")
    )

    try:
        confidence = float(
            data.get("confidence", 0.0)
        )

    except (TypeError, ValueError):
        confidence = 0.0

    if not math.isfinite(confidence):
        confidence = 0.0

    return {
        "is_plant": (
            data.get("is_plant") is True
        ),

        "crop": crop,

        "confidence": float(
            np.clip(confidence, 0, 1)
        ),

        "reason": str(
            data.get("reason", "")
        )
    }


# ==========================================================
# 9. IMAGE PREPROCESSING
# ==========================================================

def _prepare_image(image):

    from tensorflow.keras.applications.mobilenet_v2 import (
        preprocess_input
    )

    image = ImageOps.exif_transpose(image)

    image = image.convert("RGB")

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    pixels = np.asarray(
        image,
        dtype=np.float32
    )

    pixels = preprocess_input(
        pixels
    )

    batch = np.expand_dims(
        pixels,
        axis=0
    )

    return batch


# ==========================================================
# 10. TFLITE INFERENCE
# ==========================================================

def _run_tflite(batch):

    interpreter = _load_tflite()

    if interpreter is None:
        raise FileNotFoundError(
            "TFLite model missing."
        )

    with TFLITE_LOCK:

        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]

        expected = tuple(
            int(i)
            for i in inp["shape"]
        )

        if expected != tuple(batch.shape):
            raise ValueError(
                f"Expected {expected}, got {batch.shape}"
            )

        input_type = np.dtype(
            inp["dtype"]
        )

        if np.issubdtype(input_type, np.integer):

            scale, zero = inp["quantization"]

            if scale <= 0:
                raise ValueError(
                    "Invalid input quantization scale."
                )

            limits = np.iinfo(
                input_type
            )

            data = np.clip(
                np.rint(batch / scale + zero),
                limits.min,
                limits.max
            ).astype(input_type)

        else:
            data = batch.astype(
                input_type
            )

        interpreter.set_tensor(
            inp["index"],
            data
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            out["index"]
        )

        output_type = np.dtype(
            out["dtype"]
        )

        if np.issubdtype(output_type, np.integer):

            scale, zero = out["quantization"]

            if scale <= 0:
                raise ValueError(
                    "Invalid output quantization scale."
                )

            output = (
                output.astype(np.float32) - zero
            ) * scale

        return output


# ==========================================================
# 11. KERAS INFERENCE
# ==========================================================

def _run_keras(batch):

    model = _load_keras()

    if model is None:
        raise FileNotFoundError(
            "Keras model missing."
        )

    return model.predict(
        batch,
        verbose=0
    )


# ==========================================================
# 12. PROCESS MODEL SCORES
# ==========================================================

def _to_scores(output):

    vals = np.asarray(
        output,
        dtype=np.float64
    ).reshape(-1)

    if len(vals) != 15:
        raise ValueError(
            "Expected 15 model outputs."
        )

    if not np.all(np.isfinite(vals)):
        raise ValueError(
            "Invalid model outputs."
        )

    if (
        np.all((vals >= 0) & (vals <= 1))
        and np.isclose(
            vals.sum(),
            1.0,
            atol=0.01
        )
    ):

        return (
            vals,
            "model output, not calibrated"
        )

    # Assumes unnormalized outputs are logits.
    # Verify this against the training model.
    exp = np.exp(
        vals - vals.max()
    )

    scores = exp / exp.sum()

    return (
        scores,
        "softmax-style score, not calibrated"
    )


# ==========================================================
# 13. BLOCK UNSUPPORTED OR UNCERTAIN RESULTS
# ==========================================================

def _blocked(
    status,
    message,
    start,
    **details
):

    return {
        "status": status,
        "error": message,

        "latency_ms": (
            time.perf_counter() - start
        ) * 1000,

        **details
    }


# ==========================================================
# 14. MAIN PLANT DISEASE PREDICTION
# ==========================================================

def predict_leaf_disease(
    image,
    verify_with_gemini=True
):

    start = time.perf_counter()

    try:

        crop_check = None

        # --------------------------------------------------
        # STEP 1: VERIFY CROP WITH GEMINI
        # --------------------------------------------------

        if verify_with_gemini:

            try:
                crop_check = identify_crop(image)

            except Exception:

                LOG.exception(
                    "Online crop verification failed"
                )

                return _blocked(
                    "verification_unavailable",

                    "Online crop verification is unavailable. "
                    "No disease diagnosis has been shown. "
                    "Disable verification only for an "
                    "unverified offline demonstration.",

                    start
                )

            crop = crop_check["crop"]

            crop_score = crop_check["confidence"]

            # Check for non-plant image.
            if not crop_check["is_plant"]:

                return _blocked(
                    "no_plant",
                    "No identifiable plant detected.",
                    start
                )

            # Check uncertain crop.
            if (
                crop == "Unknown"
                or crop_score < 0.60
            ):

                return _blocked(
                    "uncertain_crop",

                    "Crop identification was uncertain. "
                    "Please upload a clearer leaf photograph.",

                    start,

                    crop=crop,
                    crop_confidence=crop_score
                )

            # Reject unsupported crops.
            if crop not in SUPPORTED_CROPS:

                return _blocked(
                    "unsupported_crop",

                    f"Crop identified as {crop}, "
                    "which is not supported. "
                    "Disease classification is available "
                    "only for Tomato, Potato and Pepper. "
                    "No diagnosis shown.",

                    start,

                    crop=crop,
                    crop_confidence=crop_score
                )

        # --------------------------------------------------
        # STEP 2: LOAD MODEL LABELS
        # --------------------------------------------------

        labels = _load_labels()

        # --------------------------------------------------
        # STEP 3: PREPROCESS THE IMAGE
        # --------------------------------------------------

        batch = _prepare_image(image)

        # --------------------------------------------------
        # STEP 4: RUN TFLITE OR KERAS
        # --------------------------------------------------

        fallback_note = None

        if TFLITE_PATH.is_file():

            try:
                output = _run_tflite(batch)
                engine = "TFLite"

            except Exception as exc:

                LOG.exception(
                    "TFLite failed; attempting Keras fallback"
                )

                if not KERAS_PATH.is_file():

                    return _blocked(
                        "inference_failed",

                        "TFLite inference failed and "
                        "the Keras fallback is missing.",

                        start
                    )

                output = _run_keras(batch)

                engine = "Keras"

                fallback_note = (
                    f"TFLite failed "
                    f"({type(exc).__name__}); "
                    "used Keras."
                )

        else:

            output = _run_keras(batch)
            engine = "Keras"

        # --------------------------------------------------
        # STEP 5: GET TOP 3 PREDICTIONS
        # --------------------------------------------------

        scores, score_type = _to_scores(
            output
        )

        order = np.argsort(scores)[::-1][:3]

        top_predictions = []

        for i in order:

            i = int(i)

            label = labels[i]

            top_predictions.append({
                "label": label,

                "display_label": _format_disease(
                    label
                ),

                "confidence": float(
                    scores[i]
                ),

                "class_index": i
            })

        best = top_predictions[0]

        predicted_crop = _get_crop(
            best["label"]
        )

        if predicted_crop == "Unknown":
            raise ValueError(
                "Unexpected class label prefix."
            )

        # --------------------------------------------------
        # STEP 6: VERIFY GEMINI-CNN AGREEMENT
        # --------------------------------------------------

        if crop_check:

            if crop_check["crop"] != predicted_crop:

                return _blocked(
                    "crop_mismatch",

                    "Crop disagreement: Gemini identified "
                    f"{crop_check['crop']}, but the disease "
                    f"model predicted {predicted_crop}. "
                    "Disease diagnosis withheld.",

                    start,

                    crop=crop_check["crop"],

                    classifier_crop=predicted_crop,

                    crop_confidence=(
                        crop_check["confidence"]
                    ),

                    top_predictions=top_predictions,

                    inference_engine=engine
                )

        # --------------------------------------------------
        # STEP 7: CHECK UNCERTAINTY
        # --------------------------------------------------

        score = best["confidence"]

        second_score = (
            top_predictions[1]["confidence"]
        )

        margin = score - second_score

        uncertainty = (
            score < 0.70
            or margin < 0.10
        )

        offline_unverified = (
            not verify_with_gemini
        )

        # --------------------------------------------------
        # STEP 8: RETURN FINAL RESULT
        # --------------------------------------------------

        return {
            "status": "success",

            "label": best["label"],

            "display_label": (
                best["display_label"]
            ),

            "crop": predicted_crop,

            "crop_confidence": (
                crop_check["confidence"]
                if crop_check
                else None
            ),

            "confidence": score,

            "top_predictions": top_predictions,

            "low_confidence": (
                uncertainty or offline_unverified
            ),

            "confidence_message": (
                "Unverified/offline prediction. "
                "This 15-class classifier cannot reliably "
                "reject unsupported plants such as Apple "
                "or Mango."

                if offline_unverified

                else "Prediction uncertain; verify "
                "with an agriculture expert."

                if uncertainty

                else "Crop check agrees, but the model "
                "score does not guarantee correctness."
            ),

            "crop_verification": (
                "match"
                if crop_check
                else "not_requested"
            ),

            "inference_engine": engine,

            "score_type": score_type,

            "top_two_margin": float(margin),

            "fallback_note": fallback_note,

            "latency_ms": (
                time.perf_counter() - start
            ) * 1000
        }

    except Exception:

        LOG.exception(
            "Plant Doctor prediction failed"
        )

        return _blocked(
            "error",

            "Plant Doctor could not complete "
            "this prediction. Check model files, "
            "labels, and application logs.",

            start
        )
