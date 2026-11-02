
import json
import time
import os
import io
import re
from pathlib import Path
from functools import lru_cache

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

TFLITE_PATH = ROOT / "models" / "plant_disease_model_15class.tflite"
MODEL_PATH = ROOT / "models" / "plant_disease_model_15class.keras"
LABELS_PATH = ROOT / "models" / "labels_15class.json"

IMAGE_SIZE = 160

SUPPORTED_CROPS = {"Tomato", "Potato", "Pepper"}


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


@lru_cache(maxsize=1)
def _load_model():
    if not MODEL_PATH.exists():
        return None

    import tensorflow as tf

    return tf.keras.models.load_model(
        MODEL_PATH, compile=False
    )


def _load_labels():
    if not LABELS_PATH.exists():
        return None

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_crop(label):
    label = str(label)

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


def identify_crop(image):
    """
    Identify the crop using Gemini Vision.
    This does NOT diagnose plant disease.
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    image_buffer = io.BytesIO()
    image.convert("RGB").save(image_buffer, format="JPEG")

    image_part = types.Part.from_bytes(
        data=image_buffer.getvalue(),
        mime_type="image/jpeg"
    )

    prompt = """
    You are a plant and crop identification assistant.

    Examine the uploaded image carefully.

    Identify the most likely crop or plant visible.

    You may identify crops worldwide, including:
    Apple, Rice, Wheat, Maize, Cotton, Grape, Potato,
    Tomato, Pepper, Mango, Banana, Citrus, Soybean,
    and other plants.

    Rules:
    1. Do not assume the image belongs to Tomato, Potato or Pepper.
    2. Do not identify a crop if there is insufficient visual evidence.
    3. If the image does not contain a recognizable plant, say Unknown.
    4. Identify the plant, not its disease.
    5. Return only valid JSON.

    Required JSON format:

    {
      "is_plant": true,
      "crop": "Apple",
      "confidence": 0.90,
      "reason": "Visible leaf characteristics suggest an apple plant."
    }

    Confidence is an approximate model estimate, not a calibrated probability.
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
        raise RuntimeError("Gemini returned an empty crop identification.")

    raw = response.text.strip()

    # Remove accidental Markdown code fences.
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    result = json.loads(raw)

    crop = str(result.get("crop", "Unknown")).strip()
    is_plant = bool(result.get("is_plant", False))

    try:
        confidence = float(result.get("confidence", 0))
    except (ValueError, TypeError):
        confidence = 0

    return {
        "is_plant": is_plant,
        "crop": crop,
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": str(result.get("reason", ""))
    }


def model_status():
    tflite_exists = TFLITE_PATH.exists()
    keras_exists = MODEL_PATH.exists()

    active_path = TFLITE_PATH if tflite_exists else MODEL_PATH

    size_mb = (
        active_path.stat().st_size / (1024 * 1024)
        if active_path.exists() else 0
    )

    return {
        "ready": (tflite_exists or keras_exists)
                 and LABELS_PATH.exists(),
        "size_mb": size_mb,
        "model_path": str(active_path),
        "labels_path": str(LABELS_PATH),
        "inference_engine": "TFLite" if tflite_exists else "Keras"
    }


def predict_leaf_disease(image):
    start = time.perf_counter()

    try:
        # STEP 1: Identify the crop using Gemini Vision.
        crop_result = identify_crop(image)

        identified_crop = crop_result["crop"]
        crop_confidence = crop_result["confidence"]

        if not crop_result["is_plant"]:
            return {
                "error": "No recognizable plant was detected. "
                         "Please upload a clear plant or leaf image."
            }

        if identified_crop.lower() == "unknown" or crop_confidence < 0.60:
            return {
                "status": "uncertain",
                "crop": "Unknown",
                "crop_confidence": crop_confidence,
                "message": "Crop identification is uncertain. "
                           "Please upload a clearer image."
            }

        # STEP 2: Unsupported crop.
        if identified_crop not in SUPPORTED_CROPS:
            return {
                "status": "crop_identified",
                "crop": identified_crop,
                "crop_confidence": crop_confidence,
                "message": (
                    f"Identified crop: {identified_crop}. "
                    "Disease detection for this crop is not "
                    "currently supported by our trained model."
                ),
                "inference_engine": "Gemini Vision",
                "latency_ms": (
                    time.perf_counter() - start
                ) * 1000
            }

        # STEP 3: Run existing disease model for supported crops.
        import tensorflow as tf

        labels = _load_labels()

        if labels is None:
            return {"error": "15-class model labels are missing."}

        image = image.convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        array = np.asarray(image, dtype=np.float32)

        array = tf.keras.applications.mobilenet_v2.preprocess_input(
            array
        )

        batch = np.expand_dims(array, axis=0)

        engine = "TFLite"

        try:
            interpreter = _load_tflite()

            if interpreter is None:
                raise RuntimeError("TFLite model unavailable")

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            interpreter.set_tensor(
                input_details[0]["index"], batch
            )

            interpreter.invoke()

            output = interpreter.get_tensor(
                output_details[0]["index"]
            )

        except Exception as tflite_error:
            print("TFLite unavailable, using Keras:", tflite_error)

            engine = "Keras"
            model = _load_model()

            if model is None:
                return {
                    "error": "Both TFLite and Keras models "
                             "are unavailable."
                }

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
        predicted_crop = _get_crop(predicted_label)

        # STEP 4: Check whether both systems agree on crop.
        if predicted_crop != identified_crop:
            return {
                "status": "crop_mismatch",
                "crop": identified_crop,
                "crop_confidence": crop_confidence,
                "message": (
                    "Crop identification and disease classifier "
                    "disagree. No disease prediction is being shown."
                ),
                "classifier_crop": predicted_crop,
                "inference_engine": engine,
                "latency_ms": (
                    time.perf_counter() - start
                ) * 1000
            }

        confidence = top[0]["confidence"]

        return {
            "status": "success",
            "label": predicted_label,
            "display_label": top[0]["display_label"],
            "crop": identified_crop,
            "crop_confidence": crop_confidence,
            "confidence": confidence,
            "top_predictions": top,
            "low_confidence": confidence < 0.70,
            "confidence_message": (
                "Low confidence! Please upload a clearer leaf image "
                "or consult an agriculture expert."
                if confidence < 0.70
                else "The model has relatively high confidence "
                     "in this prediction."
            ),
            "inference_engine": engine,
            "latency_ms": (
                time.perf_counter() - start
            ) * 1000
        }

    except Exception as exc:
        print("Plant Doctor error:", repr(exc))

        return {
            "error": (
                f"Plant identification failed: {type(exc).__name__}. "
                "Please try again. Check the terminal for details."
            )
        }
