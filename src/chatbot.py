
import os
import time
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTIONS = """
You are AgriSense AI, an agricultural information assistant.

Be helpful, specific, and explain recommendations clearly.
Use the supplied context when discussing the user's farm or image result.
Do not claim that an image classifier's output is a confirmed diagnosis.
If evidence is insufficient, ask for additional information.
Distinguish general educational information from crop-specific
or location-specific advice.

For pesticide selection, dosage, or disease-control decisions,
recommend checking the product label and consulting a qualified
local agricultural expert or extension service.

Do not invent citations, sensor readings, test results, or capabilities.
Farm readings may be manually entered and are not verified live measurements.
You may answer in Hindi or English based on the user's language.
"""


def ask_farmer_assistant(question, farm_context="", history=None):

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        return (
            "Gemini API key is not configured. "
            "Please add GEMINI_API_KEY to your "
            "Render Environment or local .env file."
        )

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        # --------------------------------------------------
        # 1. PREPARE CONVERSATION HISTORY
        # --------------------------------------------------

        contents = []

        for item in (history or [])[-8:]:

            role = item.get("role")
            content = item.get("content", "")

            if not isinstance(content, str) or not content:
                continue

            if role == "user":
                contents.append({
                    "role": "user",
                    "parts": [{"text": content}],
                })

            elif role == "assistant":
                contents.append({
                    "role": "model",
                    "parts": [{"text": content}],
                })

        # --------------------------------------------------
        # 2. PREPARE USER QUESTION
        # --------------------------------------------------

        user_input = (
            "You are AgriSense AI, a helpful agricultural assistant. "
            "Give clear, practical answers about crop health, plant "
            "symptoms, irrigation, soil, and farming practices. "
            "Ask follow-up questions when important details are missing. "
            "Never pretend physical sensors are connected. "
            "Do not claim a disease diagnosis is certain from limited "
            "information. For pesticide decisions, recommend following "
            "product labels and consulting a qualified local expert.\n\n"
            "Farm context (may contain user-entered values, not verified "
            "measurements):\n"
            + (farm_context or "No farm context supplied.")
            + "\n\nCurrent question:\n"
            + str(question)
        )

        contents.append({
            "role": "user",
            "parts": [{"text": user_input}],
        })

        # --------------------------------------------------
        # 3. GEMINI MODEL CONFIGURATION
        # --------------------------------------------------

        model_name = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.5-flash-lite"
        ).strip()

        # --------------------------------------------------
        # 4. SEND GEMINI REQUEST
        # --------------------------------------------------

        max_attempts = 3

        for attempt in range(max_attempts):

            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config={
                        "system_instruction": SYSTEM_INSTRUCTIONS,
                        "max_output_tokens": 1500,
                    },
                )

                if not response or not response.text:
                    return (
                        "The AI assistant returned an empty response. "
                        "Please try again."
                    )

                return response.text.strip()

            except Exception as exc:

                error_text = str(exc).upper()

                logger.exception(
                    "Gemini request failed using model %s",
                    model_name
                )

                # ------------------------------------------
                # DAILY QUOTA / RATE LIMIT
                # ------------------------------------------

                if (
                    "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                ):

                    return (
                        "Gemini API quota or rate limit has been reached. "
                        "Please check your usage and limits in Google AI Studio. "
                        "Daily quota exhaustion cannot be fixed by immediately "
                        "retrying the request."
                    )

                # ------------------------------------------
                # MODEL NOT AVAILABLE
                # ------------------------------------------

                if (
                    "404" in error_text
                    or "NOT_FOUND" in error_text
                ):

                    return (
                        f"Gemini model '{model_name}' is unavailable "
                        "for this API key or endpoint. "
                        "Check the model identifier and access "
                        "in Google AI Studio."
                    )

                # ------------------------------------------
                # INVALID REQUEST
                # ------------------------------------------

                if "400" in error_text:

                    return (
                        "Gemini rejected the request (400). "
                        "Check the request configuration and Render Logs."
                    )

                # ------------------------------------------
                # PERMISSION ERROR
                # ------------------------------------------

                if (
                    "401" in error_text
                    or "403" in error_text
                ):

                    return (
                        "Gemini API authentication or permission failed. "
                        "Check GEMINI_API_KEY and API access."
                    )

                # ------------------------------------------
                # TEMPORARY SERVER FAILURE
                # ------------------------------------------

                temporary_error = any(
                    code in error_text
                    for code in (
                        "500",
                        "502",
                        "503",
                        "504",
                        "UNAVAILABLE",
                        "INTERNAL"
                    )
                )

                if temporary_error and attempt < max_attempts - 1:

                    wait_seconds = 2 ** (attempt + 1)

                    logger.warning(
                        "Temporary Gemini failure. "
                        "Retrying in %s seconds.",
                        wait_seconds
                    )

                    time.sleep(wait_seconds)
                    continue

                if temporary_error:
                    return (
                        "Gemini is temporarily unavailable or experiencing "
                        "high demand. Please try again later."
                    )

                # ------------------------------------------
                # OTHER ERROR
                # ------------------------------------------

                return (
                    f"Gemini request failed: {type(exc).__name__}. "
                    "Please check the Render Logs for details."
                )

    except Exception as exc:

        logger.exception(
            "Farmer Assistant initialization failed"
        )

        return (
            f"Farmer Assistant error: {type(exc).__name__}. "
            "Check the Render Logs."
        )
