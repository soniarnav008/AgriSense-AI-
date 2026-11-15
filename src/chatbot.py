
import os
import time
from dotenv import load_dotenv

load_dotenv()

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
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return (
            "Gemini API key is not configured. Add GEMINI_API_KEY "
            "to your local .env file, then restart the app."
        )

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        # Prepare previous conversation history.
        contents = []

        for item in (history or [])[-8:]:
            role = item.get("role")
            content = item.get("content", "")

            if role == "user" and content:
                contents.append({
                    "role": "user",
                    "parts": [{"text": content}],
                })

            elif role == "assistant" and content:
                contents.append({
                    "role": "model",
                    "parts": [{"text": content}],
                })

        # Include the latest farm assessment with the question.
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
            + question
        )

        contents.append({
            "role": "user",
            "parts": [{"text": user_input}],
        })

        # Retry temporary API failures up to 3 total attempts.
        max_retries = 3
        response = None

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model="gemini-flash-lite-latest",
                    contents=contents,
                    config={
                        "system_instruction": SYSTEM_INSTRUCTIONS,
                        "max_output_tokens": 1500,
                    },
                )
                break

            except Exception as exc:
                error_text = str(exc).upper()

                is_temporary = (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                    or "500" in error_text
                    or "502" in error_text
                    or "504" in error_text
                    or "INTERNAL" in error_text
                )

                # Do not retry non-temporary errors.
                if not is_temporary:
                    raise

                # Stop after the final attempt.
                if attempt == max_retries - 1:
                    raise

                # Wait 2 seconds, then 4 seconds before retrying.
                wait_seconds = 2 ** (attempt + 1)

                print(
                    f"Gemini temporary error: {exc}. "
                    f"Retrying in {wait_seconds} seconds "
                    f"(attempt {attempt + 1}/{max_retries})."
                )

                time.sleep(wait_seconds)

        # Safely handle an empty response.
        if response is None or not response.text:
            return (
                "The AI assistant returned an empty response. "
                "Please try asking your question again."
            )

        return response.text.strip()

    except Exception as exc:
        print("Gemini error details:", repr(exc))

        return (
            f"Gemini error: {type(exc).__name__}. "
            "Please try again later. Check the VS Code terminal "
            "for details if the problem continues."
        )
