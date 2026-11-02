
from datetime import datetime, timedelta


# ---------------------------------------------------------
# CROP KNOWLEDGE BASE
# ---------------------------------------------------------

CROP_GUIDANCE = {
    "Tomato": {
        "water": "Check root-zone moisture regularly. Avoid waterlogging and wetting leaves.",
        "inspection": "Inspect leaves for early blight, late blight, leaf spots and pest damage.",
        "prevention": "Maintain airflow, remove infected plant debris and avoid overhead irrigation.",
        "heat": "Monitor for wilting and heat stress during hot afternoons.",
    },
    "Potato": {
        "water": "Maintain consistent soil moisture, especially during tuber development.",
        "inspection": "Inspect foliage for early blight, late blight and unusual yellowing.",
        "prevention": "Use healthy planting material and remove infected plant debris.",
        "heat": "Monitor soil moisture and avoid unnecessary water stress.",
    },
    "Pepper": {
        "water": "Maintain consistent root-zone moisture without waterlogging.",
        "inspection": "Check leaves for bacterial spots, wilting and pest damage.",
        "prevention": "Improve airflow and avoid handling wet plants.",
        "heat": "Monitor flowers and leaves for heat-related stress.",
    },
    "Apple": {
        "water": "Check soil moisture around the root zone and adjust irrigation to tree age.",
        "inspection": "Inspect leaves and fruit for spots, mildew and pest damage.",
        "prevention": "Remove fallen infected leaves and maintain canopy airflow.",
        "heat": "Monitor young trees and fruit for heat and water stress.",
    },
    "Rice": {
        "water": "Manage field water according to crop stage and local agricultural guidance.",
        "inspection": "Inspect for leaf blast, bacterial leaf blight and pest damage.",
        "prevention": "Use balanced nutrients and avoid excessive nitrogen application.",
        "heat": "Monitor water availability during hot weather.",
    },
    "Wheat": {
        "water": "Schedule irrigation according to growth stage and soil moisture.",
        "inspection": "Check for rust, powdery mildew and unusual leaf discoloration.",
        "prevention": "Use locally recommended varieties and balanced fertilization.",
        "heat": "Monitor moisture during sensitive flowering and grain-filling stages.",
    },
    "Maize": {
        "water": "Check root-zone moisture, especially around tasseling and silking.",
        "inspection": "Inspect leaves and stems for leaf blight and pest damage.",
        "prevention": "Remove heavily infected residues and follow crop rotation guidance.",
        "heat": "Monitor for leaf rolling and moisture stress.",
    },
    "Cotton": {
        "water": "Monitor root-zone moisture and avoid excessive irrigation.",
        "inspection": "Inspect for sucking pests, bollworms and leaf discoloration.",
        "prevention": "Use integrated pest management and monitor pest populations.",
        "heat": "Check for wilting and moisture stress during hot periods.",
    },
    "Grape": {
        "water": "Monitor root-zone moisture and avoid excessive canopy wetness.",
        "inspection": "Inspect leaves and bunches for powdery mildew and downy mildew.",
        "prevention": "Maintain canopy ventilation and remove infected plant material.",
        "heat": "Monitor vines and fruit for heat stress.",
    },
    "Soybean": {
        "water": "Monitor soil moisture during flowering and pod development.",
        "inspection": "Inspect for rust, leaf spots and insect damage.",
        "prevention": "Use crop rotation and locally recommended disease management.",
        "heat": "Monitor moisture during flowering and pod filling.",
    },
}


# ---------------------------------------------------------
# EXISTING FARM MONITORING RULES
# Kept compatible with the existing app.py
# ---------------------------------------------------------

def assess_conditions(crop, moisture, temperature, humidity, rainfall, notes=""):
    """Transparent demo screening rules, not a trained predictive model."""

    inputs = {
        "crop": crop,
        "soil_moisture_percent": int(moisture),
        "temperature_c": int(temperature),
        "relative_humidity_percent": int(humidity),
        "recent_rainfall_mm": float(rainfall),
        "field_notes": notes,
    }

    alerts = []

    if moisture <= 20 and rainfall < 5:
        alerts.append({
            "level": "High",
            "title": "Low soil-moisture indicator",
            "detail": (
                "The entered moisture value is low and little recent rainfall "
                "was reported. Verify the reading, soil type and crop stage."
            )
        })

    elif moisture <= 35 and rainfall < 5:
        alerts.append({
            "level": "Moderate",
            "title": "Review soil moisture",
            "detail": (
                "Check root-zone moisture and crop-specific irrigation guidance."
            )
        })

    if temperature >= 38:
        alerts.append({
            "level": "High",
            "title": "High-temperature indicator",
            "detail": (
                "Check local weather guidance and observe the crop for heat stress."
            )
        })

    elif temperature >= 33:
        alerts.append({
            "level": "Moderate",
            "title": "Warm-condition indicator",
            "detail": (
                "Warm conditions may increase water demand for some crops."
            )
        })

    if humidity >= 85:
        alerts.append({
            "level": "Moderate",
            "title": "High-humidity indicator",
            "detail": (
                "Prolonged high humidity can favor some diseases. "
                "Inspect plants and leaf-wetness conditions."
            )
        })

    if rainfall >= 25:
        alerts.append({
            "level": "Moderate",
            "title": "Recent rainfall reported",
            "detail": (
                "Consider rainfall and drainage before making irrigation decisions."
            )
        })

    return {
        "inputs": inputs,
        "alerts": alerts
    }


# ---------------------------------------------------------
# RISK SCORE
# ---------------------------------------------------------

def calculate_risk(alerts):
    """A demo screening score, not a probability of crop failure."""

    score = 0

    for alert in alerts:
        if alert.get("level") == "High":
            score += 30
        elif alert.get("level") == "Moderate":
            score += 15

    score = min(score, 100)

    if score >= 60:
        level = "High"
    elif score >= 30:
        level = "Moderate"
    else:
        level = "Low"

    return {
        "score": score,
        "level": level
    }


# ---------------------------------------------------------
# TREATMENT AND PREVENTION PLANNER
# ---------------------------------------------------------

def generate_recommendations(crop, inputs, alerts):

    guidance = CROP_GUIDANCE.get(crop, {
        "water": "Check soil moisture and follow local crop-specific irrigation guidance.",
        "inspection": "Inspect leaves, stems and roots for unusual symptoms.",
        "prevention": "Maintain field hygiene and monitor crop health regularly.",
        "heat": "Observe the crop for signs of heat stress.",
    })

    recommendations = []

    moisture = inputs["soil_moisture_percent"]
    temperature = inputs["temperature_c"]
    humidity = inputs["relative_humidity_percent"]
    rainfall = inputs["recent_rainfall_mm"]

    if moisture <= 20 and rainfall < 5:
        recommendations.append({
            "priority": "High",
            "title": "Review irrigation requirement",
            "action": guidance["water"]
        })

    elif moisture <= 35:
        recommendations.append({
            "priority": "Moderate",
            "title": "Check root-zone moisture",
            "action": guidance["water"]
        })

    if temperature >= 33:
        recommendations.append({
            "priority": "High" if temperature >= 38 else "Moderate",
            "title": "Heat-stress monitoring",
            "action": guidance["heat"]
        })

    if humidity >= 85:
        recommendations.append({
            "priority": "Moderate",
            "title": "Disease prevention inspection",
            "action": guidance["inspection"] + " " + guidance["prevention"]
        })

    if rainfall >= 25:
        recommendations.append({
            "priority": "Moderate",
            "title": "Post-rainfall field inspection",
            "action": (
                "Check field drainage, standing water and root-zone moisture "
                "before deciding on further irrigation."
            )
        })

    if not recommendations:
        recommendations.append({
            "priority": "Normal",
            "title": "Routine crop maintenance",
            "action": guidance["inspection"] + " " + guidance["prevention"]
        })

    return recommendations


# ---------------------------------------------------------
# CROP-CARE CALENDAR
# ---------------------------------------------------------

def generate_care_calendar(crop):

    guidance = CROP_GUIDANCE.get(crop, {
        "water": "Check soil moisture and follow local crop guidance.",
        "inspection": "Inspect leaves, stems and roots.",
        "prevention": "Maintain field hygiene and monitor crop health.",
        "heat": "Observe for heat stress.",
    })

    today = datetime.now().date()

    tasks = [
        {
            "day": 0,
            "task": "Current field inspection",
            "description": guidance["inspection"]
        },
        {
            "day": 1,
            "task": "Soil moisture review",
            "description": guidance["water"]
        },
        {
            "day": 3,
            "task": "Preventive crop care",
            "description": guidance["prevention"]
        },
        {
            "day": 5,
            "task": "Weather and heat-stress check",
            "description": guidance["heat"]
        },
        {
            "day": 7,
            "task": "Weekly crop health review",
            "description": "Compare crop appearance and field conditions with the previous inspection."
        }
    ]

    calendar = []

    for item in tasks:
        date = today + timedelta(days=item["day"])

        calendar.append({
            "date": date.isoformat(),
            "task": item["task"],
            "description": item["description"]
        })

    return calendar


# ---------------------------------------------------------
# HISTORICAL CROP INTELLIGENCE
# ---------------------------------------------------------

def analyze_history(crop, records):

    history = []

    for record in records:
        if record.get("record_type") != "farm_conditions":
            continue

        if record.get("crop", "").lower() != crop.lower():
            continue

        payload = record.get("payload", {})
        inputs = payload.get("inputs", {})

        if not inputs:
            continue

        history.append({
            "created_at": record.get("created_at"),
            "moisture": inputs.get("soil_moisture_percent"),
            "temperature": inputs.get("temperature_c"),
            "humidity": inputs.get("relative_humidity_percent"),
            "rainfall": inputs.get("recent_rainfall_mm"),
            "alert_count": len(payload.get("alerts", []))
        })

    if not history:
        return {
            "available": False,
            "message": "No previous farm assessments found for this crop.",
            "records": []
        }

    moisture_values = [
        item["moisture"] for item in history
        if item["moisture"] is not None
    ]

    temperature_values = [
        item["temperature"] for item in history
        if item["temperature"] is not None
    ]

    humidity_values = [
        item["humidity"] for item in history
        if item["humidity"] is not None
    ]

    def average(values):
        return round(sum(values) / len(values), 1) if values else None

    return {
        "available": True,
        "total_assessments": len(history),
        "average_moisture": average(moisture_values),
        "average_temperature": average(temperature_values),
        "average_humidity": average(humidity_values),
        "records": history
    }


# ---------------------------------------------------------
# MAIN SMART CROP ADVISOR ENGINE
# ---------------------------------------------------------

def build_crop_advisor(assessment, records=None):

    if records is None:
        records = []

    inputs = assessment.get("inputs", {})
    alerts = assessment.get("alerts", [])

    crop = inputs.get("crop", "Unknown")

    risk = calculate_risk(alerts)

    recommendations = generate_recommendations(
        crop,
        inputs,
        alerts
    )

    calendar = generate_care_calendar(crop)

    history = analyze_history(crop, records)

    return {
        "crop": crop,
        "risk": risk,
        "alerts": alerts,
        "recommendations": recommendations,
        "calendar": calendar,
        "history": history,
        "assessment": assessment
    }
