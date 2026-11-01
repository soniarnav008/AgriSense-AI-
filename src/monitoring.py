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
    # These deliberately broad rules are only demo flags. Soil moisture thresholds
    # vary by crop, soil texture, sensor calibration, and growth stage.
    if moisture <= 20 and rainfall < 5:
        alerts.append({
            "level": "High",
            "title": "Low soil-moisture indicator",
            "detail": "The entered moisture value is low and little recent rainfall was reported. Verify the reading, soil type, and crop growth stage before deciding on irrigation."
        })
    elif moisture <= 35 and rainfall < 5:
        alerts.append({
            "level": "Moderate",
            "title": "Review soil moisture",
            "detail": "Consider checking the root-zone moisture and crop-specific irrigation guidance. This generic threshold is not a crop-specific prescription."
        })
    if temperature >= 38:
        alerts.append({
            "level": "High",
            "title": "High-temperature indicator",
            "detail": "The entered temperature is high. Check local weather guidance and observe the crop for heat stress."
        })
    elif temperature >= 33:
        alerts.append({
            "level": "Moderate",
            "title": "Warm-condition indicator",
            "detail": "Warm conditions may increase water demand for some crops. Interpret this with crop, soil, and growth-stage information."
        })
    if humidity >= 85:
        alerts.append({
            "level": "Moderate",
            "title": "High-humidity indicator",
            "detail": "Prolonged high humidity can favor some diseases, depending on crop and leaf-wetness conditions. Inspect plants and consult crop-specific guidance."
        })
    if rainfall >= 25:
        alerts.append({
            "level": "Moderate",
            "title": "Recent rainfall reported",
            "detail": "Account for rainfall and drainage before making irrigation decisions; the effective amount reaching the root zone may vary."
        })
    return {"inputs": inputs, "alerts": alerts}
