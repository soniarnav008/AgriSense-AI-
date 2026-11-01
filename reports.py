import json

def make_markdown_report(records):
    lines = [
        "# AgriSense AI — Assessment History",
        "",
        "> Prototype report. User-entered data and model predictions are not independently verified.",
        "",
    ]
    for i, item in enumerate(records, start=1):
        lines.extend([
            f"## Assessment {i}",
            f"- **Date:** {item['created_at']}",
            f"- **Type:** {item['record_type']}",
            f"- **Crop:** {item['crop']}",
            "",
            "```json",
            json.dumps(item["payload"], indent=2, ensure_ascii=False),
            "```",
            "",
        ])
    return "\n".join(lines)
