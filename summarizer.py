import json
import os

def summarize_events(json_file):
    if not os.path.exists(json_file):
        return "No events detected."

    with open(json_file, "r") as f:
        events = json.load(f)

    if not events:
        return "No objects detected in the video."

    summary_lines = [f"{i+1}. {e['count']} {e['object']}(s) detected" for i, e in enumerate(events)]
    return "Summary of detected objects:\n" + "\n".join(summary_lines)
