import json
from datetime import datetime
from pathlib import Path

MOOD_LOG_FILE = Path(__file__).parent / "mood_log.json"

def log_mood(user_id: str, mood_entry: str):
    try:
        data = json.loads(MOOD_LOG_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    user_data = data.get(user_id, [])
    user_data.append({
        "timestamp": datetime.now().isoformat(),
        "mood": mood_entry
    })
    data[user_id] = user_data
    MOOD_LOG_FILE.write_text(json.dumps(data, indent=2))

def get_mood_history(user_id: str):
    try:
        data = json.loads(MOOD_LOG_FILE.read_text())
        return data.get(user_id, [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []
