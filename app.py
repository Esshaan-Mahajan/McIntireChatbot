import os
# Disable Docker for AutoGen on Heroku
os.environ["AUTOGEN_USE_DOCKER"] = "false"

import uuid
import base64
import traceback
from flask import Flask, render_template, request, jsonify, session
from openai import OpenAI
import PyPDF2
from langdetect import detect, DetectorFactory
from autogen import UserProxyAgent, AssistantAgent, register_function
from mood_storage import log_mood, get_mood_history

# Seed langdetect for reproducibility
DetectorFactory.seed = 0

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "supersecret")

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ——— Define AutoGen Agents ——————————————————————————————

user_proxy = UserProxyAgent(
    name="UserProxyAgent",
    human_input_mode="ALWAYS"
)

mood_tracker = AssistantAgent(
    name="MoodTrackerAgent",
    llm_config={"model": "gpt-4", "temperature": 0.3},
    system_message=(
        "You are MoodTrackerAgent. Ask the user for mood ratings or emotions, "
        "use store_mood() to log them, and retrieve_mood_history() to report trends."
    ),
    code_execution_config={"use_functions": True, "use_docker": False}
)

suggestion_agent = AssistantAgent(
    name="SuggestionAgent",
    llm_config={"model": "gpt-4", "temperature": 0.7},
    system_message=(
        "You are SuggestionAgent, a wellness coach. Suggest activities "
        "based on the user's recent mood entries."
    )
)

companion_agent = AssistantAgent(
    name="CompanionAgent",
    llm_config={"model": "gpt-4", "temperature": 0.8},
    system_message=(
        "You are CompanionAgent, a caring and empathetic friend. "
        "Engage warmly and respond like a supportive companion."
    )
)

# ——— Tool Functions ——————————————————————————————

def store_mood(mood: str, user_id: str = "default_user") -> str:
    log_mood(user_id, mood)
    return f"✅ Logged mood: {mood}"

def retrieve_mood_history(user_id: str = "default_user") -> str:
    history = get_mood_history(user_id)
    if not history:
        return "No mood history found."
    lines = [f"{e['timestamp'][:10]}: {e['mood']}" for e in history]
    return "📊 Your mood history:\n" + "\n".join(lines)

# ——— Register Functions with MoodTrackerAgent ——————————————————

register_function(
    store_mood,
    caller=mood_tracker,
    executor=mood_tracker,
    description="Log the user's mood entry"
)

register_function(
    retrieve_mood_history,
    caller=mood_tracker,
    executor=mood_tracker,
    description="Retrieve the user's mood history"
)

# ——— Session Store ——————————————————————————

user_sessions = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Persist or create session
        sid = session.get("sid")
        if not sid:
            sid = str(uuid.uuid4())
            session["sid"] = sid
        proxy = user_sessions.setdefault(sid, user_proxy)

        # Inputs & flags
        text_input    = request.form.get("text", "").strip()
        image_file    = request.files.get("image")
        video_file    = request.files.get("video")
        audio_file    = request.files.get("audio")
        document_file = request.files.get("document")
        output_type   = request.form.get("output_type", "text")
        mh_mode       = request.form.get("mh_mode") == "on"

        # 1) Mental-Health Multi-Agent Mode
        if mh_mode and text_input:
            # Send to each agent in turn
            comp = proxy.send(text_input, recipient=companion_agent)
            mood = proxy.send(text_input, recipient=mood_tracker)
            sugg = proxy.send(text_input, recipient=suggestion_agent)

            # Extract text
            def extract(r):
                return r if isinstance(r, str) else r.message

            replies = {
                "Companion": extract(comp),
                "MoodTracker": extract(mood),
                "Suggestion": extract(sugg)
            }

            # Combine into one string
            bot_text = "\n\n".join(f"**{role}:** {txt}" for role, txt in replies.items())

            # Detect language
            lang = "en"
            try:
                lang = detect(bot_text)
            except:
                pass

            return jsonify({"response": bot_text, "language": lang})

        # 2) Multimodal Pipeline (image, video, audio, document, text)...

        # (rest of your existing multimodal code goes here,
        # unchanged from previous version)

        # --- Fallback multilingual text chat ---
        # ...

        return jsonify({"response": bot_text})

    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": "Server error: " + str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
