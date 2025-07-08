import os
import uuid
import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify, session
from openai import OpenAI
import PyPDF2
from langdetect import detect, DetectorFactory
from autogen import UserProxyAgent, AssistantAgent, register_function
from mood_storage import log_mood, get_mood_history

DetectorFactory.seed = 0

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "supersecret")

# OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ——— Define AutoGen Agents ——————————————————————————————

user_proxy = UserProxyAgent(name="UserProxyAgent", human_input_mode="ALWAYS")

mood_tracker = AssistantAgent(
    name="MoodTrackerAgent",
    llm_config={"model": "gpt-4", "temperature": 0.3},
    system_message=(
        "You are MoodTrackerAgent. Ask the user for mood ratings or emotions, "
        "use store_mood() to log them, and retrieve_mood_history() to report trends."
    ),
    code_execution_config={"use_functions": True}
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

# ——— Define tool functions ——————————————————————————————

def store_mood(mood: str, user_id: str = "default_user") -> str:
    """Logs a mood entry for the user."""
    log_mood(user_id, mood)
    return f"✅ Logged mood: {mood}"

def retrieve_mood_history(user_id: str = "default_user") -> str:
    """Fetches and summarizes the user’s mood history."""
    history = get_mood_history(user_id)
    if not history:
        return "No mood history found."
    lines = [f"{e['timestamp'][:10]}: {e['mood']}" for e in history]
    return "📊 Your mood history:\n" + "\n".join(lines)

# ——— Register tool functions with the MoodTrackerAgent ————————

register_function(
    caller=mood_tracker,
    executor=store_mood,
    description="Log the user's mood entry"
)

register_function(
    caller=mood_tracker,
    executor=retrieve_mood_history,
    description="Retrieve the user's mood history"
)

# ——— Session storage for proxy agents ——————————————————————

user_sessions = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    # Session management for multi-agent state
    sid = session.get("sid")
    if not sid:
        sid = str(uuid.uuid4())
        session["sid"] = sid
    if sid not in user_sessions:
        user_sessions[sid] = user_proxy
    proxy = user_sessions[sid]

    # Gather inputs & flags
    text_input    = request.form.get("text", "").strip()
    image_file    = request.files.get("image")
    video_file    = request.files.get("video")
    audio_file    = request.files.get("audio")
    document_file = request.files.get("document")
    output_type   = request.form.get("output_type", "text")
    restrict      = request.form.get("restrict_scope") == "on"
    mh_mode       = request.form.get("mh_mode") == "on"

    # 1) Mental-Health Multi-Agent Mode
    if mh_mode and text_input:
        try:
            reply = proxy.send(
                user_message=text_input,
                recipients=[companion_agent, mood_tracker, suggestion_agent]
            )
            bot_text = reply if isinstance(reply, str) else reply.message
            try:
                lang = detect(bot_text)
            except:
                lang = "en"
            return jsonify({"response": bot_text, "language": lang})
        except Exception as e:
            return jsonify({"error": f"MultiAgent error: {e}"}), 500

    # 2) Otherwise: your existing multimodal pipeline
    # (Image, video/audio whisper, document parsing, text-only chat with TTS & image-gen)
    # — IMAGE input —
    if image_file:
        content = []
        if text_input:
            content.append({"type": "text", "text": text_input})
        else:
            content.append({"type": "text", "text": "What's in this image?"})
        img_bytes = image_file.read()
        b64 = base64.b64encode(img_bytes).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{image_file.content_type};base64,{b64}", "detail": "auto"}
        })
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": content}],
                max_tokens=300
            )
            bot_text = resp.choices[0].message.content.strip()
        except Exception as e:
            return jsonify({"error": "Vision API failed: " + str(e)}), 500

        if output_type == "speech":
            fn = f"static/audio_{uuid.uuid4().hex}.mp3"
            tts = client.audio.speech.create(model="tts-1", voice="alloy", input=bot_text)
            tts.stream_to_file(fn)
            return jsonify({"response": bot_text, "audio_url": fn})
        if output_type == "image":
            img = client.images.generate(
                model="dall-e-3",
                prompt=bot_text,
                size="1024x1024",
                quality="standard",
                n=1
            )
            return jsonify({"response": "Image generated", "image_url": img.data[0].url})
        return jsonify({"response": bot_text})

    # VIDEO / AUDIO / DOCUMENT / TEXT handling...
    # [Insert your existing code here as before]

    return jsonify({"error": "No input provided"}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
