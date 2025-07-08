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
        # Manage per-user proxy state
        sid = session.get("sid")
        if not sid:
            sid = str(uuid.uuid4())
            session["sid"] = sid
        proxy = user_sessions.setdefault(sid, user_proxy)

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
            # Pass agents as a tuple to avoid unhashable-list errors
            reply = proxy.send(
                text_input,
                recipient=(companion_agent, mood_tracker, suggestion_agent)
            )
            bot_text = reply if isinstance(reply, str) else reply.message
            lang = "en"
            try:
                lang = detect(bot_text)
            except:
                pass
            return jsonify({"response": bot_text, "language": lang})

        # 2) Multimodal Pipeline

        # IMAGE input
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
                "image_url": {
                    "url": f"data:{image_file.content_type};base64,{b64}",
                    "detail": "auto"
                }
            })
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": content}],
                max_tokens=300
            )
            bot_text = resp.choices[0].message.content.strip()

            if output_type == "speech":
                fn = f"static/audio_{uuid.uuid4().hex}.mp3"
                tts = client.audio.speech.create(
                    model="tts-1", voice="alloy", input=bot_text
                )
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
                return jsonify({
                    "response": "Image generated",
                    "image_url": img.data[0].url
                })
            return jsonify({"response": bot_text})

        # VIDEO → Whisper
        if video_file:
            ft = (video_file.filename, video_file.stream, video_file.content_type)
            transcription = client.audio.transcriptions.create(
                model="whisper-1", file=ft, response_format="text"
            )
            user_input = transcription.strip()

        # AUDIO → Whisper
        elif audio_file:
            ft = (audio_file.filename, audio_file.stream, audio_file.content_type)
            transcription = client.audio.transcriptions.create(
                model="whisper-1", file=ft, response_format="text"
            )
            user_input = transcription.strip()

        # DOCUMENT
        elif document_file:
            fn = document_file.filename.lower()
            if fn.endswith(".txt"):
                user_input = document_file.read().decode("utf-8").strip()
            elif fn.endswith(".pdf"):
                reader = PyPDF2.PdfReader(document_file)
                text = "".join(p.extract_text() + "\n" for p in reader.pages)
                user_input = text.strip()
            else:
                return jsonify({"error": "Unsupported document format."}), 400

        # TEXT
        elif text_input:
            user_input = text_input
        else:
            return jsonify({"error": "No input provided"}), 400

        # Fallback ChatCompletion (multilingual)
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant fluent in many languages. "
                        "Detect and reply in the user's language."
                    )
                },
                {"role": "user", "content": user_input}
            ],
            max_tokens=150
        )
        bot_text = resp.choices[0].message.content.strip()

        if output_type == "speech":
            fn = f"static/audio_{uuid.uuid4().hex}.mp3"
            tts = client.audio.speech.create(
                model="tts-1", voice="alloy", input=bot_text
            )
            tts.stream_to_file(fn)
            return jsonify({"response": bot_text, "audio_url": fn})
        if output_type == "image":
            img = client.images.generate(
                model="dall-e-3",
                prompt=user_input,
                size="1024x1024",
                quality="standard",
                n=1
            )
            return jsonify({
                "response": "Image generated",
                "image_url": img.data[0].url
            })

        return jsonify({"response": bot_text})

    except Exception as e:
        # Log full traceback for debugging
        app.logger.error(traceback.format_exc())
        return jsonify({"error": "Server error: " + str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
