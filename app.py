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

# fix randomness in langdetect
DetectorFactory.seed = 0

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "supersecret")

# OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ——— Registerable “tool” functions for MoodTrackerAgent ——————————

@register_function
def store_mood(mood: str, user_id: str = "default_user") -> str:
    log_mood(user_id, mood)
    return f"✅ Logged mood: {mood}"

@register_function
def retrieve_mood_history(user_id: str = "default_user") -> str:
    history = get_mood_history(user_id)
    if not history:
        return "No mood history found."
    lines = [f"{e['timestamp'][:10]}: {e['mood']}" for e in history]
    return "📊 Your mood history:\n" + "\n".join(lines)

# ——— Define AutoGen Agents ——————————————————————————————

user_proxy = UserProxyAgent(name="UserProxyAgent", human_input_mode="ALWAYS")

mood_tracker = AssistantAgent(
    name="MoodTrackerAgent",
    llm_config={"model": "gpt-4", "temperature": 0.3},
    system_message=(
        "You are MoodTrackerAgent. Ask the user for mood ratings or emotions, "
        "use store_mood(mood) to log them, and retrieve_mood_history() to report trends."
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

# per-session proxy storage
user_sessions = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    # — Session management for multi-agent state ——
    sid = session.get("sid")
    if not sid:
        sid = str(uuid.uuid4())
        session["sid"] = sid
    if sid not in user_sessions:
        user_sessions[sid] = user_proxy
    proxy = user_sessions[sid]

    # — Gather inputs & flags ——
    text_input    = request.form.get("text", "").strip()
    image_file    = request.files.get("image")
    video_file    = request.files.get("video")
    audio_file    = request.files.get("audio")
    document_file = request.files.get("document")
    output_type   = request.form.get("output_type", "text")
    restrict      = request.form.get("restrict_scope") == "on"
    mh_mode       = request.form.get("mh_mode") == "on"

    # — 1) Mental-Health Multi-Agent Mode ——
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

    # — 2) Otherwise: Multimodal pipeline ——

    # IMAGE input
    if image_file:
        # prepare vision+text payload
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

        # output
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

    # VIDEO input → whisper
    if video_file:
        try:
            file_tuple = (video_file.filename, video_file.stream, video_file.content_type)
            transcription = client.audio.transcriptions.create(
                model="whisper-1", file=file_tuple, response_format="text"
            )
            user_input = transcription.strip()
        except Exception as e:
            return jsonify({"error": "Video transcription failed: " + str(e)}), 500

    # AUDIO input → whisper
    elif audio_file:
        try:
            file_tuple = (audio_file.filename, audio_file.stream, audio_file.content_type)
            transcription = client.audio.transcriptions.create(
                model="whisper-1", file=file_tuple, response_format="text"
            )
            user_input = transcription.strip()
        except Exception as e:
            return jsonify({"error": "Audio transcription failed: " + str(e)}), 500

    # DOCUMENT input
    elif document_file:
        try:
            fn = document_file.filename.lower()
            if fn.endswith(".txt"):
                user_input = document_file.read().decode("utf-8").strip()
            elif fn.endswith(".pdf"):
                reader = PyPDF2.PdfReader(document_file)
                text = "".join(p.extract_text() + "\n" for p in reader.pages)
                user_input = text.strip()
            else:
                return jsonify({"error": "Unsupported document format."}), 400
        except Exception as e:
            return jsonify({"error": "Document processing failed: " + str(e)}), 500

    # TEXT input
    elif text_input:
        user_input = text_input
    else:
        return jsonify({"error": "No input provided"}), 400

    # Fallback ChatCompletion (multilingual)
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content":
                    "You are a helpful assistant fluent in many languages. "
                    "Detect the language of the user’s message and reply in that same language."
                },
                {"role": "user", "content": user_input}
            ],
            max_tokens=150
        )
        bot_text = resp.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # render according to requested output
    if output_type == "speech":
        fn = f"static/audio_{uuid.uuid4().hex}.mp3"
        tts = client.audio.speech.create(model="tts-1", voice="alloy", input=bot_text)
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
        return jsonify({"response": "Image generated", "image_url": img.data[0].url})
    return jsonify({"response": bot_text})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
