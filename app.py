import os
import uuid
import base64
import traceback
from flask import Flask, render_template, request, jsonify, session
from openai import OpenAI
import PyPDF2
from langdetect import detect, DetectorFactory

# For mood logging
from mood_storage import log_mood, get_mood_history

# Seed langdetect for reproducibility
DetectorFactory.seed = 0

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "supersecret")

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ——— Helper functions ——————————————————————————————

def store_mood(text: str, user_id: str = "default_user") -> str:
    """
    Log the user's mood entry (expects something like '4 out of 10').
    Returns a confirmation message.
    """
    # You might parse out the numeric rating if you wish; here we store the raw text.
    log_mood(user_id, text)
    return f"✅ Logged mood entry: “{text}”"

def retrieve_mood_history(user_id: str = "default_user") -> str:
    """
    Fetches and formats the user's past mood entries.
    """
    history = get_mood_history(user_id)
    if not history:
        return "No mood history found."
    lines = [f"{e['timestamp'][:10]}: {e['mood']}" for e in history]
    return "📊 Your mood history:\n" + "\n".join(lines)

# ——— Routes ——————————————————————————————

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        # --- Session management ---
        sid = session.get("sid")
        if not sid:
            sid = str(uuid.uuid4())
            session["sid"] = sid

        # --- Gather inputs & flags ---
        text_input    = request.form.get("text", "").strip()
        image_file    = request.files.get("image")
        video_file    = request.files.get("video")
        audio_file    = request.files.get("audio")
        document_file = request.files.get("document")
        output_type   = request.form.get("output_type", "text")   # "text", "image", or "speech"
        mh_mode       = request.form.get("mh_mode") == "on"       # Mental-health toggle

        # ——— 1) Mental-Health “Multi-Agent” Mode ——————————
        if mh_mode and text_input:
            # 1) MoodTracker logs and confirms
            mood_reply = store_mood(text_input)

            # 2) SuggestionAgent via direct ChatCompletion
            suggestion_resp = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a wellness coach. Based on the user's mood input, "
                            "suggest two concrete coping strategies they can try right now."
                        )
                    },
                    {"role": "user", "content": text_input}
                ],
                max_tokens=150
            ).choices[0].message.content.strip()

            # 3) CompanionAgent via direct ChatCompletion
            companion_resp = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a caring friend. Acknowledge the user's feelings "
                            "and offer empathy in a warm, human-like tone."
                        )
                    },
                    {"role": "user", "content": text_input}
                ],
                max_tokens=150
            ).choices[0].message.content.strip()

            # Stitch the three replies together
            bot_text = (
                f"**MoodTracker:** {mood_reply}\n\n"
                f"**Suggestion:** {suggestion_resp}\n\n"
                f"**Companion:** {companion_resp}"
            )

            # Optionally detect the language
            lang = "en"
            try:
                lang = detect(bot_text)
            except:
                pass

            return jsonify({"response": bot_text, "language": lang})

        # ——— 2) Multimodal Pipeline ————————————————

        # IMAGE ➔ Vision-capable model
        if image_file:
            content = []
            if text_input:
                content.append({"type": "text", "text": text_input})
            else:
                content.append({"type": "text", "text": "What's in this image?"})
            img_bytes = image_file.read()
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{image_file.content_type};base64,{b64}", "detail": "auto"}
            })
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": content}],
                max_tokens=300
            )
            bot_text = resp.choices[0].message.content.strip()

            # Speech output
            if output_type == "speech":
                fn = f"static/audio_{uuid.uuid4().hex}.mp3"
                tts = client.audio.speech.create(model="tts-1", voice="alloy", input=bot_text)
                tts.stream_to_file(fn)
                return jsonify({"response": bot_text, "audio_url": fn})

            # Image generation
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

        # VIDEO ➔ Whisper transcription
        if video_file:
            ft = (video_file.filename, video_file.stream, video_file.content_type)
            transcription = client.audio.transcriptions.create(
                model="whisper-1", file=ft, response_format="text"
            )
            user_input = transcription.strip()

        # AUDIO ➔ Whisper transcription
        elif audio_file:
            ft = (audio_file.filename, audio_file.stream, audio_file.content_type)
            transcription = client.audio.transcriptions.create(
                model="whisper-1", file=ft, response_format="text"
            )
            user_input = transcription.strip()

        # DOCUMENT (.txt or .pdf)
        elif document_file:
            fn = document_file.filename.lower()
            if fn.endswith(".txt"):
                user_input = document_file.read().decode("utf-8").strip()
            elif fn.endswith(".pdf"):
                reader = PyPDF2.PdfReader(document_file)
                text = "".join(page.extract_text() + "\n" for page in reader.pages)
                user_input = text.strip()
            else:
                return jsonify({"error": "Unsupported document format."}), 400

        # PLAIN TEXT
        elif text_input:
            user_input = text_input

        else:
            return jsonify({"error": "No input provided"}), 400

        # ——— 3) Fallback multilingual ChatCompletion —————————

        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": (
                     "You are a helpful assistant fluent in many languages. "
                     "Detect and reply in the user's language."
                 )},
                {"role": "user", "content": user_input}
            ],
            max_tokens=150
        )
        bot_text = resp.choices[0].message.content.strip()

        # Speech output
        if output_type == "speech":
            fn = f"static/audio_{uuid.uuid4().hex}.mp3"
            tts = client.audio.speech.create(model="tts-1", voice="alloy", input=bot_text)
            tts.stream_to_file(fn)
            return jsonify({"response": bot_text, "audio_url": fn})

        # Image generation
        if output_type == "image":
            img = client.images.generate(
                model="dall-e-3",
                prompt=user_input,
                size="1024x1024",
                quality="standard",
                n=1
            )
            return jsonify({"response": "Image generated", "image_url": img.data[0].url})

        # Default: text
        return jsonify({"response": bot_text})

    except Exception:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": "Server error"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
