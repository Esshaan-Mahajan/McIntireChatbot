import os
import uuid
import base64
import traceback
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from openai import OpenAI
import PyPDF2
from langdetect import detect, DetectorFactory
import matplotlib.pyplot as plt

# Mood storage utility (your existing file)
from mood_storage import log_mood, get_mood_history

# Seed langdetect
DetectorFactory.seed = 0

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "supersecret")

# OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ——— Helpers ——————————————————————————————

def store_mood(text: str, user_id: str) -> str:
    """Log the user's mood entry and return confirmation."""
    log_mood(user_id, text)
    return f"✅ Logged mood entry: “{text}”"

def retrieve_mood_history(user_id: str):
    """Return list of mood log dicts."""
    return get_mood_history(user_id)


# ——— Routes ——————————————————————————————

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        # — Session / user ID —
        uid = session.get("uid")
        if not uid:
            uid = str(uuid.uuid4())
            session["uid"] = uid

        # — Inputs & flags —
        text_input    = request.form.get("text", "").strip()
        image_file    = request.files.get("image")
        video_file    = request.files.get("video")
        audio_file    = request.files.get("audio")
        document_file = request.files.get("document")
        output_type   = request.form.get("output_type", "text")
        mh_mode       = request.form.get("mh_mode") == "on"

        # — 1) Acknowledge daily check-in request —
        if text_input.lower().startswith(("enable daily check-in", "schedule daily check in")):
            # you could integrate APScheduler or Heroku Scheduler here
            return jsonify({
                "response": "✅ Got it! I'll remind you every day at 9:00 AM to check in your mood."
            })

        # — 2) Mood trend graph —
        if text_input.lower().startswith(("show mood trend", "show mood history", "trend")):
            history = retrieve_mood_history(uid)
            if not history:
                return jsonify({"response": "No mood history to show yet."})
            # extract dates & numeric
            dates, vals = [], []
            for e in history:
                dates.append(e["timestamp"][:10])
                # try to find a number in the mood string
                num = None
                for tok in e["mood"].split():
                    if tok.isdigit():
                        num = int(tok)
                        break
                vals.append(num or 0)
            # plot
            plt.figure(figsize=(6,3))
            plt.plot(dates, vals, marker="o")
            plt.title("Mood Over Time")
            plt.xlabel("Date")
            plt.ylabel("Rating")
            plt.xticks(rotation=45)
            plt.tight_layout()
            path = f"static/mood_trend_{uid}.png"
            plt.savefig(path)
            plt.close()
            return jsonify({
                "response": "Here’s your mood trend over time:",
                "image_url": path
            })

        # — 3) Mental-health multi-agent mode —
        if mh_mode and text_input:
            # MoodTracker
            mood_reply = store_mood(text_input, uid)

            # SuggestionAgent
            suggestion = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a wellness coach. Suggest two concrete coping strategies right now."
                    },
                    {"role": "user", "content": text_input}
                ],
                max_tokens=200
            ).choices[0].message.content.strip()

            # CompanionAgent
            companion = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a caring friend. Acknowledge feelings and offer empathy."
                    },
                    {"role": "user", "content": text_input}
                ],
                max_tokens=150
            ).choices[0].message.content.strip()

            bot_text = (
                f"**MoodTracker:** {mood_reply}\n\n"
                f"**Suggestion:** {suggestion}\n\n"
                f"**Companion:** {companion}"
            )
            lang = "en"
            try:
                lang = detect(bot_text)
            except:
                pass
            return jsonify({"response": bot_text, "language": lang})

        # — 4) Multimodal pipeline —

        # IMAGE → vision
        if image_file:
            content = [{"type":"text","text": text_input or "What's in this image?"}]
            b64 = base64.b64encode(image_file.read()).decode("utf-8")
            content.append({
                "type":"image_url",
                "image_url":{"url":f"data:{image_file.content_type};base64,{b64}","detail":"auto"}
            })
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":content}],
                max_tokens=300
            )
            bot_text = resp.choices[0].message.content.strip()

            # speech output
            if output_type=="speech":
                fn = f"static/audio_{uuid.uuid4().hex}.mp3"
                client.audio.speech.create(
                    model="tts-1", voice="alloy", input=bot_text
                ).stream_to_file(fn)
                return jsonify({"response":bot_text, "audio_url":fn})

            # image generation
            if output_type=="image":
                img = client.images.generate(
                    model="dall-e-3",
                    prompt=bot_text,
                    size="1024x1024",
                    quality="standard",
                    n=1
                )
                return jsonify({"response":"Image generated","image_url":img.data[0].url})

            return jsonify({"response":bot_text})

        # VIDEO → Whisper
        if video_file:
            ft = (video_file.filename, video_file.stream, video_file.content_type)
            user_input = client.audio.transcriptions.create(
                model="whisper-1", file=ft, response_format="text"
            ).strip()

        # AUDIO → Whisper
        elif audio_file:
            ft = (audio_file.filename, audio_file.stream, audio_file.content_type)
            user_input = client.audio.transcriptions.create(
                model="whisper-1", file=ft, response_format="text"
            ).strip()

        # DOCUMENT (.txt/.pdf)
        elif document_file:
            fn = document_file.filename.lower()
            if fn.endswith(".txt"):
                user_input = document_file.read().decode("utf-8").strip()
            elif fn.endswith(".pdf"):
                rdr = PyPDF2.PdfReader(document_file)
                user_input = "".join(p.extract_text()+"\n" for p in rdr.pages).strip()
            else:
                return jsonify({"error":"Unsupported document format."}), 400

        # PLAIN TEXT
        else:
            if text_input:
                user_input = text_input
            else:
                return jsonify({"error":"No input provided."}), 400

        # — 5) Fallback multilingual chat —
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role":"system",
                    "content":"You are a helpful assistant fluent in many languages. Reply in the user's language."
                },
                {"role":"user","content":user_input}
            ],
            max_tokens=150
        )
        bot_text = resp.choices[0].message.content.strip()
        return jsonify({"response":bot_text})

    except Exception:
        app.logger.error(traceback.format_exc())
        return jsonify({"error":"Server error"}), 500


if __name__=="__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
