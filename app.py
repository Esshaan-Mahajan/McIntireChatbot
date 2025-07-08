import os
import uuid
import base64
import traceback
from flask import Flask, render_template, request, jsonify, session
from openai import OpenAI
import PyPDF2
from langdetect import detect, DetectorFactory
from automations import create as create_task
from mood_storage import log_mood, get_mood_history

# For trend graph
import matplotlib.pyplot as plt

# Seed langdetect for reproducibility
DetectorFactory.seed = 0

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "supersecret")

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ——— Helper functions ——————————————————————————————

def store_mood(text: str, user_id: str = "default_user") -> str:
    log_mood(user_id, text)
    return f"✅ Logged mood entry: “{text}”"

def retrieve_mood_history(user_id: str = "default_user") -> list[dict]:
    return get_mood_history(user_id)

# ——— Routes ——————————————————————————————

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Session management
        sid = session.get("sid")
        if not sid:
            sid = str(uuid.uuid4())
            session["sid"] = sid

        # Inputs & flags
        text_input    = request.form.get("text", "").strip()
        mh_mode       = request.form.get("mh_mode") == "on"
        output_type   = request.form.get("output_type", "text")

        # 1) Scheduling daily check-ins
        if text_input.lower().startswith(("enable daily check-in", "schedule daily check in")):
            create_task(
                title="Daily Mood Check-In",
                prompt="Tell me to check in my mood and log how I'm feeling.",
                schedule="BEGIN:VEVENT\nRRULE:FREQ=DAILY;BYHOUR=9;BYMINUTE=0;BYSECOND=0\nEND:VEVENT"
            )
            return jsonify({
                "response": "✅ Got it! I'll remind you every day at 9:00 AM to check in your mood."
            })

        # 2) Show mood trend graph
        if text_input.lower().startswith(("show mood trend", "show mood history", "trend")):
            history = retrieve_mood_history(sid)
            if not history:
                return jsonify({"response": "No mood history to show."})

            # Extract dates and numeric ratings if possible
            dates, values = [], []
            for entry in history:
                dates.append(entry["timestamp"][:10])
                # Try to extract a number out of the mood string
                try:
                    num = int(next(tok for tok in entry["mood"].split() if tok.isdigit()))
                except StopIteration:
                    num = None
                values.append(num)

            # Plot
            plt.figure(figsize=(6, 3))
            plt.plot(dates, values, marker='o')
            plt.title("Mood Over Time")
            plt.xlabel("Date")
            plt.ylabel("Rating")
            plt.xticks(rotation=45)
            plt.tight_layout()
            graph_path = f"static/mood_trend_{sid}.png"
            plt.savefig(graph_path)
            plt.close()

            return jsonify({
                "response": "Here’s your mood trend over time:",
                "image_url": graph_path
            })

        # 3) Mental-Health Mode
        if mh_mode and text_input:
            # MoodTracker logs
            mood_reply = store_mood(text_input)

            # SuggestionAgent
            suggestion_resp = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": 
                        "You are a wellness coach. Suggest two coping strategies based on the user's mood."},
                    {"role": "user", "content": text_input}
                ],
                max_tokens=200
            ).choices[0].message.content.strip()

            # CompanionAgent
            companion_resp = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": 
                        "You are a caring friend. Acknowledge feelings and offer empathy."},
                    {"role": "user", "content": text_input}
                ],
                max_tokens=150
            ).choices[0].message.content.strip()

            bot_text = (
                f"**MoodTracker:** {mood_reply}\n\n"
                f"**Suggestion:** {suggestion_resp}\n\n"
                f"**Companion:** {companion_resp}"
            )
            lang = "en"
            try:
                lang = detect(bot_text)
            except:
                pass
            return jsonify({"response": bot_text, "language": lang})

        # 4) Multimodal Pipeline (image/video/audio/document/text)...

        # IMAGE
        image_file = request.files.get("image")
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
            if output_type=="speech":
                fn = f"static/audio_{uuid.uuid4().hex}.mp3"
                client.audio.speech.create(model="tts-1",voice="alloy",input=bot_text).stream_to_file(fn)
                return jsonify({"response":bot_text,"audio_url":fn})
            if output_type=="image":
                img = client.images.generate(model="dall-e-3",prompt=bot_text,size="1024x1024",quality="standard",n=1)
                return jsonify({"response":"Image generated","image_url":img.data[0].url})
            return jsonify({"response":bot_text})

        # VIDEO ➔ Whisper
        video_file = request.files.get("video")
        if video_file:
            ft = (video_file.filename, video_file.stream, video_file.content_type)
            user_input = client.audio.transcriptions.create(model="whisper-1",file=ft,response_format="text").strip()

        # AUDIO ➔ Whisper
        audio_file = request.files.get("audio")
        if audio_file:
            ft = (audio_file.filename, audio_file.stream, audio_file.content_type)
            user_input = client.audio.transcriptions.create(model="whisper-1",file=ft,response_format="text").strip()

        # DOCUMENT
        document_file = request.files.get("document")
        if document_file:
            fn = document_file.filename.lower()
            if fn.endswith(".txt"):
                user_input = document_file.read().decode("utf-8").strip()
            elif fn.endswith(".pdf"):
                rdr = PyPDF2.PdfReader(document_file)
                user_input = "".join(p.extract_text()+"\n" for p in rdr.pages).strip()
            else:
                return jsonify({"error":"Unsupported document."}),400

        # TEXT fallback
        if not (image_file or video_file or audio_file or document_file):
            if text_input:
                user_input = text_input
            else:
                return jsonify({"error":"No input provided"}),400

        # 5) Multilingual Chat fallback
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system","content":
                    "You are a helpful assistant fluent in many languages. Reply in the user's language."},
                {"role":"user","content":user_input}
            ],
            max_tokens=150
        )
        bot_text = resp.choices[0].message.content.strip()
        return jsonify({"response":bot_text})

    except Exception:
        app.logger.error(traceback.format_exc())
        return jsonify({"error":"Server error"}),500

if __name__=="__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0", port=port, debug=True)
