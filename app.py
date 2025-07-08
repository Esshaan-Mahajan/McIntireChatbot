import os, uuid, base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import PyPDF2                 # pip install PyPDF2
from langdetect import detect, DetectorFactory  # pip install langdetect

DetectorFactory.seed = 0
app = Flask(__name__)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    # gather inputs
    text_input    = request.form.get("text","").strip()
    image_file    = request.files.get("image")
    video_file    = request.files.get("video")
    audio_file    = request.files.get("audio")
    document_file = request.files.get("document")
    restrict      = request.form.get("restrict_scope") == "on"
    output_type   = request.form.get("output_type","text")

    doc_text = None
    if document_file:
        name = document_file.filename.lower()
        try:
            if name.endswith(".txt"):
                doc_text = document_file.read().decode("utf-8").strip()
            elif name.endswith(".pdf"):
                reader = PyPDF2.PdfReader(document_file)
                doc_text = "\n".join(p.extract_text() for p in reader.pages if p.extract_text())
            else:
                return jsonify({"error":"Unsupported document; only .txt or .pdf"}),400
        except Exception as e:
            return jsonify({"error":"Document error: "+str(e)}),500

    # handle image/video/audio same as before (omitted here for brevity)...
    # [ you can keep your existing multimodal branches above ]

    # finally: resolve user_input (text)
    user_input = text_input
    if not user_input and not doc_text:
        return jsonify({"error":"No input provided"}),400

    # build system prompt
    if restrict:
        if not doc_text:
            return jsonify({"error":"Toggle ON requires an uploaded document"}),400
        system_prompt = f"You are a helpful assistant. You MUST answer ONLY using the following data:\n\n{doc_text}"
    else:
        system_prompt = (
            "You are a helpful assistant fluent in many languages. "
            "Detect the user’s language and reply in that same language."
        )

    # call the chat API
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system", "content": system_prompt},
                {"role":"user",   "content": user_input}
            ],
            max_tokens=150
        )
        bot_text = resp.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({"error":"Chat failed: "+str(e)}),500

    # detect language for frontend TTS
    try:
        lang = detect(bot_text)
    except:
        lang = "en"

    # if speech output requested
    if output_type == "speech":
        try:
            fn = f"static/audio_{uuid.uuid4().hex}.mp3"
            tts = client.audio.speech.create(model="tts-1", voice="alloy", input=bot_text)
            tts.stream_to_file(fn)
            return jsonify({"response":bot_text, "audio_url":fn, "language":lang})
        except Exception as e:
            return jsonify({"error":"TTS failed: "+str(e)}),500

    # image generation (if required)
    if output_type == "image":
        try:
            img = client.images.generate(
                model="dall-e-3",
                prompt=bot_text,
                size="1024x1024",
                quality="standard",
                n=1
            )
            return jsonify({"response":"Image generated","image_url":img.data[0].url})
        except Exception as e:
            return jsonify({"error":"Image gen failed: "+str(e)}),500

    # default: text
    return jsonify({"response":bot_text, "language":lang})

if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0", port=port, debug=True)
