import os
import uuid
import base64
from io import BytesIO
from openai import OpenAI
from flask import Flask, render_template, request, jsonify
import PyPDF2  # pip install PyPDF2
from langdetect import detect, DetectorFactory  # pip install langdetect

# Ensure consistent language detection
DetectorFactory.seed = 0

app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    # Retrieve inputs
    text_input    = request.form.get("text", "").strip()
    image_file    = request.files.get("image")
    video_file    = request.files.get("video")
    audio_file    = request.files.get("audio")
    document_file = request.files.get("document")
    output_type   = request.form.get("output_type", "text")  # "text", "image", "speech"

    user_input = None

    # 1) Image
    if image_file:
        content_list = []
        if text_input:
            content_list.append({"type": "text", "text": text_input})
        else:
            content_list.append({"type": "text", "text": "What's in this image?"})
        img_bytes    = image_file.read()
        b64          = base64.b64encode(img_bytes).decode("utf-8")
        data_uri     = f"data:{image_file.content_type};base64,{b64}"
        content_list.append({
            "type": "image_url",
            "image_url": {"url": data_uri, "detail": "auto"}
        })
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": content_list}],
                max_tokens=300,
            )
            chatbot_response = resp.choices[0].message.content.strip()
        except Exception as e:
            return jsonify({"error": "Vision API failed: " + str(e)}), 500

        # return based on output type
        if output_type == "speech":
            try:
                fn = f"static/audio_{uuid.uuid4().hex}.mp3"
                tts = client.audio.speech.create(
                    model="tts-1", voice="alloy", input=chatbot_response
                )
                tts.stream_to_file(fn)
                lang = _detect_lang(chatbot_response)
                return jsonify({"response": chatbot_response, "audio_url": fn, "language": lang})
            except Exception as e:
                return jsonify({"error": "Speech synthesis failed: " + str(e)}), 500
        if output_type == "image":
            try:
                img = client.images.generate(
                    model="dall-e-3",
                    prompt=chatbot_response,
                    size="1024x1024",
                    quality="standard",
                    n=1,
                )
                return jsonify({"response": "Image generated", "image_url": img.data[0].url})
            except Exception as e:
                return jsonify({"error": "Image generation failed: " + str(e)}), 500

        return jsonify({"response": chatbot_response})

    # 2) Video → whisper
    if video_file:
        try:
            tup = (video_file.filename, video_file.stream, video_file.content_type)
            transcription = client.audio.transcriptions.create(
                model="whisper-1", file=tup, response_format="text"
            )
            user_input = transcription.strip()
        except Exception as e:
            return jsonify({"error": "Video transcription failed: " + str(e)}), 500

    # 3) Audio → whisper
    elif audio_file:
        try:
            tup = (audio_file.filename, audio_file.stream, audio_file.content_type)
            transcription = client.audio.transcriptions.create(
                model="whisper-1", file=tup, response_format="text"
            )
            user_input = transcription.strip()
        except Exception as e:
            return jsonify({"error": "Audio transcription failed: " + str(e)}), 500

    # 4) Document
    elif document_file:
        try:
            name = document_file.filename.lower()
            if name.endswith(".txt"):
                user_input = document_file.read().decode("utf-8").strip()
            elif name.endswith(".pdf"):
                reader = PyPDF2.PdfReader(document_file)
                text = ""
                for p in reader.pages:
                    text += p.extract_text() + "\n"
                user_input = text.strip()
            else:
                return jsonify({"error": "Unsupported format. Use .txt or .pdf"}), 400
        except Exception as e:
            return jsonify({"error": "Document processing failed: " + str(e)}), 500

    # 5) Plain text
    elif text_input:
        user_input = text_input
    else:
        return jsonify({"error": "No input provided"}), 400

    # Process text input
    if user_input:
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant fluent in many languages. "
                        "Detect the language of the user’s message and reply in that same language."
                    )
                },
                {"role": "user", "content": user_input}
            ]
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo", messages=messages, max_tokens=150
            )
            chatbot_response = resp.choices[0].message.content.strip()
        except Exception as e:
            return jsonify({"error": "Chat completion failed: " + str(e)}), 500

        # speech
        if output_type == "speech":
            try:
                fn = f"static/audio_{uuid.uuid4().hex}.mp3"
                tts = client.audio.speech.create(
                    model="tts-1", voice="alloy", input=chatbot_response
                )
                tts.stream_to_file(fn)
                lang = _detect_lang(chatbot_response)
                return jsonify({"response": chatbot_response, "audio_url": fn, "language": lang})
            except Exception as e:
                return jsonify({"error": "Speech synthesis failed: " + str(e)}), 500

        # image gen
        if output_type == "image":
            try:
                img = client.images.generate(
                    model="dall-e-3",
                    prompt=chatbot_response,
                    size="1024x1024",
                    quality="standard",
                    n=1,
                )
                return jsonify({"response": "Image generated", "image_url": img.data[0].url})
            except Exception as e:
                return jsonify({"error": "Image generation failed: " + str(e)}), 500

        # plain text
        return jsonify({"response": chatbot_response, "language": detect(chatbot_response)})

def _detect_lang(text: str) -> str:
    try:
        return detect(text)
    except:
        return "en"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
