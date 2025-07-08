import os
import uuid
import base64
from io import BytesIO
from openai import OpenAI
from flask import Flask, render_template, request, jsonify
import PyPDF2  # Install via: pip install PyPDF2

app = Flask(__name__)

# Create an OpenAI client using your API key from the environment.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    # Retrieve input modalities and desired output type.
    text_input = request.form.get("text", "").strip()
    image_file = request.files.get("image")
    video_file = request.files.get("video")
    audio_file = request.files.get("audio")
    document_file = request.files.get("document")
    output_type = request.form.get("output_type", "text")  # Options: "text", "image", "speech"

    user_input = None

    # Priority order: Image > Video > Audio > Document > Text.
    if image_file:
        # --- Process image input using vision model ---
        content_list = []
        if text_input:
            content_list.append({"type": "text", "text": text_input})
        else:
            content_list.append({"type": "text", "text": "What's in this image?"})
        image_bytes = image_file.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        image_data = f"data:{image_file.content_type};base64,{base64_image}"
        content_list.append({
            "type": "image_url",
            "image_url": {"url": image_data, "detail": "auto"}
        })
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Vision-capable model
                messages=[{"role": "user", "content": content_list}],
                max_tokens=300,
            )
            chatbot_response = response.choices[0].message.content.strip()
        except Exception as e:
            return jsonify({"error": "Vision API failed: " + str(e)}), 500

        if output_type == "speech":
            try:
                audio_filename = f"static/audio_{uuid.uuid4().hex}.mp3"
                tts_response = client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=chatbot_response,
                )
                tts_response.stream_to_file(audio_filename)
                return jsonify({"response": chatbot_response, "audio_url": audio_filename})
            except Exception as e:
                return jsonify({"error": "Speech synthesis failed: " + str(e)}), 500
        elif output_type == "image":
            try:
                image_response = client.images.generate(
                    model="dall-e-3",
                    prompt=chatbot_response,
                    size="1024x1024",
                    quality="standard",
                    n=1,
                )
                image_url = image_response.data[0].url
                return jsonify({"response": "Image generated", "image_url": image_url})
            except Exception as e:
                return jsonify({"error": "Image generation failed: " + str(e)}), 500
        else:
            return jsonify({"response": chatbot_response})

    elif video_file:
        # --- Process video input by transcribing its audio track ---
        try:
            file_tuple = (video_file.filename, video_file.stream, video_file.content_type)
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=file_tuple,
                response_format="text"
            )
            user_input = transcription.strip()
        except Exception as e:
            return jsonify({"error": "Video transcription failed: " + str(e)}), 500

    elif audio_file:
        # --- Process audio input ---
        try:
            file_tuple = (audio_file.filename, audio_file.stream, audio_file.content_type)
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=file_tuple,
                response_format="text"
            )
            user_input = transcription.strip()
        except Exception as e:
            return jsonify({"error": "Audio transcription failed: " + str(e)}), 500

    elif document_file:
        # --- Process document input (supports .txt and .pdf) ---
        try:
            filename = document_file.filename.lower()
            if filename.endswith(".txt"):
                user_input = document_file.read().decode("utf-8").strip()
            elif filename.endswith(".pdf"):
                pdf_reader = PyPDF2.PdfReader(document_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                user_input = text.strip()
            else:
                return jsonify({"error": "Unsupported document format. Only .txt and .pdf are supported."}), 400
        except Exception as e:
            return jsonify({"error": "Document processing failed: " + str(e)}), 500

    elif text_input:
        user_input = text_input
    else:
        return jsonify({"error": "No input provided"}), 400

    # --- Process text input using a text-only model ---
    if user_input:
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant fluent in many languages. "
                        "Detect the language of the user’s message and reply in that same language."
                    )
                },
                    {"role": "user", "content": user_input}
                ],
                max_tokens=150
            )
            chatbot_response = response.choices[0].message.content.strip()
        except Exception as e:
            return jsonify({"error": "Chat completion failed: " + str(e)}), 500

        if output_type == "speech":
            try:
                audio_filename = f"static/audio_{uuid.uuid4().hex}.mp3"
                tts_response = client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=chatbot_response,
                )
                tts_response.stream_to_file(audio_filename)
                return jsonify({"response": chatbot_response, "audio_url": audio_filename})
            except Exception as e:
                return jsonify({"error": "Speech synthesis failed: " + str(e)}), 500
        elif output_type == "image":
            try:
                image_response = client.images.generate(
                    model="dall-e-3",
                    prompt=user_input,
                    size="1024x1024",
                    quality="standard",
                    n=1,
                )
                image_url = image_response.data[0].url
                return jsonify({"response": "Image generated", "image_url": image_url})
            except Exception as e:
                return jsonify({"error": "Image generation failed: " + str(e)}), 500
        else:
            return jsonify({"response": chatbot_response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
