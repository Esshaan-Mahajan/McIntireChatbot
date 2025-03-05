import os
import uuid
import base64
from openai import OpenAI
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Create an OpenAI client using your API key from the environment.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    # Retrieve input modalities and desired output type from the form.
    text_input = request.form.get("text", "").strip()
    audio_file = request.files.get("audio")
    image_file = request.files.get("image")
    output_type = request.form.get("output_type", "text")  # Options: "text", "image", "speech"

    # --------------------------------------------
    # 1. If an image is provided, use the vision-capable model.
    #    If text is provided along with the image, include it;
    #    otherwise, use a default prompt.
    # --------------------------------------------
    if image_file:
        content_list = []
        if text_input:
            content_list.append({"type": "text", "text": text_input})
        else:
            content_list.append({"type": "text", "text": "What's in this image?"})

        # Read the image bytes and encode as a Base64 data URI.
        image_bytes = image_file.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        image_data = f"data:{image_file.content_type};base64,{base64_image}"
        content_list.append({
            "type": "image_url",
            "image_url": {
                "url": image_data,
                "detail": "auto"  # Options: "low", "high", or "auto"
            }
        })

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Use a vision-capable model
                messages=[{
                    "role": "user",
                    "content": content_list
                }],
                max_tokens=300,
            )
            chatbot_response = response.choices[0].message.content.strip()
        except Exception as e:
            return jsonify({"error": "Vision API failed: " + str(e)}), 500

        # Handle output for the image branch.
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

    # --------------------------------------------
    # 2. Else if an audio file is provided (and no image), transcribe it.
    # --------------------------------------------
    elif audio_file:
        try:
            # Prepare the audio file as a tuple: (filename, file stream, content type)
            file_tuple = (audio_file.filename, audio_file.stream, audio_file.content_type)
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=file_tuple,
                response_format="text"
            )
            # transcription is returned as a plain text string.
            user_input = transcription.strip()
        except Exception as e:
            return jsonify({"error": "Audio transcription failed: " + str(e)}), 500

    # --------------------------------------------
    # 3. Else if only text is provided.
    # --------------------------------------------
    elif text_input:
        user_input = text_input
    else:
        return jsonify({"error": "No input provided"}), 400

    # --------------------------------------------
    # 4. Process the text input using a text-only model.
    # --------------------------------------------
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful chatbot."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=150
        )
        chatbot_response = response.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({"error": "Chat completion failed: " + str(e)}), 500

    # --------------------------------------------
    # 5. Return the output in the desired format.
    # --------------------------------------------
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
    app.run(debug=True)




# import os
# import uuid
# import base64
# from openai import OpenAI
# from flask import Flask, render_template, request, jsonify

# app = Flask(__name__)

# # Create an OpenAI client using your API key from the environment.
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/chat", methods=["POST"])
# def chat():
#     # Retrieve input modalities and desired output type from the form.
#     text_input = request.form.get("text", "").strip()
#     audio_file = request.files.get("audio")
#     image_file = request.files.get("image")
#     output_type = request.form.get("output_type", "text")  # Options: "text", "image", "speech"

#     # --------------------------------------------
#     # 1. If an image is provided, use the vision-capable model.
#     # --------------------------------------------
#     if image_file:
#         content_list = []
#         if text_input:
#             content_list.append({"type": "text", "text": text_input})
#         else:
#             content_list.append({"type": "text", "text": "What's in this image?"})

#         # Read the image bytes and encode as a Base64 data URI.
#         image_bytes = image_file.read()
#         base64_image = base64.b64encode(image_bytes).decode("utf-8")
#         image_data = f"data:{image_file.content_type};base64,{base64_image}"
#         content_list.append({
#             "type": "image_url",
#             "image_url": {
#                 "url": image_data,
#                 "detail": "auto"  # Options: "low", "high", or "auto"
#             }
#         })

#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4o-mini",  # Use a vision-capable model
#                 messages=[{
#                     "role": "user",
#                     "content": content_list
#                 }],
#                 max_tokens=300,
#             )
#             chatbot_response = response.choices[0].message.content.strip()
#         except Exception as e:
#             return jsonify({"error": "Vision API failed: " + str(e)}), 500

#         # Handle output for the image branch.
#         if output_type == "speech":
#             try:
#                 audio_filename = f"static/audio_{uuid.uuid4().hex}.mp3"
#                 tts_response = client.audio.speech.create(
#                     model="tts-1",
#                     voice="alloy",
#                     input=chatbot_response,
#                 )
#                 tts_response.stream_to_file(audio_filename)
#                 return jsonify({"response": chatbot_response, "audio_url": audio_filename})
#             except Exception as e:
#                 return jsonify({"error": "Speech synthesis failed: " + str(e)}), 500
#         elif output_type == "image":
#             try:
#                 image_response = client.image.generations.create(
#                     prompt=chatbot_response,
#                     n=1,
#                     size="1024x1024"
#                 )
#                 image_url = image_response.data[0].url
#                 return jsonify({"response": "Image generated", "image_url": image_url})
#             except Exception as e:
#                 return jsonify({"error": "Image generation failed: " + str(e)}), 500
#         else:
#             return jsonify({"response": chatbot_response})

#     # --------------------------------------------
#     # 2. Else if an audio file is provided (and no image), transcribe it.
#     # --------------------------------------------
#     elif audio_file:
#         try:
#             # Prepare the audio file as a tuple: (filename, file stream, content type)
#             file_tuple = (audio_file.filename, audio_file.stream, audio_file.content_type)
#             transcription = client.audio.transcriptions.create(
#                 model="whisper-1",
#                 file=file_tuple,
#                 response_format="text"
#             )
#             # Since transcription is a string when response_format is "text":
#             user_input = transcription.strip()
#         except Exception as e:
#             return jsonify({"error": "Audio transcription failed: " + str(e)}), 500

#     # --------------------------------------------
#     # 3. Else if only text is provided.
#     # --------------------------------------------
#     elif text_input:
#         user_input = text_input
#     else:
#         return jsonify({"error": "No input provided"}), 400

#     # --------------------------------------------
#     # 4. Process the text input using a text-only model.
#     # --------------------------------------------
#     try:
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are a helpful chatbot."},
#                 {"role": "user", "content": user_input}
#             ],
#             max_tokens=150
#         )
#         chatbot_response = response.choices[0].message.content.strip()
#     except Exception as e:
#         return jsonify({"error": "Chat completion failed: " + str(e)}), 500

#     # --------------------------------------------
#     # 5. Return the output in the desired format.
#     # --------------------------------------------
#     if output_type == "speech":
#         try:
#             audio_filename = f"static/audio_{uuid.uuid4().hex}.mp3"
#             tts_response = client.audio.speech.create(
#                 model="tts-1",
#                 voice="alloy",
#                 input=chatbot_response,
#             )
#             tts_response.stream_to_file(audio_filename)
#             return jsonify({"response": chatbot_response, "audio_url": audio_filename})
#         except Exception as e:
#             return jsonify({"error": "Speech synthesis failed: " + str(e)}), 500
#     elif output_type == "image":
#         try:
#             image_response = client.image.generations.create(
#                 prompt=user_input,
#                 n=1,
#                 size="1024x1024"
#             )
#             image_url = image_response.data[0].url
#             return jsonify({"response": "Image generated", "image_url": image_url})
#         except Exception as e:
#             return jsonify({"error": "Image generation failed: " + str(e)}), 500
#     else:
#         return jsonify({"response": chatbot_response})

# if __name__ == "__main__":
#     app.run(debug=True)





# import os
# import uuid
# import base64
# from openai import OpenAI
# from flask import Flask, render_template, request, jsonify
# from gtts import gTTS

# app = Flask(__name__)

# # Create an OpenAI client using the recommended approach.
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/chat", methods=["POST"])
# def chat():
#     # Retrieve input modalities and desired output type
#     text_input = request.form.get("text", "").strip()
#     audio_file = request.files.get("audio")
#     image_file = request.files.get("image")
#     output_type = request.form.get("output_type", "text")  # "text", "image", or "speech"

#     # -------------------------------
#     # 1. If an image is provided, use the vision model.
#     #    If text is provided along with the image, include it; otherwise, use a default prompt.
#     # -------------------------------
#     if image_file:
#         content_list = []
#         if text_input:
#             content_list.append({"type": "text", "text": text_input})
#         else:
#             content_list.append({"type": "text", "text": "What's in this image?"})

#         # Read and encode the image into Base64 with a data URI prefix.
#         image_bytes = image_file.read()
#         base64_image = base64.b64encode(image_bytes).decode('utf-8')
#         image_data = f"data:{image_file.content_type};base64,{base64_image}"
#         content_list.append({
#             "type": "image_url",
#             "image_url": {
#                 "url": image_data,
#                 "detail": "auto"  # Options: "low", "high", or "auto"
#             }
#         })

#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4o-mini",  # Use a vision-capable model (alternatives: gpt-4o, gpt-4-turbo, etc.)
#                 messages=[{
#                     "role": "user",
#                     "content": content_list
#                 }],
#                 max_tokens=300,
#             )
#             chatbot_response = response.choices[0].message.content.strip()
#         except Exception as e:
#             return jsonify({"error": "Vision API failed: " + str(e)}), 500

#         # Handle output: text, speech (using gTTS), or image (via image generation API)
#         if output_type == "speech":
#             try:
#                 audio_filename = f"static/audio_{uuid.uuid4().hex}.mp3"
#                 tts = gTTS(text=chatbot_response, lang="en")
#                 tts.save(audio_filename)
#                 return jsonify({"response": chatbot_response, "audio_url": audio_filename})
#             except Exception as e:
#                 return jsonify({"error": "Speech synthesis failed: " + str(e)}), 500
#         elif output_type == "image":
#             try:
#                 image_response = client.image.generations.create(
#                     prompt=chatbot_response,
#                     n=1,
#                     size="1024x1024"
#                 )
#                 image_url = image_response.data[0].url
#                 return jsonify({"response": "Image generated", "image_url": image_url})
#             except Exception as e:
#                 return jsonify({"error": "Image generation failed: " + str(e)}), 500
#         else:
#             return jsonify({"response": chatbot_response})

#     # -------------------------------
#     # 2. Else if audio is provided (and no image), process audio input.
#     # -------------------------------
#     elif audio_file:
#         try:
#             transcription = client.audio.transcriptions.create(file=audio_file, model="whisper-1")
#             user_input = transcription.get("text", "").strip()
#         except Exception as e:
#             return jsonify({"error": "Audio transcription failed: " + str(e)}), 500

#     # -------------------------------
#     # 3. Else if only text is provided.
#     # -------------------------------
#     elif text_input:
#         user_input = text_input
#     else:
#         return jsonify({"error": "No input provided"}), 400

#     # -------------------------------
#     # 4. Process the derived text input using a text-only model.
#     # -------------------------------
#     try:
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are a helpful chatbot."},
#                 {"role": "user", "content": user_input}
#             ],
#             max_tokens=150
#         )
#         chatbot_response = response.choices[0].message.content.strip()
#     except Exception as e:
#         return jsonify({"error": "Chat completion failed: " + str(e)}), 500

#     # -------------------------------
#     # 5. Return the output as requested (text, speech, or image generation).
#     # -------------------------------
#     if output_type == "speech":
#         try:
#             audio_filename = f"static/audio_{uuid.uuid4().hex}.mp3"
#             tts = gTTS(text=chatbot_response, lang="en")
#             tts.save(audio_filename)
#             return jsonify({"response": chatbot_response, "audio_url": audio_filename})
#         except Exception as e:
#             return jsonify({"error": "Speech synthesis failed: " + str(e)}), 500
#     elif output_type == "image":
#         try:
#             image_response = client.image.generations.create(
#                 prompt=user_input,
#                 n=1,
#                 size="1024x1024"
#             )
#             image_url = image_response.data[0].url
#             return jsonify({"response": "Image generated", "image_url": image_url})
#         except Exception as e:
#             return jsonify({"error": "Image generation failed: " + str(e)}), 500
#     else:
#         return jsonify({"response": chatbot_response})

# if __name__ == "__main__":
#     app.run(debug=True)



# import os
# import uuid
# from openai import OpenAI
# from flask import Flask, render_template, request, jsonify
# from gtts import gTTS
# from PIL import Image
# import pytesseract

# app = Flask(__name__)

# # Create an OpenAI client using the recommended approach.
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/chat", methods=["POST"])
# def chat():
#     # Retrieve input modalities from the form
#     text_input = request.form.get("text", "").strip()
#     audio_file = request.files.get("audio")
#     image_file = request.files.get("image")
#     output_type = request.form.get("output_type", "text")  # "text", "image", or "speech"

#     user_input = None

#     # Determine the input: prioritize text, then audio, then image
#     if text_input:
#         user_input = text_input
#     elif audio_file:
#         try:
#             # Transcribe audio using Whisper (ensure your OpenAI key has access)
#             transcription = client.audio.transcriptions.create(file=audio_file, model="whisper-1")
#             user_input = transcription.get("text", "")
#         except Exception as e:
#             return jsonify({"error": "Audio transcription failed: " + str(e)}), 500
#     elif image_file:
#         try:
#             # Use OCR (pytesseract) to extract text from the uploaded image
#             image = Image.open(image_file)
#             user_input = pytesseract.image_to_string(image)
#         except Exception as e:
#             return jsonify({"error": "Image processing failed: " + str(e)}), 500
#     else:
#         return jsonify({"error": "No input provided"}), 400

#     # Process according to the desired output type
#     if output_type in ["text", "speech"]:
#         try:
#             # Generate a text reply using ChatGPT API
#             response = client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "system", "content": "You are a helpful chatbot."},
#                     {"role": "user", "content": user_input}
#                 ],
#                 max_tokens=150
#             )
#             chatbot_response = response.choices[0].message.content.strip()
#         except Exception as e:
#             return jsonify({"error": "Chat completion failed: " + str(e)}), 500

#         if output_type == "text":
#             return jsonify({"response": chatbot_response})
#         else:
#             try:
#                 # Convert the text reply to speech using gTTS
#                 audio_filename = f"static/audio_{uuid.uuid4().hex}.mp3"
#                 tts = gTTS(text=chatbot_response, lang="en")
#                 tts.save(audio_filename)
#                 return jsonify({"response": chatbot_response, "audio_url": audio_filename})
#             except Exception as e:
#                 return jsonify({"error": "Speech synthesis failed: " + str(e)}), 500

#     elif output_type == "image":
#         try:
#             # Generate an image based on the user input (as a prompt) using the image generation API
#             image_response = client.image.generations.create(
#                 prompt=user_input,
#                 n=1,
#                 size="1024x1024"
#             )
#             # Extract the image URL (assumes response.data[0].url exists)
#             image_url = image_response.data[0].url
#             return jsonify({"response": "Image generated", "image_url": image_url})
#         except Exception as e:
#             return jsonify({"error": "Image generation failed: " + str(e)}), 500
#     else:
#         return jsonify({"error": "Invalid output type specified."}), 400

# if __name__ == "__main__":
#     app.run(debug=True)






# import os
# from openai import OpenAI
# from flask import Flask, render_template, request, jsonify

# app = Flask(__name__)

# # Create an OpenAI client using the recommended approach.
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/chat", methods=["POST"])
# def chat():
#     user_input = request.form.get("message")
#     if not user_input:
#         return jsonify({"error": "No message provided"}), 400

#     try:
#         # Use the new API call with the client
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",  # Change to your desired model if needed, e.g., "gpt-4o"
#             messages=[
#                 {"role": "system", "content": "You are a helpful chatbot."},
#                 {"role": "user", "content": user_input}
#             ],
#             max_tokens=150
#         )
#         chatbot_response = response.choices[0].message.content.strip()
#         return jsonify({"response": chatbot_response})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)




# import os
# import openai
# from flask import Flask, render_template, request, jsonify

# app = Flask(__name__)

# # Set your OpenAI API key from environment variable
# openai.api_key = os.getenv("OPENAI_API_KEY")

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/chat", methods=["POST"])
# def chat():
#     user_input = request.form.get("message")
#     if not user_input:
#         return jsonify({"error": "No message provided"}), 400

#     try:
#         # Call the ChatGPT API (gpt-3.5-turbo in this example)
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are a helpful chatbot."},
#                 {"role": "user", "content": user_input}
#             ],
#             max_tokens=150
#         )
#         chatbot_response = response.choices[0].message.content.strip()
#         return jsonify({"response": chatbot_response})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     # Running in debug mode is useful for development
#     app.run(debug=True)
