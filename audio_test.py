# @app.route("/audioget", methods=["POST"])
# def audio_get():
#     audio_data = request.form["msg"]

#     audio_file_path = "audio_data/audio.wav"  
#     with open(audio_file_path, "wb") as audio_file:
#         audio_file.write(audio_data.encode('utf-8'))

#     converted_text = query(audio_file_path)

#     # Your existing code for generating a response
#     input_text = converted_text
#     result = qa({"query": input_text})
#     response_text = result["result"]

#     return str(result["result"])



import requests

API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v2"
headers = {"Authorization": "Bearer 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"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()['text']

# output = query("sample1.flac")
# print(output)

def convert_audio_to_text(audio_file_path):
    # Implement audio-to-text conversion logic using a library like SpeechRecognition
    # For example:
    # import speech_recognition as sr
    # recognizer = sr.Recognizer()
    # with sr.AudioFile(audio_file_path) as source:
    #    audio_data = recognizer.record(source)
    #    text = recognizer.recognize_google(audio_data)
    #    return text
    pass
