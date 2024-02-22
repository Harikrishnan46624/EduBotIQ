# # @app.route("/audioget", methods=["POST"])
# # def audio_get():
# #     audio_data = request.form["msg"]

# #     audio_file_path = "audio_data/audio.wav"  
# #     with open(audio_file_path, "wb") as audio_file:
# #         audio_file.write(audio_data.encode('utf-8'))

# #     converted_text = query(audio_file_path)

# #     # Your existing code for generating a response
# #     input_text = converted_text
# #     result = qa({"query": input_text})
# #     response_text = result["result"]

# #     return str(result["result"])



# import requests
# import os
# import json

# API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v2"
# headers_key = "headers"
# headers_value = os.getenv(headers_key)

# # Check if headers_value is None and provide a default value if needed
# headers = json.loads(headers_value) if headers_value else {}

# def query(filename):
#     with open(filename, "rb") as f:
#         data = f.read()
#     response = requests.post(API_URL, headers=headers, data=data)
    
#     try:
#         response_json = response.json()
#         print(response_json)  # Print the entire JSON response for debugging
#         return response_json['text']
#     except KeyError:
#         print("Error: 'text' key not found in the JSON response.")
#         return None

# output = query("Recording.m4a")
# if output:
#     print(output)
# else:
#     print("Failed to get the output.")




import requests

API_URL = "https://api-inference.huggingface.co/models/Harikrishnan46624/finetuned_llama2-1.1b-chat"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "What is NLP?",
})


print(output)
