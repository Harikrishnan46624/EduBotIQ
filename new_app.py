import logging
import os
from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from src.prompt import *
from langchain_community.llms import HuggingFaceHub



app = Flask(__name__)

log_directory = 'logs'
os.makedirs(log_directory, exist_ok=True)

# Configure logging to write to both file and terminal
log_file_path = os.path.join(log_directory, 'app_log.log')
logging.basicConfig(filename=log_file_path, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Redirect Flask logs to the terminal
log = logging.getLogger('werkzeug')
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


# Load the Sentence embeddings
embeddings = download_hugging_face_embeddings()

# Load the FAISS database with the embeddings
DB_FAISS_PATH = 'vectorstore/db_faiss'

db = FAISS.load_local("vectorstore/db_faiss", embeddings=embeddings)


PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


chain_type_kwargs = {"prompt": PROMPT}


model_id = 'Harikrishnan46624/finetuned_llama2-1.1b-chat'
llm = HuggingFaceHub(huggingfacehub_api_token='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',
                     repo_id=model_id,
                     model_kwargs={'max_new_tokens': 256, 'temperature': 0.3})



#Setup RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type = "stuff", 
    retriever = db.as_retriever(search_kwargs={'k': 2}),
    return_source_documents = True, 
    chain_type_kwargs = chain_type_kwargs)


@app.route('/')
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    input_msg = request.form["msg"]
    
    logging.info(f"User Input: {input_msg}")
    print("User: ", input_msg)

    result = qa({"query": input_msg})
    print("Full Result:", result)
    
    model_response = result.get("result", "")
    
    # Assuming the prompt is at the beginning of the response
    prompt_length = len(PROMPT.render({"context": "", "question": ""}))
    extracted_response = model_response[prompt_length:].replace('\n', ' ')

    print("Response: ", extracted_response)
    logging.info(f"Response: {extracted_response}")
    
    # Return the extracted response directly
    return str(extracted_response)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug=False)