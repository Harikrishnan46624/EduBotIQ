import logging
import os
from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from src.prompt import *
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import Chroma



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


# Load the Chroma database with the embeddings
# directory = 'database'
# db = Chroma(persist_directory=directory, embedding_function=embeddings)


PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


chain_type_kwargs = {"prompt": PROMPT}


#Load llmm model tiny lamma model
model_path = r"E:\projects\EduBotIQ\tiny_model\tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
llm = CTransformers(model=model_path,
                    model_type="llama",
                    config={'max_new_tokens': 256, 'temperature': 0.3})


#Setup RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type = "stuff", 
    retriever = db.as_retriever(search_kwargs={'k': 2}),
    return_source_documents = True, 
    chain_type_kwargs = chain_type_kwargs)


# from langchain.memory import ConversationBufferMemory

# prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template,)

# memory = ConversationBufferMemory(memory_key="history", input_key="question")

# chain_type_kwargs = {"verbose": True, "prompt": prompt, "memory":memory}

# retriever = db.as_retriever(search_kwargs={'k': 2})

# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type='stuff',
#     retriever=retriever,
#     verbose=True,
#     chain_type_kwargs=chain_type_kwargs
# )


@app.route('/')
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    input = request.form["msg"]
    
    logging.info(f"User Input: {input}")
    print("User: ", input)

    result = qa({"query": input})
    print("Response: ", result["result"])
    logging.info(f"Response: {result['result']}")
    return str(result["result"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug=False)