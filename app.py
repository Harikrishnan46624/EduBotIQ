from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from src.prompt import *
from langchain_community.llms import CTransformers
import os


app = Flask(__name__)


embeddings = download_hugging_face_embeddings()


DB_FAISS_PATH = 'vectorstore/db_faiss'


# Load the FAISS database with the embeddings
db = FAISS.load_local("vectorstore/db_faiss", embeddings=embeddings)


PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


chain_type_kwargs = {"prompt": PROMPT}


llm = CTransformers(model=r"E:\projects\EduBotIQ\tiny_model\tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
                    model_type="llama",
                    config={'max_new_tokens': 512, 'temperature': 0.3})


qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=db.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)


@app.route('/')
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug=False)