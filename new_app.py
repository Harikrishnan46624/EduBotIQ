import logging
import os
from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from dotenv import load_dotenv
import langchain
load_dotenv()
import re



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



def chain(input:str):
    model_id = 'Harikrishnan46624/finetuned_llama2-1.1b-chat'
    llm = HuggingFaceHub(huggingfacehub_api_token=os.getenv('HuggingFaceAPI'),
                        repo_id=model_id,
                        model_kwargs={'max_new_tokens': 256, 'temperature': 0.3})

    prompt_template = """
    Use the following pieces of information to answer the student's question.
    If you don't know the answer or if the question is outside the scope of artificial intelligence, mention that the question is beyond the scope of your expertise.

    Question: {question}


    Only return the answers related to artificial intelligence below and nothing else.
    Helpful answer:
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["question"])

    chain = LLMChain(llm=llm,prompt=PROMPT,verbose=False)
    answer = chain.run(input)
    

    # # Extract only the answer from the result string using regex
    # answer_pattern = re.compile(r'Answer: (.+)', re.DOTALL)
    # match = answer_pattern.search(result_string)

    # answer = match.group(1) if match else ''

    return answer
 


@app.route('/')
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        input_message = request.form["msg"]
        logging.info(f"User Input: {input_message}")

        print("User: ", input_message)
        
        langchain.verbose = True

        result = chain(str(input_message))

        print("Response: ", result)
        logging.info(f"Response: {result}")

        return str(result)

    except Exception as e:
        logging.exception(f"An error occurred: {str(e)}")
        return "An error occurred."



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug=False)