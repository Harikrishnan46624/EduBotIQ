import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from audio_recorder_streamlit import audio_recorder
import requests
import os


# Download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

embeddings = download_hugging_face_embeddings()

db = FAISS.load_local("vectorstore/db_faiss", embeddings=embeddings)

# prompt_template = """
# Use the following pieces of information to answer the user's question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.

# Context: {context}
# Question: {question}

# Only return the helpful answer below and nothing else.
# Helpful answer:
# """


prompt_template = """
Use the following pieces of information to answer the student's question.
If you don't know the answer, mention that the question is beyond the scope of your expertise.

Context: {context}
Question: {question}

Only return the answers related to artificial intelligence below and nothing else.
Helpful answer:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}


#Small tiny model
llm = CTransformers(model=r"E:\projects\EduBotIQ\tiny_model\tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
                    model_type="llama",
                    config={'max_new_tokens': 512, 'temperature': 0.5})




qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)



# Streamlit app
st.title("EduBotIQ")

# Using Streamlit's st.form for a more reactive approach
with st.form("user_input_form"):
    # User input
    text_input = st.text_input("Input Prompt:")

    audio_input = audio_recorder()

    # Submit button
    submit_button = st.form_submit_button("Submit")

if submit_button:
    print("User: ", text_input)
    result = qa({"query": text_input})
    print("Response : ", result["result"])
    response = str(result["result"])
    st.write(response)

API_URL = os.getenv("API_URL")
headers = os.getenv("headers")


def convert_audio_to_text(audio_data):
    response = requests.post(API_URL, headers=headers, data=audio_data)
    return response.json()['text']


if audio_input is not None:
    audio_file_path = "audio_data/audio.wav"
    with open(audio_file_path, "wb") as audio_file:
        audio_file.write(audio_input)

    audio_to_text = convert_audio_to_text(audio_input)
    print(audio_to_text)

    input_text = audio_to_text
    result = qa({"query": input_text})
    response = str(result["result"])
    st.write(response) 