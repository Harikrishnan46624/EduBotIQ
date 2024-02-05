import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers

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

# Chat history to store conversation
chat_history = []

# Streamlit app
st.title("Chatbot with History")

# Using Streamlit's st.form for a more reactive approach
with st.form("user_input_form"):
    # User input
    user_input = st.text_input("Input Prompt:")

    # Submit button
    submit_button = st.form_submit_button("Submit")

# Process user input and update chat history when the form is submitted
if submit_button:
    # Add user input to chat history
    chat_history.append({"user": user_input, "role": "user"})
    print("Question: ", user_input)

    # Truncate user input to fit within the maximum context length
    user_input = user_input[:512]

    # Get bot's response
    result = qa({"query": user_input})
    bot_response = result["result"]
    print("Result: ", result)

    # Add bot's response to chat history
    chat_history.append({"bot": bot_response, "role": "bot"})

# Display chat history vertically
st.text("Chat History:")
for entry in chat_history:
    if entry["role"] == "user":
        st.text(f"User: {entry['user']}")
    elif entry["role"] == "bot":
        # Display bot's response in a paragraph format
        st.write(f"Bot: {entry['bot']}")

