


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