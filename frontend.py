import streamlit as st
from rag_pipeline import answer_query, retrieve_docs, llm_model

# Title
st.title("🧠 Blaze Q&A Chatbot")

# Input box for user query
user_query = st.text_area("Ask a question:", height=150, placeholder="Type your question here...")

# Button to send the query
ask_question = st.button("Ask BLAZE")

# When button is pressed
if ask_question and user_query.strip():
    st.chat_message("user").write(user_query)

    # Call the RAG pipeline
    retrieved_docs = retrieve_docs(user_query)
    response = answer_query(documents=retrieved_docs, model=llm_model, query=user_query)

    st.chat_message("AI Lawyer").write(response)
elif ask_question:
    st.error("Please enter a valid question.")
