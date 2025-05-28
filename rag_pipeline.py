from langchain_groq import ChatGroq
from vector_database import faiss_db
from langchain_core.prompts import ChatPromptTemplate

# Uncomment the following if you're NOT using pipenv
from dotenv import load_dotenv
load_dotenv()

#Step1: Setup LLM (Use DeepSeek R1 with Groq)
llm_model=ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0.1)

#Step2: Retrieve Docs

def retrieve_docs(query):
    return faiss_db.similarity_search(query)

def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context

#Step3: Answer Question

custom_prompt_template = """
Answer the user's question using only the context provided below.

If the answer is not in the context, say: "I don’t know based on the available information." 
Do not guess or make up information. Be brief and clear.

Keep the response friendly and easy to follow. Avoid long sentences or technical jargon.


Always speak directly to the user. Use "you" instead of "the user."
Question: {question} 
Context: {context} 
Answer:
"""

def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    return chain.invoke({"question": query, "context": context})

''''question="Blaze answers on Twitter"
retrieved_docs=retrieve_docs(question)
print("AI Lawyer: ",answer_query(documents=retrieved_docs, model=llm_model, query=question))'''