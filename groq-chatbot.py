import sys
import asyncio

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables FIRST
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

groq_api_key = os.getenv("API_KEY")
if not groq_api_key:
    st.error("Groq API Key not found in .env file")
    st.stop()
else:
    print("Success!")

def load_website_data(urls):
    loader = UnstructuredURLLoader(urls=urls)
    return loader.load()


def chunk_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(docs)


def build_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(text_chunks, embeddings)


def load_llm(api_key):
    return ChatGroq(groq_api_key=api_key, model_name="qwen-2.5-32b", streaming=True)


st.title("Custom Website Chatbot (Analytics Vidhya)")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

urls = ["https://www.analyticsvidhya.com/"]
docs = load_website_data(urls)
text_chunks = chunk_documents(docs)

vectorstore = build_vectorstore(text_chunks)
retriever = vectorstore.as_retriever()

llm = load_llm(groq_api_key)

system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use detailed sentences maximum and keep the answer accurate. "
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=combine_docs_chain
)

# Chat Interface
for msg in st.session_state.conversation:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["message"])
    else:
        st.chat_message("assistant").write(msg["message"])

# User input handling
user_input = st.chat_input("Type your message here") if hasattr(st, "chat_input") else st.text_input("Your message:")
if user_input:
    st.session_state.conversation.append({"role": "user", "message": user_input})
    if hasattr(st, "chat_message"):
        st.chat_message("user").write(user_input)
    else:
        st.markdown(f"**User:** {user_input}")

    with st.spinner("Processing..."):
        response = qa_chain.invoke({"input": user_input})

    assistant_response = response.get("answer", "I'm not sure, please try again.")
    st.session_state.conversation.append({"role": "assistant", "message": assistant_response})

    if hasattr(st, "chat_message"):
        st.chat_message("assistant").write(assistant_response)
    else:
        st.markdown(f"**Assistant:** {assistant_response}")