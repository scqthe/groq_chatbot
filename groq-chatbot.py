# -- Meta Model: llama-3.2-1b-preview
# -- -- Offers 128k tokens context window
# -- -- ~3100 tps generation speed, but limited at 8k max output tokens

# -- Alibaba Model: qwen-2.5-32b
# -- -- Offers 128k tokens context window
# -- -- Heavily rate-limited at 200 tps, but uncapped max output tokens

# -- DeepSeek Model: deepseek-r1-distill-llama-70b
# -- -- Offers 128k tokens context window
# -- -- Uncapped max output tokens and slightly faster than Alibaba's at 275 tps generation speed

# -- Google Model: gemma2-9b-it
# -- -- Offers 8k tokens context window
# -- -- Uncapped max output tokens and uncapped tps

import sqlite3
import re
from datetime import datetime
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

# Load environment vars
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


# Database connection
@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect('user_data.db', check_same_thread=False)
    conn.execute('''CREATE TABLE IF NOT EXISTS contacts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  dob TEXT,
                  phone TEXT,
                  email TEXT,
                  timestamp DATETIME)''')
    return conn


# Format validation functions for contact prompts
def validate_name(name):
    return re.match(r'^[A-Za-z ]{2,30}$', name.strip()) is not None

def validate_dob(dob):
    try:
        datetime.strptime(dob, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def validate_phone(phone):
    return re.match(r'^\+?[1-9]\d{4,14}$', phone) is not None

def validate_email(email):
    return re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]{2,}$', email) is not None


# Chatbot setup functions, all cached for latency optimization
@st.cache_resource
def load_website_data(urls):
    loader = UnstructuredURLLoader(urls=urls)
    return loader.load()

@st.cache_resource
def chunk_documents(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(_docs)

@st.cache_resource
def build_vectorstore(_text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(_text_chunks, embeddings)

@st.cache_resource
def load_llm(api_key):
    return ChatGroq(
        groq_api_key=api_key,
        model_name="deepseek-r1-distill-llama-70b",
        streaming=True,
        temperature=0.1
    )

@st.cache_resource
def create_chains(_llm, _retriever):
    system_prompt = """Provide only the final answer to the query without any intermediate reasoning. 
    If asked for contact info collection, respond exactly with "COLLECT_CONTACT_INFO". 
    Context: {context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    combine_docs_chain = create_stuff_documents_chain(_llm, prompt)
    return create_retrieval_chain(retriever=_retriever, combine_docs_chain=combine_docs_chain)


# App setup
st.title("F1 Chatbot")

# State management
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'contact_info' not in st.session_state:
    st.session_state.contact_info = None
if 'resources_loaded' not in st.session_state:
    st.session_state.resources_loaded = False
if 'user_msg_count' not in st.session_state:
    st.session_state.user_msg_count = 0
if 'contact_prompt_shown' not in st.session_state:
    st.session_state.contact_prompt_shown = False

# Preload resources for URLs and vectorized chunks
if not st.session_state.resources_loaded:
    with st.spinner("Initializing system..."):
        urls = ["https://www.formula1.com/"]
        docs = load_website_data(urls)
        text_chunks = chunk_documents(docs)
        vectorstore = build_vectorstore(text_chunks)
        retriever = vectorstore.as_retriever()
        llm = load_llm(os.getenv("API_KEY"))
        qa_chain = create_chains(llm, retriever)
        st.session_state.update({
            'retriever': retriever,
            'qa_chain': qa_chain,
            'resources_loaded': True
        })

# Display chat history
for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

def handle_contact_prompt():
    if (st.session_state.user_msg_count >= 3
            and not st.session_state.contact_prompt_shown
            and not st.session_state.contact_info
            and not any(msg['content'] == "To provide better assistance..." for msg in st.session_state.conversation)):
        st.session_state.conversation.append({
            'role': 'assistant',
            'content': "To provide better assistance, would you like to share your contact information? (yes/no)"
        })
        st.session_state.contact_info = {'phase': 'confirmation'}
        st.session_state.contact_prompt_shown = True
        st.session_state.user_msg_count = 0
        st.rerun()

# Helper function for obtaining contact info from user through prompts
def handle_contact_collection(user_input):
    steps = [
        ('name', "Please enter your full name:", validate_name),
        ('dob', "Enter your date of birth (YYYY-MM-DD):", validate_dob),
        ('phone', "Enter your phone number:", validate_phone),
        ('email', "Enter your email address:", validate_email)
    ]

    if st.session_state.contact_info.get('phase') == 'confirmation':
        cleaned_input = user_input.lower().strip()
        if cleaned_input in ['yes', 'y', 'sure']:
            st.session_state.contact_info = {
                'phase': 'collection',
                'current_step': 0,
                'data': {}
            }
            st.session_state.conversation.append({'role': 'assistant', 'content': steps[0][1]})
        elif cleaned_input in ['no', 'n', 'not now']:
            st.session_state.conversation.append(
                {'role': 'assistant', 'content': "No problem! Ask me anything about F1."})
            st.session_state.contact_info = None
            st.session_state.contact_prompt_shown = True  # prevent looping future auto-prompts
        else:
            st.session_state.conversation.append({'role': 'assistant',
                                                  'content': "Please answer with 'yes' or 'no'. Would you like to share your contact information?"})
        return

    current_step = st.session_state.contact_info['current_step']
    field, prompt, validator = steps[current_step]

    if validator(user_input):
        st.session_state.contact_info['data'][field] = user_input
        st.session_state.contact_info['current_step'] += 1

        if current_step + 1 < len(steps):
            st.session_state.conversation.append({'role': 'assistant', 'content': steps[current_step + 1][1]})
        else:
            # Save to database
            with get_db_connection() as conn:
                c = conn.cursor()
                c.execute('''INSERT INTO contacts 
                           (name, dob, phone, email, timestamp)
                           VALUES (?, ?, ?, ?, ?)''',
                          (st.session_state.contact_info['data']['name'],
                           st.session_state.contact_info['data']['dob'],
                           st.session_state.contact_info['data']['phone'],
                           st.session_state.contact_info['data']['email'],
                           datetime.now()))
                conn.commit()

            st.session_state.conversation.append({
                'role': 'assistant',
                'content': "Thank you! Your information has been saved. How else can I help?"
            })
            st.session_state.contact_info = None
    else:
        st.session_state.conversation.append({
            'role': 'assistant',
            'content': f"Invalid {field} format. Please try again: {prompt}"
        })


# Non-contact / actual chat processing
if user_input := st.chat_input("Ask me about Formula 1..."):
    # Update user message count
    st.session_state.user_msg_count += 1

    st.session_state.conversation.append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Check if user is manually initiating contact info process
    manual_trigger_phrases = ["contact info", "save details", "share information"]
    if any(phrase in user_input.lower() for phrase in manual_trigger_phrases):
        if not st.session_state.contact_info:
            st.session_state.conversation.append({
                'role': 'assistant',
                'content': "Would you like to share your contact information? (yes/no)"
            })
            st.session_state.contact_info = {'phase': 'confirmation'}

    # Check if contact process is automatically registering
    handle_contact_prompt()

    if st.session_state.contact_info:
        handle_contact_collection(user_input)
    else:
        # Process query
        with st.spinner(""):
            response = st.session_state.qa_chain.invoke({"input": user_input})

        # Attempt to trim output so DeepSeek model's DeepThink text isn't dumped
        answer = response.get("answer", "I need to check that information. Could you please rephrase?")
        answer = answer.split("Final Answer:")[-1].strip()

        st.session_state.conversation.append({'role': 'assistant', 'content': answer})

        # Display out
        with st.chat_message("assistant"):
            st.write(answer)

    # Trigger UI update
    st.rerun()
