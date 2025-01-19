import streamlit as st
import pandas as pd
import snowflake
from snowflake.snowpark.context import get_active_session
from snowflake import snowpark
from huggingface_hub import InferenceClient

import time
import random
import hashlib

# --------------------------------------------------------------------------------
# Utility function to generate a unique hash-based key
# --------------------------------------------------------------------------------
def generate_unique_key():
    timestamp_us = int(time.time() * 1_000_000)
    random_num = random.randint(0, 999999999)
    raw_string = f"{timestamp_us}-{random_num}"
    unique_hash = hashlib.sha256(raw_string.encode()).hexdigest()
    return unique_hash

# --------------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Chat App",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------------
# Snowflake connection
# --------------------------------------------------------------------------------
connection_params = {
    "account": st.secrets["snowflake"]["account"],
    "user": st.secrets["snowflake"]["user"],
    "password": st.secrets["snowflake"]["password"],
    "role": st.secrets["snowflake"]["role"],
    "database": st.secrets["snowflake"]["database"],
    "schema": st.secrets["snowflake"]["schema"],
    "warehouse": st.secrets["snowflake"]["warehouse"]
}
session = snowpark.Session.builder.configs(connection_params).create()

# --------------------------------------------------------------------------------
# Hugging Face Inference client
# --------------------------------------------------------------------------------
client = InferenceClient(api_key=st.secrets["API_KEY"])

# --------------------------------------------------------------------------------
# Session State Initialization
# --------------------------------------------------------------------------------
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {}

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

# --------------------------------------------------------------------------------
# Function to create a new chat
# --------------------------------------------------------------------------------
def add_new_chat():
    new_id = generate_unique_key()
    chat_number = len(st.session_state.all_chats) + 1
    st.session_state.all_chats[new_id] = {
        "title": f"Chat {chat_number}",
        "messages": [],       # list of {"role": "user"/"assistant", "content": "..."}
        "UserPrompt": [],
        "BotResponse": [],
        "chatpk": new_id,
        "summary": ""
    }
    # Set new chat as active
    st.session_state.active_chat_id = new_id

# If no chats exist, create the first chat automatically
if not st.session_state.all_chats:
    add_new_chat()

# --------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------
if st.sidebar.button("New Chat"):
    add_new_chat()

st.sidebar.header("Chats")
st.sidebar.write("_________________")

# List existing chats in the sidebar as clickable items
for chat_id, chat_data in st.session_state.all_chats.items():
    if st.sidebar.button(chat_data["title"], use_container_width=True):
        st.session_state.active_chat_id = chat_id

st.title("Interactive Chatbot")

# --------------------------------------------------------------------------------
# Retrieve the active chat
# --------------------------------------------------------------------------------
if st.session_state.active_chat_id:
    active_chat = st.session_state.all_chats[st.session_state.active_chat_id]
else:
    st.write("No active chat selected. Use the sidebar to create or pick a chat.")
    st.stop()

# --------------------------------------------------------------------------------
# (Optional) RAG / Summarization Helper Functions
# --------------------------------------------------------------------------------
slide_window = 2  # number of last user->assistant pairs to keep

def get_chat_history(chat_data):
    relevant_messages = chat_data["messages"][-(2*slide_window):]
    user_bot_pairs = []
    pair = {}
    for msg in relevant_messages:
        if msg["role"] == "user":
            pair["User"] = msg["content"]
        elif msg["role"] == "assistant":
            pair["Response"] = msg["content"]
            user_bot_pairs.append(pair)
            pair = {}
    return user_bot_pairs

def summarize_question_with_history(chat_history, question):
    prompt = f"""
    Based on the chat history below and the question, generate a query that extends the question
    with the chat history provided. The query should be in natural language.
    Answer with only the query. Do not add any explanation.

    Chat_history: {chat_history}
    User_Query: {question}
    """
    summary_resp = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3", 
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096
    ).choices[0].message
    return summary_resp

num_chunks = 10

def similar_chunks(myquestion):
    if isinstance(myquestion, dict):
        myquestion = myquestion.get("content", "")
   
    cmd = """
    WITH results AS (
        SELECT 
            RELATIVE_PATH,
            VECTOR_COSINE_SIMILARITY(docs_chunks_table.chunk_vec,
               SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m', ?)) as similarity,
            chunk
        FROM docs_chunks_table
        ORDER BY similarity DESC
        LIMIT ?
    )
    SELECT chunk, relative_path FROM results 
    """
    
    df_context = session.sql(cmd, params=[myquestion, num_chunks]).to_pandas()
    
    prompt_context1 = "  ".join(df_context["CHUNK"].astype(str))
    relative_path = df_context["RELATIVE_PATH"].iloc[0]
    
    prompt_context = f"""
    'You are an expert legal assistant extracting information from context provided. 
    Answer the question based on the context. The context is not visible to the user. The context should be referred to as your knowledge.
    Use the context to answer questions where applicable. Be concise and do not hallucinate. 
    If you don’t have the information just say so.
    Context: {prompt_context1}
    Question: {myquestion}
    Answer:'
    """
    
    cmd2 = f"SELECT GET_PRESIGNED_URL(@docs, '{relative_path}', 360) as URL_LINK FROM directory(@docs)"
    df_url_link = session.sql(cmd2).to_pandas()
    url_link = df_url_link["URL_LINK"].iloc[0]

    return prompt_context, url_link, relative_path

def create_prompt(myquestion, chat_data, rag=1):
    chat_history = get_chat_history(chat_data)
    if chat_history:
        question_with_context = summarize_question_with_history(chat_history, myquestion)
        prompt_context, url_link, relative_path = similar_chunks(question_with_context)
    else:
        prompt_context, url_link, relative_path = similar_chunks(myquestion)

    prompt = f"""
    You are an expert chat assistant that extracts information from the CONTEXT 
    provided between <context> and </context> tags.
    You consider the CHAT HISTORY provided between <chat_history> and </chat_history> tags.
    When answering the question contained between <question> and </question> tags:
        - Be concise and do not hallucinate.
        - If you don’t have the information, just say so.
        - Do not mention the CONTEXT or the CHAT HISTORY explicitly in your answer.

    Chat_history: {chat_history}
    context: {prompt_context}
    User_Query: {myquestion}
    Answer:
    """
    return prompt, url_link, relative_path

def complete(myquestion, chat_data, model_name="mistral-large", rag=1):
    if isinstance(myquestion, list):
        myquestion = myquestion[-1]["content"]

    prompt, url_link, relative_path = create_prompt(myquestion, chat_data, rag)
    chat_data["UserPrompt"].append({"role": "user", "content": prompt})

    completion = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3", 
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096
    )
    return completion.choices[0].message

def get_response(question, chat_data):
    return complete(question, chat_data).content

# --------------------------------------------------------------------------------
# Function to create a 3-word summary using Mistral
# --------------------------------------------------------------------------------
def threeWordSummary(chat_id):
    """
    Summarize the entire conversation of the chat in exactly 3 "words" (no punctuation).
    Then, replace the chat's title with that summary.
    This happens only once when the chat's total messages reach 6.
    """
    chat_data = st.session_state.all_chats[chat_id]
    
    # Combine all messages into a single string
    conversation_text = ""
    for msg in chat_data["messages"]:
        # optional formatting
        conversation_text += f"{msg['role'].title()}: {msg['content']}\n"

    prompt_for_summary = f"""
    Summarize the conversation below in exactly 3 words. 
    Do not add any explanation or punctuation:

    {conversation_text}
    """
    # Call Mistral to get the summary
    completion = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=[{"role": "user", "content": prompt_for_summary}],
        max_tokens=20
    )
    three_word_summary = completion.choices[0].message["content"].strip()
    
    # Replace the chat's title
    st.session_state.all_chats[chat_id]["title"] = three_word_summary

# --------------------------------------------------------------------------------
# Display existing messages (for the active chat)
# --------------------------------------------------------------------------------
for message in active_chat["messages"]:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

# --------------------------------------------------------------------------------
# Chat input for the active chat
# --------------------------------------------------------------------------------
if user_input := st.chat_input("Type your message:"):
    # 1) Add user's message
    active_chat["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # 2) Generate chatbot's response
    with st.spinner("Thinking..."):
        bot_response = get_response(active_chat["messages"], active_chat)
        active_chat["messages"].append({"role": "assistant", "content": bot_response})
        st.chat_message("assistant").write(bot_response)

    # 3) Insert into Snowflake conversation history (optional)
    insert_cmd = """
        INSERT INTO CONVERSATION_HISTORY (USER_PROMPT, RESPONSE, CHAT_ID)
        VALUES (?, ?, ?)
    """
    session.sql(insert_cmd, params=[user_input, bot_response, active_chat["chatpk"]]).collect()

    # 4) Check if total messages == 6, then create 3-word summary *once*
    if len(active_chat["messages"]) == 6:
        threeWordSummary(st.session_state.active_chat_id)
