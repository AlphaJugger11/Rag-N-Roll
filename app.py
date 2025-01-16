import streamlit as st
import pandas as pd
import snowflake
from snowflake.snowpark.context import get_active_session
from snowflake import snowpark
from huggingface_hub import InferenceClient
# session = get_active_session()
import time
import random
import hashlib

def generate_unique_key():
    # Get a high-resolution timestamp (microseconds)
    timestamp_us = int(time.time() * 1_000_000)
    
    # Generate a random number
    random_num = random.randint(0, 999999999)
    
    # Combine them into a string
    raw_string = f"{timestamp_us}-{random_num}"
    
    # Optionally hash the combined string to get a shorter or more uniform key
    unique_hash = hashlib.sha256(raw_string.encode()).hexdigest()
    
    return unique_hash

st.set_page_config(
    page_title="Chat App",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"  # Sidebar starts in collapsed state
)

connection_params = {
    "account": st.secrets["snowflake"]["account"],
    "user": st.secrets["snowflake"]["user"],
    "password": st.secrets["snowflake"]["password"],
    "role": st.secrets["snowflake"]["role"],
    "database": st.secrets["snowflake"]["database"],
    "schema": st.secrets["snowflake"]["schema"],
    "warehouse": st.secrets["snowflake"]["warehouse"]
}
## just to check the changes
# st.write(st.secrets)

# Establishing Snowflake session
session = snowpark.Session.builder.configs(connection_params).create()

# Initialize the Hugging Face Inference client
client = InferenceClient(api_key=st.secrets["API_KEY"])

# Initial page config


# Streamlit app layout
st.sidebar.title("Chat App")
st.sidebar.write(st.session_state.summary)

st.title("Interactive Chatbot")

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.UserPrompt = []  #user prompts
    st.session_state.BotResponse = []     #bot response
    st.session_state.chatpk=generate_unique_key()
    st.session_state.summary = ""



slide_window = 2 # no of last conversations to keep

def get_chat_history():
    #Get the history from the st.session_stage.messages according to the slide window parameter 
    chat_history = []
    
    start_index = max(0, len(st.session_state.messages) - slide_window)
    for i in range (start_index , len(st.session_state.messages) -1):
         chat_history.append({
             "User": st.session_state.UserPrompt[i],
             "Response": st.session_state.BotResponse[i]})

    return chat_history

def summarize_question_with_history(chat_history, question):
# To get the right context, use the LLM to first summarize the previous conversation
# This will be used to get embeddings and find similar chunks in the docs for context

    prompt = f"""
        Based on the chat history below and the question, generate a query that extend the question
        with the chat history provided. The query should be in natual language. 
        Answer with only the query. Do not add any explanation.
        
        <chat_history>
        {chat_history}
        </chat_history>
        <question>
        {question}
        </question>
        """

    st.session_state.summary = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3", 
        messages=prompt, 
        max_tokens=4096
    ).choices[0].message
    

# *****************************Adding functions*********************************
num_chunks = 10
def create_prompt(myquestion, rag=1):
    # st.write(myquestion)
    # st.write(type(myquestion))


    if isinstance(myquestion, dict):
        myquestion = myquestion.get("content", "")  # Extract content if myquestion is a dict

    if rag == 1:    
        cmd = """
        with results as
        (SELECT RELATIVE_PATH,
          VECTOR_COSINE_SIMILARITY(docs_chunks_table.chunk_vec,
                   SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m', ?)) as similarity,
          chunk
        from docs_chunks_table
        order by similarity desc
        limit ?)
        select chunk, relative_path from results 
        """
        
        df_context = session.sql(cmd, params=[myquestion, num_chunks]).to_pandas()
        
        prompt_context = "  ".join(df_context["CHUNK"].astype(str))  # Merge all chunks into one string
        # st.write(prompt_context)
        relative_path = df_context["RELATIVE_PATH"].iloc[0]  # Get the first relative path
        # st.write(relative_path)
        
        prompt = f"""
        'You are an expert legal assistant extracting information from context provided. 
        Answer the question based on the context.The context is not visible to the user. The context should be reffered to as your knowledge.
        use the context to answer questions where applicable.Be concise and do not hallucinate. 
        If you don’t have the information just say so.
        Context: {prompt_context}
        Question: {myquestion}
        Answer:'
        """
        
        cmd2 = f"select GET_PRESIGNED_URL(@docs, '{relative_path}', 360) as URL_LINK from directory(@docs)"
        df_url_link = session.sql(cmd2).to_pandas()
        url_link = df_url_link["URL_LINK"].iloc[0]
    else:
        prompt = f"Question: {myquestion} Answer: '"
        url_link = "None"
        relative_path = "None"
        
    return prompt, url_link, relative_path


def complete(myquestion, model_name, rag=1):
    # st.write(myquestion)
    # st.write(type(myquestion))

    if isinstance(myquestion, list):  # Check if myquestion is a list of messages
        myquestion = myquestion[-1]["content"]  # Extract the latest message content

    prompt, url_link, relative_path = create_prompt(myquestion, rag)
    st.session_state.UserPrompt.append({"role": "user", "content": prompt})
    # Hugging Face Inference
    completion = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.3", 
        messages=prompt, 
        max_tokens=4096
    )
    return completion.choices[0].message

    # return df_response, url_link, relative_path

def get_response(question):
    # st.write(question)
    # st.write(type(question))

    model = 'mistral-large'
    rag = 1
    response = complete(question, model, rag)
    # st.markdown(response)
    return response


# Display previous chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("User").write(message["content"])
    else:
        st.chat_message("Assistant").write(message["content"])

# Input field for user to type a message
if user_input := st.chat_input("Type your message:"):
    # Add user's message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("User").write(user_input)
    # Generate chatbot's response
    with st.spinner("Thinking..."):
        # st.write('here')
        bot_response =(get_response(st.session_state.messages).content )
        # st.write('here')
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        st.session_state.BotResponse.append({"role": "assistant", "content": bot_response})
        # st.write('here')

    # Display the chatbot's response
    st.chat_message("Assistant").write(bot_response)

    cmd3 = """
     INSERT INTO CONVERSATION_HISTORY (USER_PROMPT, RESPONSE, CHAT_ID )
                VALUES (?, ?,?)
    """
    session.sql(cmd3, params=[user_input, bot_response,st.session_state.chatpk]).collect()
