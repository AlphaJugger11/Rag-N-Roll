import streamlit as st
import pandas as pd
import snowflake
from snowflake.snowpark.context import get_active_session
from snowflake import snowpark
from huggingface_hub import InferenceClient
# session = get_active_session()

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
st.sidebar.write("This is a simple chat app built with Streamlit.")

st.title("Interactive Chatbot")

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# *****************************Adding functions*********************************
num_chunks = 3
def create_prompt (myquestion, rag=1):
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
        
        context_lenght = len(df_context) -1

        prompt_context = ""
        for i in range (0, context_lenght):
            prompt_context += df_context._get_value(i, 'CHUNK')

        prompt_context = prompt_context.replace("'", "")
        relative_path =  df_context._get_value(0,'RELATIVE_PATH')
    
        prompt = f"""
          'You are an expert legal assistance extracting information from context provided. 
           Answer the question based on the context. Be concise and do not hallucinate. 
           If you don´t have the information just say so.
          Context: {prompt_context}
          Question:  
           {myquestion} 
           Answer: '
           """
        cmd2 = f"select GET_PRESIGNED_URL(@docs, '{relative_path}', 360) as URL_LINK from directory(@docs)"
        df_url_link = session.sql(cmd2).to_pandas()
        url_link = df_url_link._get_value(0,'URL_LINK')

    else:
        prompt = f"""
         'Question:  
           {myquestion} 
           Answer: '
           """
        url_link = "None"
        relative_path = "None"
      
    return prompt, url_link, relative_path

def complete(myquestion, model_name, rag = 1):
    prompt, url_link, relative_path =create_prompt (myquestion, rag)
    # cmd = f"""
    #          select SNOWFLAKE.CORTEX.COMPLETE(?,?) as response
    #        """   
    # df_response = session.sql(cmd, params=['mistral-large', prompt]).collect()
    messages = [{
    		"role": "user",
		"content": prompt

    }]
    # client = InferenceClient(api_key = st.secrets['API_KEY'])
    completion = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.3", 
	        messages=messages, 
	        max_tokens=500
    )
    return completion.choices[0].message
    # return df_response, url_link, relative_path

def get_response (question):
    model = 'mistral-large'
    rag = 1
    response, url_link, relative_path = complete(question, model, rag)
    res_text = response[0].RESPONSE
    st.markdown(res_text)
    if rag == 1:
        display_url = f"Link to [{relative_path}]({url_link}) that may be useful"
        st.markdown(display_url)
    return res_text

# Function to generate a response using the Hugging Face model
# def get_response(messages):
#     """
#     Generate a response using the Hugging Face Mistral model.
#     """
#     completion = client.chat.completions.create(
#         model="mistralai/Mistral-Nemo-Instruct-2407",
#         messages=messages,
#         max_tokens=4096
#     )
#     return completion["choices"][0]["message"]["content"]

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
        bot_response = get_response(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

    # Display the chatbot's response
    st.chat_message("Assistant").write(bot_response)
