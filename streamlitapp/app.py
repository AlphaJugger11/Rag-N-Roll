import streamlit as st
from huggingface_hub import InferenceApi  # Change this line to use InferenceApi

# Initialize the Hugging Face Inference client
client = InferenceApi(repo_id="mistralai/Mistral-Nemo-Instruct-2407", token=st.secrets["API_KEY"])

# Initial page config
st.set_page_config(
    page_title="Chat App",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"  # Sidebar starts in collapsed state
)

# Streamlit app layout
st.sidebar.title("Chat App")
st.sidebar.write("This is a simple chat app built with Streamlit.")

st.title("Interactive Chatbot")

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to generate a response using the Hugging Face model
def get_response(messages):
    """
    Generate a response using the Hugging Face Mistral model.
    """
<<<<<<<< HEAD:app.py
    completion = client.chat.completions.create(
        model="mistralai/Mistral-Nemo-Instruct-2407",
        messages=messages,
        max_tokens=4096
    )
========
    completion = client.chat(messages=messages, max_tokens=500)  # Use the correct method
>>>>>>>> deb246e7d313b76e28a7866932300fe94e9ac738:streamlitapp/app.py
    return completion["choices"][0]["message"]["content"]

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
