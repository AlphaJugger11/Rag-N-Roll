import streamlit as st
from huggingface_hub import InferenceClient

# Initialize the Hugging Face Inference client
client = InferenceClient(api_key=st.secrets["API_KEY"])

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
    completion = client.chat.completions.create(
        model="mistralai/Mistral-Nemo-Instruct-2407",
        messages=messages,
        max_tokens=500
    )
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
