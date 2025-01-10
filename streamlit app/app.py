# import streamlit as st
# import tensorflow as tf
# from transformers import AutoTokenizer, TFAutoModelForCausalLM, pipeline

# # Replace with the actual Mistral model name if available in TF or convertible from PyTorch
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

# # 1. Load the tokenizer and model, with optional from_pt=True if no native TF weights exist
# @st.cache_resource  # Cache so we don't re-load the model on every interaction
# def load_model():
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     # Try to load in TF directly; if it fails because no TF weights exist, set from_pt=True.
#     model = TFAutoModelForCausalLM.from_pretrained(MODEL_NAME, from_pt=True)
    
#     # Initialize a Transformers pipeline that runs on TensorFlow
#     text_gen_pipeline = pipeline(
#         task="text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         framework="tf"  # Explicitly use TensorFlow backend
#     )
#     return text_gen_pipeline

# # 2. Initialize pipeline
# chat_pipeline = load_model()

# st.sidebar.title("Chat App")
# st.sidebar.write("This is a simple chat app built with Streamlit and a TensorFlow-based Mistral model.")

# st.title("Chat App")

# # Initialize session state for chat messages
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# def get_response(user_input):
#     """
#     Generate a response using the local Mistral model under TensorFlow/Keras.
#     Adjust generation parameters (e.g., max_new_tokens, temperature, top_p) as desired.
#     """
#     outputs = chat_pipeline(
#         user_input,
#         max_new_tokens=128,
#         do_sample=True,
#         temperature=0.7,
#         top_p=0.9
#     )
#     # The pipeline returns a list of dicts like:
#     # [ { 'generated_text': 'Prompt + completion...' } ]
#     generated_text = outputs[0]["generated_text"]

#     # If the pipeline returns the prompt + completion, you may want to remove
#     # the original user prompt from the front:
#     # generated_text = generated_text[len(user_input):].strip()

#     return generated_text

# # Display previous chat messages
# with st.container():
#     for message in st.session_state.messages:
#         if message["role"] == "user":
#             st.chat_message("User").write(message["content"])
#         else:
#             st.chat_message("Assistant").write(message["content"])

# # Input field for user to type messages
# with st.form("chat_input", clear_on_submit=True):
#     user_input = st.text_input("Type your message:", "")
#     submitted = st.form_submit_button("Send")

# # Process user input
# if submitted and user_input:
#     # Add user's message to session state
#     st.session_state.messages.append({"role": "user", "content": user_input})
    
#     # Generate chatbot's response
#     bot_response = get_response(user_input)
#     st.session_state.messages.append({"role": "assistant", "content": bot_response})

#     # Refresh to display new messages
#     st.experimental_rerun(
import streamlit as st
from huggingface_hub import InferenceClient

# Initialize the Hugging Face Inference client
client = InferenceClient(api_key="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")

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
