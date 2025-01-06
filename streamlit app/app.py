import streamlit as st

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to simulate a response (replace this with actual logic or a chatbot API)
def get_response(user_input):
    return f"You said: {user_input}. This is the chatbot's response."

# Sidebar for app title and instructions
st.sidebar.title("Chat App")
st.sidebar.write("This is a simple chat app built with Streamlit.")

# Main Chat UI
st.title("Chat App")

# Display previous chat messages
with st.container(height=520,border=False):
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message(name=message["role"]).write(message['content'])
        else:
            st.chat_message(name=message['role']).write(message["content"])

# Input field for user to type messages
with st.form("chat_input", clear_on_submit=True):
    user_input = st.text_input("Type your message:", "")
    submitted = st.form_submit_button("Send")

# Process user input
if submitted and user_input:
    # Add user's message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Generate chatbot's response
    bot_response = get_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

    # Refresh to display new messages
    st.rerun()

