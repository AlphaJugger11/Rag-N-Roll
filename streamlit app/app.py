
import streamlit as st

# Title and basic text
st.title("My Streamlit App")
st.write("Welcome to my Streamlit app hosted on Streamlit Community Cloud!")

# Example interactive component
name = st.text_input("Enter your name")
if name:
    st.write(f"Hello, {name}!")
