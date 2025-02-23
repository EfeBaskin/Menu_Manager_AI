import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.title("Menu Manager AI")

st.header("Upload Data File")
uploaded_file = st.file_uploader("Upload a CSV or XLSX file", type=["csv", "xlsx"])
if uploaded_file is not None:
    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
    response = requests.post(f"{API_URL}/upload", files=files)
    if response.status_code == 200:
        st.success(response.json().get("message", "File uploaded and dataframe updated."))
    else:
        error_detail = response.json().get("detail", "Unknown error")
        st.error(f"Error uploading file: {error_detail}")

st.header("Enter Your Prompt")
user_prompt = st.text_input("For example: 'I want something with ____', 'Delete this record', or 'Tell me about _______")
if st.button("Submit Prompt"):
    if user_prompt:
        st.spinner("Processing prompt...")
        response = requests.post(f"{API_URL}/agent", json={"prompt": user_prompt})
        if response.status_code == 200:
            result = response.json().get("response", "No response received")
            st.write("**Response:**", result)
        else:
            st.error("Error processing prompt.")
    else:
        st.error("Please enter a prompt.")

st.header("Current DataFrame")
response = requests.get(f"{API_URL}/data")
if response.status_code == 200:
    data = response.json().get("data", [])
    if data:
        df = pd.DataFrame(data)
        st.dataframe(df)
    else:
        st.write("No data available.")
else:
    st.error("Error fetching data.")
