import streamlit as st
import requests
import json
from config import Config

API_URL = "http://localhost:8000"  # Update this if your API is hosted elsewhere

st.title("Automated Comprehensive Reference Assistant")

# User Authentication (simplified for demonstration)
if 'user_id' not in st.session_state:
    st.session_state.user_id = 'default_user'  # In a real app, implement proper authentication

# Query Interface
user_query = st.text_input("Enter your query:")

if user_query:
    with st.spinner("Processing your query..."):
        response = requests.post(f"{API_URL}/query", json={"text": user_query, "user_id": st.session_state.user_id})
        if response.status_code == 200:
            result = response.json()
            st.subheader("Response:")
            st.write(result["response"])
            
            st.subheader("Relevant Information:")
            for chunk in result["relevant_chunks"]:
                st.write(f"From: {chunk['doc_id']} (Relevance: {chunk['score']:.2f})")
                st.write(chunk['text'])
                st.write("---")
        else:
            st.error("An error occurred while processing your query.")

# Admin Interface (simplified)
st.sidebar.title("Admin Interface")
admin_password = st.sidebar.text_input("Admin Password", type="password")

if admin_password == Config.ADMIN_PASSWORD:  # Implement proper authentication in a real app
    st.sidebar.subheader("Update Knowledge Base")
    doc_id = st.sidebar.text_input("Document ID:")
    content = st.sidebar.text_area("Document Content:")
    
    if st.sidebar.button("Update Knowledge Base"):
        update_response = requests.post(f"{API_URL}/update", json={"documents": {doc_id: content}})
        if update_response.status_code == 200:
            result = update_response.json()
            st.sidebar.success(f"Updated {result['documents_updated']} documents and {result['chunks_updated']} chunks.")
        else:
            st.sidebar.error("An error occurred while updating the knowledge base.")