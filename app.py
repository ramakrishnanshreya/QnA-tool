import streamlit as st
import requests
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

# Load model and tokenizer manually
try:
    model_name = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

def load_content_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load content from {url}: {e}")
        return None

def answer_question(content, question):
    try:
        inputs = tokenizer(question, content, return_tensors='pt', truncation=True, max_length=512)
        
        with torch.no_grad():  # Disable gradient calculation
            outputs = model(**inputs)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits

            start_index = torch.argmax(start_scores)
            end_index = torch.argmax(end_scores) + 1
        
        tokens = inputs['input_ids'][0][start_index:end_index]
        answer = tokenizer.decode(tokens, skip_special_tokens=True)
        return answer
    except Exception as e:
        st.error(f"Failed to generate answer: {e}")
        return "I'm sorry, I couldn't find an answer to your question based on the provided content."

# Streamlit App Template
st.title('Web Content Q&A Tool')
st.header('Enter URLs and ask questions')

# Input field for multiple URLs
uploaded_urls = st.text_area('Enter one or more URLs (separated by newline)', placeholder='http://example.com\nhttp://example2.com')

# Input field for question
query_text = st.text_input('Enter your question:', placeholder='Ask something about the loaded content.')

# Button to trigger the question answering
if st.button('Ask'):
    if uploaded_urls and query_text:
        # Load content from each URL
        combined_content = ""
        for url in uploaded_urls.splitlines():
            if url:  # Check if URL is not empty
                content = load_content_from_url(url)
                if content:  # Only add content if loading was successful
                    combined_content += content + "\n\n"
        
        # Answer the question based on combined content
        if combined_content:
            try:
                answer = answer_question(combined_content, query_text)
                st.write(f"**Answer:** {answer}")
            except Exception as e:
                st.error(f"Failed to generate answer: {e}")
        else:
            st.error("No content loaded from the provided URLs.")
    else:
        st.error("Please enter URLs and a question.")

# Optional: Display loaded content
if st.checkbox('Show loaded content'):
    if uploaded_urls:
        combined_content = ""
        for url in uploaded_urls.splitlines():
            if url:  # Check if URL is not empty
                content = load_content_from_url(url)
                if content:
                    combined_content += content + "\n\n"
        st.write(combined_content)
    else:
        st.error("No URLs provided.")













