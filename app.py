import streamlit as st
import requests
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Load model and tokenizer manually
try:
    model_name = "deepset/tinyroberta-squad2"  # Try a different model if needed
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

def load_content_from_url(url):
    try:
        response = requests.get(url)
        return response.text
    except Exception as e:
        return f"Failed to load content: {e}"

def answer_question(content, question):
    inputs = tokenizer(question, content, return_tensors='pt')
    outputs = model(**inputs)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores) + 1
    tokens = inputs['input_ids'][0][start_index:end_index]
    answer = tokenizer.decode(tokens, skip_special_tokens=True)
    return answer

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
                combined_content += content + "\n\n"
        
        # Answer the question based on combined content
        try:
            answer = answer_question(combined_content, query_text)
            st.write(f"**Answer:** {answer}")
        except Exception as e:
            st.error(f"Failed to generate answer: {e}")
    else:
        st.error("Please enter URLs and a question.")

# Optional: Display loaded content
if st.checkbox('Show loaded content'):
    if uploaded_urls:
        combined_content = ""
        for url in uploaded_urls.splitlines():
            if url:  # Check if URL is not empty
                content = load_content_from_url(url)
                combined_content += content + "\n\n"
        st.write(combined_content)
    else:
        st.error("No URLs provided.")












