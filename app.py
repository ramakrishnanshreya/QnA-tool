import streamlit as st
import requests
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

# 1. Load the document from the URL

def load_content_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load content from {url}: {e}")
        return None

# 2. Split the document into smaller chunks

def split_into_chunks(text, max_chunk_length=400, stride=150):
    """Splits the text into chunks with overlap."""
    chunks = []
    for i in range(0, len(text), max_chunk_length - stride):
        chunk = text[i:i + max_chunk_length]
        chunks.append(chunk)
    return chunks

# 3. Answer the question (chunking and model inference)

def answer_question(content, question):
    """Answers the question using a distilled QA model, handling long inputs."""
    model_name = "distilbert-base-uncased-distilled-squad"  # A good distilled QA model
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return "I'm sorry, I couldn't load the question answering model."
        
    chunks = split_into_chunks(content)
    best_answer = None
    best_score = float('-inf')  # Initialize with negative infinity

    for chunk in chunks:
        try:
            inputs = tokenizer(question, chunk, return_tensors="pt", truncation=True, max_length=512)

            with torch.no_grad():
                outputs = model(**inputs)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                start_index = torch.argmax(start_logits)
                end_index = torch.argmax(end_logits) + 1
            
            tokens = inputs["input_ids"][0][start_index:end_index]
            answer = tokenizer.decode(tokens, skip_special_tokens=True)
            
            # Calculate a score (example: sum of logit values). Adjust as needed.
            score = torch.max(start_logits) + torch.max(end_logits)

            if answer and score > best_score: # Only update if a valid answer is found
                best_answer = answer
                best_score = score

        except Exception as e:
            st.warning(f"Failed to process chunk: {e}") # reduced scope of st.error
            continue # This is critical, keep going if there's an error
            
    if best_answer:
        return best_answer
    else:
        return "I'm sorry, I couldn't find an answer based on the available content."



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
            if url:
                content = load_content_from_url(url)
                if content: # Prevents adding None to the content
                    combined_content += content + "\n\n"

        # Answer the question
        if combined_content:
            answer = answer_question(combined_content, query_text)
            st.write(f"**Answer:** {answer}")
        else:
            st.error("No content loaded from the provided URLs.")
    else:
        st.error("Please enter URLs and a question.")

# Optional: Display loaded content
if st.checkbox('Show loaded content'):
    if uploaded_urls:
        combined_content = ""
        for url in uploaded_urls.splitlines():
            if url:
                content = load_content_from_url(url)
                if content:
                    combined_content += content + "\n\n"
        st.write(combined_content)
    else:
        st.error("No URLs provided.")














