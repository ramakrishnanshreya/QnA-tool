import streamlit as st
import requests
from bs4 import BeautifulSoup

# Function to scrape the content from a URL
def scrape_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')  # Extract text from all <p> tags
        text = ' '.join([para.get_text() for para in paragraphs])  # Join all text into one string
        return text
    except Exception as e:
        return f"Error scraping the URL: {e}"

# Function to answer the user's question based on the scraped content
def generate_answer(text, question):
    if question.lower() in text.lower():  # Simple text search for now
        return f"Found relevant content: {text[:300]}..."  # Return a short preview
    else:
        return "Sorry, I couldn't find an answer to your question."

# Streamlit UI
st.title("Web Content Q&A Tool")

# Input field for URLs (User can enter multiple URLs)
url_input = st.text_area("Enter one or more URLs (separate by commas):")

# Split the input URLs into a list
if url_input:
    urls = [url.strip() for url in url_input.split(",")]

    # Allow the user to input a question
    question_input = st.text_area("Ask a question based on the webpage content:")

    if question_input:
        # Scrape content from each URL
        scraped_texts = []
        for url in urls:
            st.write(f"Scraping content from: {url}...")
            scraped_text = scrape_url(url)
            scraped_texts.append(scraped_text)
            
            # Display part of the scraped content
            st.write(f"Scraped Content from {url} (First 500 characters):")
            st.text(scraped_text[:500])  # Show a preview of the scraped content

        # Combine all scraped content
        combined_text = " ".join(scraped_texts)

        # Answer the question based on the combined scraped content
        st.write("Answer to your question:")
        answer = generate_answer(combined_text, question_input)
        st.write(answer)

