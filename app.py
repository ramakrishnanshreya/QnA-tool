import streamlit as st
import requests
from bs4 import BeautifulSoup
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama  # Change this if using GPT4All

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize FAISS index
d = 384  # Embedding size of MiniLM
index = faiss.IndexFlatL2(d)
documents = []

# Load LLM model (Change path if using GPT4All)
llm = Llama(model_path="path_to_llama_model.gguf")

# Streamlit UI
st.title("Web Content Q&A Tool")

# URL Input
url = st.text_input("Enter URL:")
if st.button("Scrape and Ingest"):
    if url:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            text = " ".join([p.get_text() for p in soup.find_all("p")])
            documents.append(text)
            embeddings = model.encode([text])
            index.add(np.array(embeddings, dtype=np.float32))
            st.success("Content Ingested Successfully!")
        except Exception as e:
            st.error(f"Failed to scrape: {e}")

# Question Input
question = st.text_input("Ask a question:")
if st.button("Get Answer"):
    if question and len(documents) > 0:
        question_embedding = model.encode([question])
        D, I = index.search(np.array(question_embedding, dtype=np.float32), k=1)
        retrieved_text = documents[I[0][0]] if I[0][0] < len(documents) else ""
        prompt = f"Answer this based ONLY on the text: {retrieved_text}\nQuestion: {question}\nAnswer:"
        response = llm(prompt)["choices"][0]["text"].strip()
        st.write("### Answer:", response)
    else:
        st.warning("Please ingest content first!")















