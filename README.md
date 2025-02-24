# Web Content Q&A Tool (UI-Focused)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FCeRRx80NXlHZNVbYzfrseJm_zwVXe9U?usp=sharing)

## Overview
The **Web Content Q&A Tool** is a web-based application that allows users to ingest content from one or more URLs and ask questions based strictly on that ingested content. It leverages a retrieval-based approach using LangChain, Chroma, and a locally hosted Flan-T5 model to generate concise and accurate answers. The tool features a single-tab Gradio interface for a streamlined user experience.

## Features
- **Content Ingestion:** Scrapes and cleans text from provided URLs.
- **Retrieval-Based QA:** Splits text into manageable chunks, embeds them, and stores them in a local vector database for fast retrieval.
- **Single-Tab Interface:** Combine URL ingestion and question answering in one simple interface.
- **Data Reset:** Easily clear ingested data to start a new session.
- **Run on Colab:** Launch the notebook on Google Colab with a single click.

## Requirements
- Python 3.7+
- [Gradio](https://gradio.app/)
- [Requests](https://docs.python-requests.org/)
- [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [LangChain](https://github.com/hwchase17/langchain)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [SentenceTransformers](https://www.sbert.net/)
- [Transformers](https://huggingface.co/transformers/)
- [Accelerate](https://github.com/huggingface/accelerate)
- [faiss-cpu](https://github.com/facebookresearch/faiss)
- [Torch](https://pytorch.org/)

## Setup and Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ramakrishnanshreya/QnA-tool.git

2. **Install Dependencies** \n
   Use the command pip install -r requirements.txt \n
   Alternatively, install the packages manually: \n
   ```bash
   pip install gradio requests beautifulsoup4 langchain chromadb sentence-transformers transformers accelerate faiss-cpu torch

## Usage
1. **Run the Application:**
   ```bash
   python app.py

This will launch a Gradio web interface in your browser.

Ingest Content & Ask Questions:
Enter one or more comma-separated URLs to ingest their content.
Type your question and click Submit to get an answer based strictly on the ingested content.
Use the Clear Data button to reset ingested content.

## Deployment
You can run this project locally or on platforms like Google Colab. Click the Run in Colab button above to open and run the notebook on Google Colab.

## Contact
For questions or inquiries, please contact ramakrishnanshreya@gmail.com.


