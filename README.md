# Conversational-RAG-with-PDF-Uploads-and-Chat-History
A Streamlit-based Conversational RAG application that lets users upload PDF documents and ask questions using Groq LLM, HuggingFace embeddings, ChromaDB vector storage, and chat history for context-aware responses.


Conversational RAG with PDF Uploads and Chat History

A conversational Retrieval-Augmented Generation (RAG) application built with Streamlit, LangChain, Groq, HuggingFace Embeddings, and ChromaDB.

Upload PDF documents, ask questions about their content, and receive context-aware answers while maintaining chat history.

Features

PDF Upload: Upload PDF documents for question answering.

Document Processing: Extract text and split documents into smaller chunks.

Semantic Search: Generate embeddings using the all-MiniLM-L6-v2 HuggingFace model.

Vector Database: Store and retrieve document embeddings using ChromaDB.

Conversational RAG: Use chat history to understand follow-up questions.

LLM Integration: Generate responses using Groq's chat model.

Interactive UI: Use a Streamlit interface to interact with your documents.

Tech Stack

Python

Streamlit

LangChain

Groq API

HuggingFace Sentence Transformers

ChromaDB

PyPDF

Project Structure

4.1-RAG Q&A Conversation/
├── app.py
├── requirements.txt
└── README.md

Installation and Setup

1. Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL
cd "4.1-RAG Q&A Conversation"

Replace YOUR_GITHUB_REPOSITORY_URL with your repository URL.

2. Create a virtual environment

python -m venv venv

Activate it on Windows PowerShell:

.\venv\Scripts\Activate.ps1

3. Install dependencies

python -m pip install -r requirements.txt

4. Configure the Groq API key

Get an API key from Groq Console.

Set the key in your terminal for the current PowerShell session:

$env:GROQ_API_KEY="your_groq_api_key"

Alternatively, create a .env file if your application is configured to load environment variables from it. Never commit API keys or other secrets to GitHub.

5. Run the application

python -m streamlit run app.py

Streamlit will provide a local URL to open the application in your browser.

How It Works

The user uploads a PDF document.

The application extracts the document text.

The extracted text is split into smaller chunks.

HuggingFace generates embeddings for the chunks.

ChromaDB stores the embeddings for similarity-based retrieval.

A history-aware retriever uses the conversation to interpret follow-up questions.

The retrieved document context is passed to the Groq language model.

The application displays the answer in the Streamlit interface.

Environment Variables

Variable

Description

GROQ_API_KEY

API key for accessing the Groq language model

Important Notes

The application requires a valid Groq API key.

The first run may take longer because the embedding model needs to be downloaded.

If using an in-memory ChromaDB client, the vector store is temporary and may need to be recreated after restarting the application.

Answers depend on the content and quality of the uploaded documents.

Future Improvements

Support multiple PDF documents.

Add persistent vector storage.

Add document citations and source references.

Deploy the application to Streamlit Community Cloud.
