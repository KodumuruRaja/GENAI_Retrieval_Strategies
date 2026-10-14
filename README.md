# GENAI Retrieval Strategies

This repository contains a small set of Python examples demonstrating different retrieval strategies commonly used in GenAI and RAG pipelines. The scripts compare sparse retrieval, dense semantic retrieval, and hybrid retrieval using LangChain, Chroma, and OpenAI embeddings.

## Overview

Retrieval is a key component in Retrieval-Augmented Generation (RAG). In this project, we explore:

- Sparse retrieval using BM25
- Dense retrieval using embeddings and vector similarity search
- Hybrid retrieval combining sparse and dense signals for better relevance

The examples process PDF content, split it into chunks, index it in a vector database, and run a sample query to inspect the retrieved results.

## Project Structure

- [Dense_retriever.py](./Dense_retriever.py) — loads a PDF, chunks it, creates a Chroma vector store, and performs dense semantic retrieval.
- [Sparse_retriever.py](./Sparse_retriever.py) — loads a PDF, chunks it, and retrieves relevant chunks using BM25 sparse search.
- [Hybrid_retriever.py](./Hybrid_retriever.py) — combines BM25 and dense vector retrieval using an ensemble retriever.
- [VectorStore_retriever.py](./VectorStore_retriever.py) — performs direct similarity search against Chroma without using a LangChain retriever wrapper.

## How the Retrieval Works

### 1. Dense Retrieval
Dense retrieval uses embeddings to convert text chunks into vector representations. A query is also embedded, and the nearest matching chunks are found using vector similarity.

### 2. Sparse Retrieval
Sparse retrieval uses lexical matching. BM25 ranks documents based on term frequency and document frequency, which works well for exact keyword-based relevance.

### 3. Hybrid Retrieval
Hybrid retrieval combines both approaches to balance:

- keyword matching from sparse retrieval
- semantic relevance from dense retrieval

This often improves results when a query contains both exact terms and conceptual intent.

## Requirements

Before running the scripts, install the required Python packages:

```bash
pip install langchain langchain-community langchain-chroma langchain-openai python-dotenv pypdf
```

You will also need:

- Python 3.9+
- An OpenAI API key
- A PDF file to index

## Environment Setup

Create a `.env` file in the project root and add your OpenAI API key:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

The scripts call `load_dotenv()`, so the environment is automatically loaded.

## Important Note About the PDF Path

Each script currently points to a hardcoded local PDF path:

```python
r"C:\Users\Megha\Desktop\GENAI_CONCEPTS\Chunking_Strategies\India's.pdf"
```

Update this path to match your own PDF location before running the script.

## Usage

Run any script individually:

```bash
python Dense_retriever.py
python Sparse_retriever.py
python Hybrid_retriever.py
python VectorStore_retriever.py
```

Each script:

1. Loads the PDF
2. Splits the document into chunks
3. Builds the retriever/index
4. Executes a sample query
5. Prints the top matching chunks

## Example Query

The scripts use a sample query similar to:

```python
query = "What is the main topic of the document?"
```

You can replace this with your own query to test retrieval quality.

## Best Practices

- Adjust `chunk_size` and `chunk_overlap` for your use case.
- Use hybrid retrieval when you want a balance of keyword and semantic matching.
- Ensure the PDF path and environment variables are correctly configured.
- Persist the vector store if you want to reuse indexed documents across runs.

## License

This project is provided for educational and experimental purposes.

## Summary

This repository is a practical learning example for understanding how different retrieval techniques behave in a GenAI pipeline. It is useful for experimenting with RAG architecture, indexing strategies, and query quality.
