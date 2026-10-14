from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Replace Chroma and OpenAI embeddings with the BM25 Sparse Retriever
from langchain_community.retrievers import BM25Retriever 
from dotenv import load_dotenv

load_dotenv()

def pdf_loader(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()
    return data

def chunk_documents(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# 1. Load and chunk documents
documents = pdf_loader(r"C:\Users\Megha\Desktop\GENAI_CONCEPTS\Chunking_Strategies\India's.pdf")
print(f"Successfully loaded {len(documents)} pages.")

chunks1 = chunk_documents(documents, chunk_size=100, chunk_overlap=20)
print(f"Number of chunks created: {len(chunks1)}")

# 2. Initialize the Sparse Retriever (BM25)
# k=2 configures it to return the top 2 most relevant chunks
sparse_retriever = BM25Retriever.from_documents(chunks1, k=2)
print("Sparse retriever successfully initialized with chunks!")

# 3. Query the Sparse Retriever
query = "What is the main topic of the document?"
docs = sparse_retriever.invoke(query)  # invoke() replaces older retrieval methods

print("\n--- Top Sparse Search Results ---")
for i, doc in enumerate(docs):
    print(f"\nResult {i+1}:\n{doc.page_content}")
