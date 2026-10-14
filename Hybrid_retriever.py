from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain.retrievers import EnsembleRetriever 
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

# 2. Setup the Sparse Retriever (BM25)
# Fetches top 2 keyword-matched documents
sparse_retriever = BM25Retriever.from_documents(chunks1, k=2)

# 3. Setup the Dense Retriever (Chroma + OpenAI)
# Fetches top 2 semantically matched documents
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
persist_directory = "./chroma_db"

vector_store = Chroma.from_documents(
    documents=chunks1,
    embedding=embeddings,
    persist_directory=persist_directory
)
dense_retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 4. Setup the Hybrid (Ensemble) Retriever
# weights=[0.5, 0.5] splits the priority evenly between sparse and dense search
hybrid_retriever = EnsembleRetriever(
    retrievers=[sparse_retriever, dense_retriever],
    weights=[0.5, 0.5]
)
print("Hybrid retriever successfully initialized!")

# 5. Query the Hybrid Retriever
query = "What is the main topic of the document?"
docs = hybrid_retriever.invoke(query)

print("\n--- Top Hybrid Search Results ---")
for i, doc in enumerate(docs):
    print(f"\nResult {i+1}:\n{doc.page_content}")
