from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
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

# 2. Initialize Embeddings and Vector Store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
persist_directory = "./chroma_db"

vector_store = Chroma.from_documents(
    documents=chunks1,
    embedding=embeddings,
    persist_directory=persist_directory
)
print("Chunks successfully added to ChromaDB!")

# 3. Expose the vector store as a Dense Retriever
# search_kwargs={"k": 2} configures it to return the top 2 matches
dense_retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 4. Query the Dense Retriever
query = "What is the main topic of the document?"
docs = dense_retriever.invoke(query)  # Standard LangChain Runnable method

print("\n--- Top Dense Search Results ---")
for i, doc in enumerate(docs):
    print(f"\nResult {i+1}:\n{doc.page_content}")
