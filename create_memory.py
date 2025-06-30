from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load pdf
DATA_PATH="data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documemnts=load_pdf_files(data=DATA_PATH)
print(f"Loaded {len(documemnts)} pages from {DATA_PATH}")

# create chunks

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(extracted_data)
    return chunks

text_chunks = create_chunks(documemnts)
print(f"Created {len(text_chunks)} chunks from {DATA_PATH}")

# Create embeddings
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")   

embedding_model= get_embedding_model()

# store embedding
DB_FAISS_PATH = "vector/db_faiss"
db= FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)


# def create_embeddings(chunks):
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     embedded_chunks = embeddings.embed_documents([chunk.page_content for chunk in chunks])
#     return embedded_chunks