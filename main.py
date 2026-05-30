from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

pdf_path='alice-in-wonderland.pdf'
loader = PyPDFLoader(pdf_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

# Check chunks
print("Number of chunks:", len(docs))

embedding_model = HuggingFaceEmbeddings()

db = FAISS.from_documents(
    docs,
    embedding_model
)

query=input("Enter yor query:")
results=db.similarity_search(query)

print(results[0].page_content)

