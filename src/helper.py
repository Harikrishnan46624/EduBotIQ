from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings



#Extract data
def load_documents(data):
    csv_loader = DirectoryLoader(data, glob='*.csv', loader_cls=CSVLoader)
    pdf_loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)

    csv_documents = csv_loader.load()
    pdf_documents = pdf_loader.load()

    combined_data = csv_documents + pdf_documents

    return combined_data



#Create text chunk
def text_split(extracted_data):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap = 20)
  text_chunks = text_splitter.split_documents(extracted_data)
  return text_chunks



#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings