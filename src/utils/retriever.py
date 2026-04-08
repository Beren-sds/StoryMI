"""
document retrieval module, provides functionality to retrieve relevant content from documents.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from PyPDF2 import PdfReader
from typing import List

class DocumentRetriever:
    """document retrieval class, responsible for loading and retrieving document content."""
    
    def __init__(self, file_path: str):

        self.file_path = file_path
        self.embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
        self.vector_store = None
        self.load_documents()
    
    def load_documents(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:

        # read PDF file
        pdf_reader = PdfReader(self.file_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        # split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        documents = text_splitter.split_text(text)
        
        # create vector store
        self.vector_store = SKLearnVectorStore.from_texts(
            texts=documents,
            embedding=self.embeddings
        )
    
    def retrieve_relevant_content(self, query: str, k: int = 1) -> List[str]:

        if not self.vector_store or not query:
            return []
        
        results = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in results] 