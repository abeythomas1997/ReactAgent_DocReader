from typing import List,Union
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pathlib import Path
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader
)

class DocumentProcessor:

    """Handles document loading and processing"""

    def __init__(self,chunck_size:int=500, chunk_overlap:int=100):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        self.splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_from_url(self, url:str)->List[Document]:
        """Load document(s) from a URL"""
        loader=WebBaseLoader(url)
        return loader.load()

    def load_from_pdf_dir(self, directory:Union[str,Path])->List[Document]:
        """Load documents from all PDFs inside a directory"""
        loader=PyPDFDirectoryLoader(str(directory))
        return loader.load()

    def load_from_txt(self, file_path:Union[str,Path])->List[Document]:
        """Load documents from a text file"""
        loader=TextLoader(str(file_path))
        return loader.load()

    def load_from_pdf(self, file_path:Union[str,Path])->List[Document]:
        """Load documents from a PDF file"""
        loader=PyPDFLoader(str(file_path))
        return loader.load()

    def load_documents(self, sources:List[str])->List[Document]:    
        """Load documents from URLs, PDF directories, or text files"""
        docs:List[Document]=[]
        for source in sources:
            if source.startswith("http"):
                docs.extend(self.load_from_url(source))
            elif source.endswith(".pdf"):
                docs.extend(self.load_from_pdf(source))
            elif source.endswith(".txt"):
                docs.extend(self.load_from_txt(source))
            else:
                raise ValueError(f"Unsupported file type: {source}")
    return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of split documents
        """
        return self.splitter.split_documents(documents)
    
    def process_urls(self, urls: List[str]) -> List[Document]:
        """
        Complete pipeline to load and split documents
        
        Args:
            urls: List of URLs to process
            
        Returns:
            List of processed document chunks
        """
        docs = self.load_documents(urls)
        return self.split_documents(docs)