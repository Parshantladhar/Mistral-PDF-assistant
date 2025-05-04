import os
import logging
from typing import List, Union, Dict, Any
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import io
import docx2txt  

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Check for key presence
if not MISTRAL_API_KEY:
    raise EnvironmentError("Missing MISTRAL_API_KEY in .env file")

# Default configuration
DEFAULT_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 20,
    "model_name": "mistral-medium",
    "temperature": 0.5,
}

def extract_text_from_pdf(file_obj) -> str:
    """Extract text from a PDF file."""
    try:
        pdf_reader = PdfReader(file_obj)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise Exception(f"Failed to process PDF: {str(e)}")

def extract_text_from_txt(file_obj) -> str:
    """Extract text from a TXT file."""
    try:
        text = file_obj.read().decode('utf-8')
        return text
    except Exception as e:
        logger.error(f"Error extracting text from TXT: {str(e)}")
        raise Exception(f"Failed to process TXT file: {str(e)}")

def extract_text_from_docx(file_obj) -> str:
    """Extract text from a DOCX file."""
    try:
        text = docx2txt.process(file_obj)
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {str(e)}")
        raise Exception(f"Failed to process DOCX file: {str(e)}")

def get_document_text(docs: List) -> Dict[str, str]:
    """Process multiple document types and return extracted text with metadata."""
    results = {}
    
    for doc in docs:
        try:
            file_extension = os.path.splitext(doc.name)[1].lower()
            
            if file_extension == '.pdf':
                text = extract_text_from_pdf(doc)
            elif file_extension == '.txt':
                text = extract_text_from_txt(doc)
            elif file_extension == '.docx':
                text = extract_text_from_docx(doc)
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                continue
                
            results[doc.name] = text
            logger.info(f"Successfully processed {doc.name}")
            
        except Exception as e:
            logger.error(f"Error processing {doc.name}: {str(e)}")
            results[doc.name] = f"ERROR: {str(e)}"
            
    return results

def get_text_chunks(text: str, config: Dict = None) -> List[str]:
    """Split text into manageable chunks."""
    if config is None:
        config = DEFAULT_CONFIG
        
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.get("chunk_size", DEFAULT_CONFIG["chunk_size"]),
            chunk_overlap=config.get("chunk_overlap", DEFAULT_CONFIG["chunk_overlap"])
        )
        chunks = text_splitter.split_text(text)
        logger.info(f"Text successfully split into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text into chunks: {str(e)}")
        raise Exception(f"Failed to process text chunks: {str(e)}")

def get_vector_store(text_chunks: List[str]) -> FAISS:
    """Create a vector store using Mistral embeddings."""
    try:
        embeddings = MistralAIEmbeddings(api_key=MISTRAL_API_KEY)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        logger.info("Vector store created successfully")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise Exception(f"Failed to create vector store: {str(e)}")

def get_conversational_chain(vector_store: FAISS, config: Dict = None) -> ConversationalRetrievalChain:
    """Initialize the conversational retrieval chain using Mistral chat model."""
    if config is None:
        config = DEFAULT_CONFIG
        
    try:
        llm = ChatMistralAI(
            model=config.get("model_name", DEFAULT_CONFIG["model_name"]), 
            api_key=MISTRAL_API_KEY,
            temperature=config.get("temperature", DEFAULT_CONFIG["temperature"])
        )

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
        )
        logger.info("Conversation chain created successfully")
        return conversation_chain
    except Exception as e:
        logger.error(f"Error creating conversation chain: {str(e)}")
        raise Exception(f"Failed to initialize conversation chain: {str(e)}")