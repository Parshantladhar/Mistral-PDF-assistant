import os
import logging
from typing import List, Dict, Any, Tuple
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
from src.document_processor import DocumentProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Labyrinth - %(levelname)s - %(message)s'
)
logger: logging.Logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()
MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY")

# Check for key presence
if not MISTRAL_API_KEY:
    logger.error("Missing MISTRAL_API_KEY")
    raise EnvironmentError("Missing API key")
logger.info("Mistral API key loaded successfully")

# Default configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    "chunk_size": 1000,
    "chunk_overlap": 20,
    "model_name": "mistral-medium",
    "temperature": 0.5,
}

def process_single_doc(doc: Any) -> Tuple[str, str]:
    """Process a single document and return its name and extracted text."""
    processor = DocumentProcessor()
    try:
        text, _ = processor.process_document(doc, doc.name)
        logger.info(f"Successfully processed {doc.name}")
        return doc.name, text
    except Exception as e:
        logger.error(f"Error processing {doc.name}: {str(e)}")
        return doc.name, f"ERROR: {str(e)}"

def get_document_text(docs: List) -> Dict[str, str]:
    """Process multiple document types in parallel and return extracted text."""
    results: Dict[str, str] = {}
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Submit all document processing tasks
        future_to_doc = {executor.submit(process_single_doc, doc): doc for doc in docs}
        
        # Collect results
        for future in future_to_doc:
            doc_name, text = future.result()
            results[doc_name] = text
            
    return results

def get_text_chunks(text: str, config: Dict[str, Any] = None) -> List[str]:
    """Split text into manageable chunks."""
    if config is None:
        config = DEFAULT_CONFIG
        
    try:
        if not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.get("chunk_size", DEFAULT_CONFIG["chunk_size"]),
            chunk_overlap=config.get("chunk_overlap", DEFAULT_CONFIG["chunk_overlap"])
        )
        chunks: List[str] = text_splitter.split_text(text)
        logger.info(f"Text successfully split into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text into chunks: {str(e)}")
        raise Exception(f"Failed to process text chunks: {str(e)}")

@st.cache_resource
def get_vector_store(text_chunks: List[str]) -> FAISS:
    """Create a vector store in memory using Mistral embeddings."""
    if not text_chunks:
        logger.error("Empty text chunks provided for vector store creation")
        return None
    try:
        embeddings = MistralAIEmbeddings(api_key=MISTRAL_API_KEY)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        logger.info("Created vector store in memory")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        return None

def get_conversational_chain(vector_store: FAISS, config: Dict[str, Any] = None) -> ConversationalRetrievalChain:
    """Initialize the conversational retrieval chain using Mistral chat model."""
    if config is None:
        config = DEFAULT_CONFIG
        
    try:
        llm = ChatMistralAI(
            model=config.get("model_name", DEFAULT_CONFIG["model_name"]), 
            api_key=MISTRAL_API_KEY,
            temperature=config.get("temperature", DEFAULT_CONFIG["temperature"])
        )

        # Use updated memory interface
        chat_history: ChatMessageHistory = ChatMessageHistory()
        memory = ConversationBufferMemory(
            chat_memory=chat_history,
            memory_key="chat_history",
            return_messages=True
        )

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