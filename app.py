import streamlit as st
import logging
from typing import List, Dict, Any, Optional
from src.helper import get_document_text, get_text_chunks, get_vector_store, DEFAULT_CONFIG
from src.config import load_config, save_config, ModelName
from src.document_processor import analyze_text, extract_keywords, DocumentProcessor
import streamlit.components.v1 as components

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Labyrinth - %(levelname)s - %(message)s'
)
logger: logging.Logger = logging.getLogger(__name__)

# App configuration
APP_TITLE: str = "Mistral Docs Assistant"
APP_DESCRIPTION: str = "Upload your documents and ask questions about their content"

# Initialize session state variables
def init_session_state() -> None:
    """Initialize session state variables."""
    session_vars: List[str] = [
        "conversation", "chat_history", "document_text", 
        "processing_done", "uploaded_files", "document_count", 
        "config", "document_processor", "show_history"
    ]
    
    for var in session_vars:
        if var not in st.session_state:
            if var == "config":
                st.session_state[var] = load_config()
            elif var in ["chat_history", "uploaded_files"]:
                st.session_state[var] = []
            elif var == "document_count":
                st.session_state[var] = 0
            elif var == "processing_done":
                st.session_state[var] = False
            elif var == "document_processor":
                st.session_state[var] = DocumentProcessor()
            elif var == "show_history":
                st.session_state[var] = True
            else:
                st.session_state[var] = None

def display_chat_history() -> None:
    """Display the chat history in a chat bubble format."""
    if not st.session_state.show_history or not st.session_state.chat_history:
        return
    
    with st.container():
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:  # User message
                with st.chat_message("user"):
                    st.markdown(message.content)
            else:  # Assistant message
                with st.chat_message("assistant"):
                    st.markdown(message.content)

def display_document_stats(document_text: Dict[str, str]) -> None:
    """Display statistics about processed documents."""
    if not document_text:
        return
    
    with st.expander("Document Details"):
        for filename, text in document_text.items():
            if text.startswith("ERROR:"):
                st.error(f"{filename}: {text}")
                if "encrypted" in text.lower():
                    st.info("Tip: Ensure the PDF is not password-protected.")
                continue
                
            st.write(f"**{filename}**")
            
            # Calculate stats
            stats: Dict[str, Any] = analyze_text(text)
            keywords: List[str] = extract_keywords(text)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Word Count", stats["word_count"])
                st.metric("Reading Time", f"{stats['reading_time_minutes']} min")
            
            with col2:
                st.metric("Sentences", stats["sentence_count"])
                st.metric("Characters", stats["char_count"])
            
            st.write("**Top Keywords:** " + ", ".join(keywords))
            st.write("---")

def process_documents(files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> None:
    """Process uploaded documents."""
    max_size: int = 10 * 1024 * 1024  # 10MB
    for file in files:
        if file.size > max_size:
            st.error(f"File {file.name} exceeds 10MB limit.")
            return
    
    with st.spinner("Processing documents..."):
        progress_bar = st.progress(0)
        try:
            # Store uploaded files in session state
            st.session_state.uploaded_files = files
            st.session_state.document_count = len(files)
            
            # Extract text from documents in parallel
            document_text: Dict[str, str] = get_document_text(files)
            st.session_state.document_text = document_text
            logger.info(f"Extracted text from {len(document_text)} documents")
            
            # Update progress bar based on completed documents
            progress_bar.progress(0.5)
            
            # Combine all document texts
            all_text: str = " ".join(text for text in document_text.values() if not text.startswith("ERROR:") and text.strip())
            logger.info(f"Combined text length: {len(all_text)} characters")
            
            # Check if any valid text was extracted
            if not all_text.strip():
                st.error("No valid text extracted from the uploaded files. Please check if the files are readable and not empty or encrypted.")
                logger.error("No valid text extracted from uploaded files")
                st.session_state.processing_done = False
                return
            
            # Process text into chunks
            text_chunks: List[str] = get_text_chunks(all_text, st.session_state.config)
            logger.info(f"Generated {len(text_chunks)} text chunks")
            if not text_chunks:
                st.error("Failed to create text chunks. The extracted text may be too short or invalid.")
                logger.error("Text chunking resulted in empty list")
                st.session_state.processing_done = False
                return
            
            # Create vector store
            vector_store = get_vector_store(text_chunks)
            if vector_store is None:
                st.error("Failed to create vector store. This may be due to an invalid API key, network issues, or incompatible file content. Please check your API key and try different files.")
                logger.error("Vector store creation returned None")
                st.session_state.processing_done = False
                return
            
            # Update progress bar
            progress_bar.progress(1.0)
            
            # Initialize conversation chain
            st.session_state.conversation = get_conversational_chain(
                vector_store, st.session_state.config
            )
            
            st.session_state.processing_done = True
            st.success(f"✅ Successfully processed {len(files)} documents!")
            
            # Show document details
            display_document_stats(document_text)
            
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            if "encrypted" in str(e).lower():
                st.info("Tip: Ensure the PDF is not password-protected.")
            logger.error(f"Document processing error: {str(e)}")
            st.session_state.processing_done = False

def handle_user_input(user_question: str) -> None:
    """Process user question and display response."""
    if not st.session_state.conversation:
        st.warning("⚠️ Please upload and process documents first.")
        return
    
    # Get response
    with st.spinner("Thinking..."):
        try:
            response = st.session_state.conversation.invoke({'question': user_question})
            st.session_state.chat_history = response['chat_history']
            st.session_state.show_history = True
            
            # Immediately display the latest response
            with st.chat_message("user"):
                st.markdown(user_question)
            with st.chat_message("assistant"):
                st.markdown(response['answer'])
                
            # Scroll to the chat input field
            components.html(
                """
                <script>
                    const inputField = document.querySelector('input[data-testid="stTextInput"][aria-label="Your question:"]');
                    if (inputField) {
                        inputField.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        inputField.focus();
                    }
                </script>
                """,
                height=0
            )
                
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            logger.error(f"Error in handle_user_input: {str(e)}")

def main() -> None:
    """Main application function."""
    # Page configuration
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📄",
        layout="wide",
    )
    
    # Initialize session state
    init_session_state()
    
    # App header
    st.title("📄 Mistral Docs Assistant")
    st.markdown(APP_DESCRIPTION)
    
    # Sidebar
    with st.sidebar:
        st.title("📁 Document Upload")
        st.markdown("Upload your documents and configure the assistant")
        
        # File upload
        uploaded_files: Optional[List[st.runtime.uploaded_file_manager.UploadedFile]] = st.file_uploader(
            "Upload documents",
            accept_multiple_files=True, 
            type=["pdf", "txt", "docx"]
        )
        
        # Process button
        if st.button("Process Documents", type="primary"):
            if uploaded_files:
                process_documents(uploaded_files)
            else:
                st.error("Please upload at least one document.")
        
        # Display document count if documents are uploaded
        if st.session_state.document_count > 0:
            st.info(f"📄 {st.session_state.document_count} documents uploaded")
        
        # History button
        if st.button("📜 Toggle History"):
            st.session_state.show_history = not st.session_state.show_history
        
        # Add configuration expander
        with st.expander("⚙️ Advanced Configuration"):
            chunk_size: int = st.slider(
                "Chunk Size", 
                min_value=500, 
                max_value=2000, 
                value=st.session_state.config["chunk_size"],
                step=100,
                help="Size of text chunks for processing",
                key="chunk_size_slider"
            )
            
            chunk_overlap: int = st.slider(
                "Chunk Overlap", 
                min_value=0, 
                max_value=100, 
                value=st.session_state.config["chunk_overlap"],
                step=5,
                help="Overlap between text chunks",
                key="chunk_overlap_slider"
            )
            
            model_name: str = st.selectbox(
                "Model Name",
                options=[ModelName.SMALL.value, ModelName.MEDIUM.value],
                index=[ModelName.SMALL.value, ModelName.MEDIUM.value].index(st.session_state.config["model_name"]),
                help="Mistral model to use",
                key="model_name_select"
            )
            
            temperature: float = st.slider(
                "Temperature", 
                min_value=0.0, 
                max_value=1.0, 
                value=st.session_state.config["temperature"],
                step=0.1,
                help="Controls randomness in responses",
                key="temperature_slider"
            )
            
            top_k: int = st.slider(
                "Top K Results", 
                min_value=1, 
                max_value=20, 
                value=st.session_state.config.get("top_k", 5),
                step=1,
                help="Number of relevant chunks to retrieve",
                key="top_k_slider"
            )
            
            # Update config when any setting changes
            st.session_state.config = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "model_name": model_name,
                "temperature": temperature,
                "top_k": top_k,
                "max_tokens": st.session_state.config.get("max_tokens", 1024)
            }
            
            # Save configuration button
            if st.button("Save Configuration"):
                if save_config(st.session_state.config):
                    st.success("Settings saved successfully!")
                else:
                    st.error("Failed to save settings.")
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation"):
            st.session_state.chat_history = []
            st.session_state.show_history = True
            st.experimental_rerun()
            
        # Add information
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app uses [Mistral AI](https://mistral.ai/) and [LangChain](https://langchain.com/) 
        to provide intelligent document question answering.
        """)
    
    # Main area
    # Display chat history if available
    display_chat_history()
    
    # Display chat interface only if documents are processed
    if st.session_state.processing_done:
        # Chat input area with form to clear input
        st.markdown("### Ask questions about your documents")
        with st.form(key="chat_form", clear_on_submit=True):
            user_question: str = st.text_input("Your question:", key="chat_input_field")
            submit_button = st.form_submit_button("Send")
            
            if submit_button and user_question:
                handle_user_input(user_question)
            
    else:
        # Display instructions when no documents are processed
        st.info("👈 Please upload your documents using the sidebar and click 'Process Documents' to begin")
        
        # Show demo area
        st.markdown("## How it works")
        st.markdown("""
        1. **Upload** your documents (PDF, TXT, DOCX)
        2. **Process** them to extract information
        3. **Ask questions** about the content
        4. Get **intelligent answers** based on your documents
        
        This application uses:
        - 🧠 Mistral AI for understanding and generating responses
        - 📊 FAISS vector database for efficient retrieval
        - 🔗 LangChain for orchestrating the AI workflow
        """)


if __name__ == "__main__":
    main()