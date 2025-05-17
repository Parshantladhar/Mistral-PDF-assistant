import streamlit as st
import logging
from src.helper import get_document_text, get_text_chunks, get_vector_store, get_conversational_chain, DEFAULT_CONFIG
from src.config import load_config, save_config
from src.document_processor import analyze_text, extract_keywords, DocumentProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# App configuration
APP_TITLE = "Mistral Docs Assistant"
APP_DESCRIPTION = "Upload your documents and ask questions about their content"

# Initialize session state variables
def init_session_state():
    """Initialize session state variables."""
    session_vars = [
        "conversation", "chat_history", "document_text", 
        "processing_done", "uploaded_files", "document_count", 
        "config", "document_processor"
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
            else:
                st.session_state[var] = None

def display_chat_history():
    """Display the chat history in a clean format."""
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:  # User message
                st.markdown(f"**You:** {message.content}")
            else:  # Assistant message
                st.markdown(f"**Assistant:** {message.content}")
        
    return chat_container

def display_document_stats(document_text):
    """Display statistics about processed documents"""
    if not document_text:
        return
    
    with st.expander("Document Details"):
        for filename, text in document_text.items():
            if text.startswith("ERROR:"):
                st.error(f"{filename}: {text}")
                continue
                
            st.write(f"**{filename}**")
            
            # Calculate stats
            stats = analyze_text(text)
            keywords = extract_keywords(text)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Word Count", stats["word_count"])
                st.metric("Reading Time", f"{stats['reading_time_minutes']} min")
            
            with col2:
                st.metric("Sentences", stats["sentence_count"])
                st.metric("Characters", stats["char_count"])
            
            st.write("**Top Keywords:** " + ", ".join(keywords))
            st.write("---")

def process_documents(files):
    """Process uploaded documents."""
    with st.spinner("Processing documents... This may take a minute."):
        try:
            # Store uploaded files in session state
            st.session_state.uploaded_files = files
            st.session_state.document_count = len(files)
            
            # Extract text from documents
            document_text = get_document_text(files)
            st.session_state.document_text = document_text
            
            # Combine all document texts
            all_text = " ".join(text for text in document_text.values() if not text.startswith("ERROR:"))
            
            # Process text into chunks and create vector store
            text_chunks = get_text_chunks(all_text, st.session_state.config)
            vector_store = get_vector_store(text_chunks)
            
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
            logger.error(f"Document processing error: {str(e)}")
            st.session_state.processing_done = False

def handle_user_input(user_question):
    """Process user question and display response."""
    if not st.session_state.conversation:
        st.warning("⚠️ Please upload and process documents first.")
        return
    
    # Get response
    with st.spinner("Thinking..."):
        try:
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']
            
            # Display or update chat history
            display_chat_history()
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            logger.error(f"Error in handle_user_input: {str(e)}")

def main():
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
        uploaded_files = st.file_uploader(
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
        
        # Add configuration expander
        with st.expander("⚙️ Advanced Configuration"):
            chunk_size = st.slider(
                "Chunk Size", 
                min_value=500, 
                max_value=2000, 
                value=st.session_state.config["chunk_size"],
                step=100,
                help="Size of text chunks for processing",
                key="chunk_size_slider"
            )
            
            chunk_overlap = st.slider(
                "Chunk Overlap", 
                min_value=0, 
                max_value=100, 
                value=st.session_state.config["chunk_overlap"],
                step=5,
                help="Overlap between text chunks",
                key="chunk_overlap_slider"
            )
            
            model_name = st.selectbox(
                "Model Name",
                options=["mistral-small", "mistral-medium"],
                index=["mistral-small", "mistral-medium"].index(st.session_state.config["model_name"]),
                help="Mistral model to use",
                key="model_name_select"
            )
            
            temperature = st.slider(
                "Temperature", 
                min_value=0.0, 
                max_value=1.0, 
                value=st.session_state.config["temperature"],
                step=0.1,
                help="Controls randomness in responses",
                key="temperature_slider"
            )
            
            top_k = st.slider(
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
                "top_k": top_k
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
            st.experimental_rerun()
            
        # Add information
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app uses [Mistral AI](https://mistral.ai/) and [LangChain](https://langchain.com/) 
        to provide intelligent document question answering.
        """)
    
    # Main area
    # Display chat interface only if documents are processed
    if st.session_state.processing_done:
        # Chat input area
        st.markdown("### Ask questions about your documents")
        user_question = st.text_input("Your question:", key="user_input")
        
        if user_question:
            handle_user_input(user_question)
            
        # Display chat history
        if st.session_state.chat_history and len(st.session_state.chat_history) > 0:
            st.markdown("### Conversation History")
            display_chat_history()
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
