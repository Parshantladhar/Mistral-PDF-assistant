import streamlit as st
import json
import os
import time
from src.helper import (
    get_document_text, 
    get_text_chunks, 
    get_vector_store, 
    get_conversational_chain,
    DEFAULT_CONFIG
)
from src.document_processor import analyze_text, extract_keywords

# App configuration
APP_TITLE = "Mistral Docs Assistant"
APP_DESCRIPTION = "Upload your documents and ask questions about their content"

# Initialize session state variables
def init_session_state():
    """Initialize session state variables."""
    session_vars = [
        "conversation", "chat_history", "document_text", 
        "processing_done", "uploaded_files", "document_count", 
        "config", "document_analysis", "error_log"
    ]
    
    for var in session_vars:
        if var not in st.session_state:
            if var == "config":
                st.session_state[var] = DEFAULT_CONFIG.copy()
            elif var in ["chat_history", "uploaded_files", "error_log"]:
                st.session_state[var] = []
            elif var == "document_count":
                st.session_state[var] = 0
            elif var == "processing_done":
                st.session_state[var] = False
            elif var == "document_analysis":
                st.session_state[var] = {}
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
            
            # Perform document analysis
            document_analysis = {}
            for filename, text in document_text.items():
                try:
                    text_analysis = analyze_text(text)
                    keywords = extract_keywords(text, top_n=7)
                    document_analysis[filename] = {
                        "text_analysis": text_analysis,
                        "keywords": keywords
                    }
                except Exception as e:
                    st.session_state.error_log.append(f"Error analyzing {filename}: {str(e)}")
            
            st.session_state.document_analysis = document_analysis
            
            # Combine all document texts
            all_text = " ".join(text for text in document_text.values())
            
            # Process text into chunks and create vector store
            text_chunks = get_text_chunks(all_text, st.session_state.config)
            vector_store = get_vector_store(text_chunks)
            
            # Initialize conversation chain
            st.session_state.conversation = get_conversational_chain(
                vector_store, st.session_state.config
            )
            
            st.session_state.processing_done = True
            st.success(f"✅ Successfully processed {len(files)} documents!")
            
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            st.session_state.error_log.append(f"Processing error: {str(e)}")
            st.session_state.processing_done = False

def handle_user_input(user_question):
    """Process user question and display response."""
    if not st.session_state.conversation:
        st.warning("⚠️ Please upload and process documents first.")
        return
    
    # Store user question in chat history if not already there
    if not st.session_state.chat_history or st.session_state.chat_history[-1].content != user_question:
        st.session_state.chat_history.append(type('obj', (object,), {'content': user_question}))
    
    # Get response
    with st.spinner("Thinking..."):
        try:
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            st.error(error_msg)
            st.session_state.error_log.append(error_msg)
            
            # Add fallback response in case of error
            st.session_state.chat_history.append(
                type('obj', (object,), {'content': "I'm sorry, I encountered an error processing your question. Please try rephrasing or ask another question."})
            )

def display_document_analysis():
    """Display document analysis in an expander."""
    with st.expander("📊 Document Analysis", expanded=False):
        if not st.session_state.document_analysis:
            st.info("No document analysis available. Please process documents first.")
            return
            
        for filename, analysis in st.session_state.document_analysis.items():
            st.subheader(f"📄 {filename}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Basic Statistics:**")
                text_analysis = analysis.get("text_analysis", {})
                st.write(f"- Word count: {text_analysis.get('word_count', 'N/A')}")
                st.write(f"- Sentence count: {text_analysis.get('sentence_count', 'N/A')}")
                st.write(f"- Avg. sentence length: {text_analysis.get('avg_sentence_length', 'N/A')} words")
                st.write(f"- Est. reading time: {text_analysis.get('reading_time_minutes', 'N/A')} min")
                
            with col2:
                st.write("**Key Topics:**")
                keywords = analysis.get("keywords", [])
                st.write(", ".join(keywords))
                
                # Show word frequency if available
                if "common_words" in text_analysis:
                    st.write("**Most Common Words:**")
                    common_words = text_analysis["common_words"][:5]  # Top 5 words
                    for word, count in common_words:
                        st.write(f"- {word}: {count}")
            
            st.markdown("---")

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
            st.slider(
                "Chunk Size", 
                min_value=500, 
                max_value=2000, 
                value=st.session_state.config["chunk_size"],
                step=100,
                help="Size of text chunks for processing",
                key="chunk_size_slider"
            )
            
            st.slider(
                "Chunk Overlap", 
                min_value=0, 
                max_value=100, 
                value=st.session_state.config["chunk_overlap"],
                step=5,
                help="Overlap between text chunks",
                key="chunk_overlap_slider"
            )
            
            # Get available models from model provider
            available_models = get_model_list()
            model_options = [model["name"] for model in available_models]
            default_index = model_options.index("mistral-medium") if "mistral-medium" in model_options else 0
            
            st.selectbox(
                "Model Name",
                options=model_options,
                index=default_index,
                help="Model to use for question answering",
                key="model_name_select"
            )
            
            st.slider(
                "Temperature", 
                min_value=0.0, 
                max_value=1.0, 
                value=st.session_state.config.get("temperature", 0.1),
                step=0.1,
                help="Controls randomness in responses (higher = more creative)",
                key="temperature_slider"
            )
            
            # Update config when any setting changes
            st.session_state.config = {
                "chunk_size": st.session_state.chunk_size_slider,
                "chunk_overlap": st.session_state.chunk_overlap_slider,
                "model_name": st.session_state.model_name_select,
                "temperature": st.session_state.temperature_slider,
            }
        
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
        tabs = st.tabs(["Chat", "Document Analysis", "Debug Info"])
        
        with tabs[0]:  # Chat tab
            # Chat input area
            st.markdown("### Ask questions about your documents")
            user_question = st.text_input("Your question:", key="user_input")
            
            if user_question:
                handle_user_input(user_question)
                
            # Display chat history
            if st.session_state.chat_history and len(st.session_state.chat_history) > 0:
                st.markdown("### Conversation History")
                display_chat_history()
        
        with tabs[1]:  # Document Analysis tab
            st.markdown("### Document Analysis")
            display_document_analysis()
            
            # Display document details
            with st.expander("Document Details"):
                for filename, text in st.session_state.document_text.items():
                    st.write(f"**{filename}**")
                    st.write(f"Text length: {len(text)} characters")
                    if len(text) > 500:
                        st.text_area(f"Preview of {filename}", text[:500] + "...", height=150)
                    else:
                        st.text_area(f"Content of {filename}", text, height=150)
                    st.write("---")
        
        with tabs[2]:  # Debug Info tab
            st.markdown("### Debug Information")
            st.json(st.session_state.config)
            
            if st.session_state.error_log:
                st.markdown("#### Error Log")
                for i, error in enumerate(st.session_state.error_log):
                    st.error(f"{i+1}. {error}")
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
