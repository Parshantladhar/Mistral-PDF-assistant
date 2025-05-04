import streamlit as st
from src.helper import get_pdf_text, get_text_chunks, get_vector_store,get_conversational_chain


def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i%2 == 0:
            st.write("User: ", message.content)
        else:
            st.write("Reply: ", message.content)


def main():
    st.set_page_config(page_title="Mistral Docs Assistant")
    st.title("Mistral Docs Assistant")
    st.write("This is a simple Streamlit app to interact with Mistral and LangChain.")

    user_question = st.text_input("Enter your question From the PDFs Files:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    if user_question:
        user_input(user_question)

    # Add your app logic here
    with st.sidebar:
        st.title("Menu")
        st.write("Upload your file and hit Enter to process your files")
        pdf_file = st.file_uploader("",accept_multiple_files=True, type=["pdf", "txt","docx"] )

        if st.button("Submit & Process"):
            if pdf_file is not None:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_file)
                    test_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(test_chunks)
                    st.session_state.conversation =get_conversational_chain(vector_store)
                    # Process the uploaded files
                    for file in pdf_file:
                        # Here you would call your processing function
                        # For example: process_files(file)
                        st.success(f"{file.name} processed successfully!")
            else:
                st.error("Please upload a file.")

if __name__ == "__main__":
    main()