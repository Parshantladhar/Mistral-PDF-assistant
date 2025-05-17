from setuptools import setup, find_packages

setup(
    name="mistral-docs-assistant",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-community",
        "langchain-mistralai"
        "faiss-cpu",
        "mistralai",
        "streamlit",
        "python-dotenv",
        "PyPDF2",
        "doc2txt",
        
    ],
    author="Parshant Kumar",
    description="An information retrieval assistant using Mistral and LangChain.",
)
