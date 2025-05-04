from setuptools import setup, find_packages

setup(
    name="mistral-docs-assistant",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "transformers",
        "faiss-cpu",
        "mistralai",
        "streamlit",
        "python-dotenv",
        "PyPDF2",
        
    ],
    author="Parshant Kumar",
    description="An information retrieval assistant using Mistral and LangChain.",
)
