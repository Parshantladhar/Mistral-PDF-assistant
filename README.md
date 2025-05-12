# 📄 Information Retrieval from Multiple PDF with 🧠💬 Mistral & LangChain

## Description:
A powerful Retrieval-Augmented Generation (RAG) application that lets you chat with your documents. Upload PDFs, Word files, or text documents and ask questions in natural language to receive accurate, context-aware answers powered by Mistral AI and LangChain. Perfect for researchers, students, and professionals who need quick insights from their document collections.

## 🧭 How to run?

### 🔹 STEPS:

### 📥 Clone the repository

```bash
git clone https://github.com/Parshantladhar/MistralRAG-Smart_Document_Q_and_A_Engine.git
cd MistralRAG-Smart_Document_Q_and_A_Engine
````

---

### 🛠 STEP 01: Create a conda environment(Optional)

```bash
conda create -n mistralapp python=3.8 -y
conda activate mistralapp
```

---

### 📦 STEP 02: Install the requirements

```bash
pip install -r requirements.txt
```

---

### 🔐 STEP 03: Create a `.env` file in the root directory and add your API key

For example:

```env
MISTRAL_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

---

### ▶️ STEP 04: Run the app

```bash
streamlit run app.py
```

Now, open the browser and go to:

```
http://localhost:####
```

---

## 💡 Tech Stack Used

* 🐍 Python
* 🦜 LangChain
* 🌐 Streamlit
* 🧠 Mistral 
* 📊 FAISS (Vector Store)

---

> Made with ❤️ for building intelligent document assistants.

```
