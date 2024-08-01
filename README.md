# 🌟 bootcamp2024_generative_ai_llm

_Learn the keys to AI and build the most potential Generative AI applications._

## 🗂 Setting Up the Environment

### 📂 Navigate to Project Folder

```dos
c:\> cd D:\bootcamp2024_generative_ai_llm
```


### 🌐 Create a Virtual Environment

Create a virtual environment named ```venv``` to isolate your project dependencies.

```dos
# Using absolute path to Python executable

D:\bootcamp2024_generative_ai_llm> c:\Program Files\Python312\python.exe -m venv venv

# Or using the Python executable in your PATH

D:\bootcamp2024_generative_ai_llm> python -m venv venv
```

### 🔌 Activate the Virtual Environment

Activate the virtual environment to use its packages and dependencies.

```dos
D:\bootcamp2024_generative_ai_llm> .\venv\Scripts\activate.bat
```
 

### 📦 Install Required Modules

Install the necessary Python modules for the project.

```dos
pip install langchain
pip install langchain_community
pip install jupyter
pip install python-dotenv 
pip install langchain_google_genai
pip install beautifulsoup4
```

### 🚀 Start Jupyter Lab

Launch Jupyter Lab to start working on your notebooks.

```dos
PS D:\bootcamp2024_generative_ai_llm> jupyter lab
```

This will open a browser window with the URL: http://localhost:8888/

## 🛠 Project Information

### 📁 Project Structure

Organize your project as follows:

```markdown
project_root/
├── .env                # Environment variables file
├── data/               # Data directory
│   ├── subfolders/
│   └── AnyDataFile.txt
├── resources/          # Resources directory
│   ├── models/
│   └── chromadb/
├── utils/              # Utilities directory
│   ├── __init__.py
│   ├── MyxxxxModules.py
│   └── MyEmbeddingFunction.py
├── study0x/            # Source code directory
│   ├── Program01.py
│   └── Program02.py
├── level_0x/           # Source code directory
│   ├── Program01.py
│   └── Program02.py
└── your_script.py      # Main script
```

### 🌍 Creating .env (Environment File)

Create a file named .env in the root folder and add the following key-value pairs:

```
OPENAI_API_KEY=…
GEMINI_API_KEY=…
GROQ_API_KEY=…
CHROMA_DB_PATH=resources/chromadb
EMBEDDINGS_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDINGS_MODEL_PATH=resources/models/sentencetransformers
USER_AGENT=C:/Program Files/Google/Chrome/Application/chrome.exe
```

## 💻 Code Snippets

### 📝 Logging

Clear the terminal and set up logging.

```python
from utils.MyUtils import clear_terminal, logger

clear_terminal()
```

### 🤖 Foundation Model

Initialize the foundation model.

```python
from utils.MyModels import BaseChatModel, LlmModel, init_llm

llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)
```

### 📈 Embeddings

Create an instance of the embedding function.

```python
from utils.MyEmbeddingFunction import SentenceEmbeddingFunction

my_embeddings = SentenceEmbeddingFunction()
```

### 🗃️ Create Vector Store (ChromaDB) from Documents

Generate a vector store using ChromaDB from the documents.

```python
from utils.MyVectorStore import chroma_from_documents

vectorstore = chroma_from_documents(
    documents=all_splits, 
    embedding=my_embeddings, 
    collection_name="qa_retrieval_chain",
)
```

### 🗂️ Get Vector Store (ChromaDB)

Retrieve the vector store.

```python
from utils.MyVectorStore import chroma_get

vectorstore = chroma_get(
    embedding_function=my_embeddings,
    collection_name="qa_retrieval_chain",     
)
```





---

> _Enjoy Coding !_
> ▄︻デ══━一💥
> # ツ