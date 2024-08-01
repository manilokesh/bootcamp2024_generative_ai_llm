# bootcamp2024_generative_ai_llm
 learn the keys to AI and build the most potential Generative AI applications.



## Navigate to project folder ##

```bash
D:\Workplace\python\bootcamp2024_generative_ai_llm>
```

## Create virtual environment as "venv" (2nd venv, 1st is syntax ) ##
```bash
D:\Workplace\python\bootcamp2024_generative_ai_llm> c:\Program Files\Python312\python.exe -m venv venv
or
D:\Workplace\python\bootcamp2024_generative_ai_llm> python -m venv venv
```

## Activate newly created virtual environmanr *venv* ##
```bash
D:\Workplace\python\bootcamp2024_generative_ai_llm> .venv\Scripts\activate.bat
```

## Install modules ##
```bash
pip install langchain
pip install langchain_community
pip install jupyter
pip install python-dotenv 
pip install langchain_google_genai
pip install beautifulsoup4
```

## Start Jupyter Lab ##
```bash
PS D:\Workplace\python\bootcamp2024_generative_ai_llm> jupyter lab
```
opens a *"http://localhost:8888/"* in browser


# Project Info #
* ## Project Structure ##
    ```markdown
    project_root/
    ├── .env  (file)
    ├── data/
    │   ├── subfolders/
    │   └── AnyDataFile.txt
    ├── resources/
    │   ├── models/
    │   └── chromadb/
    ├── utils/
    │   ├── __init__.py
    │   ├── MyxxxxModules.py
    │   └── MyEmbeddingFunction.py
    └── your_script.py
    ```

    * ## Creating .env ( Environmental file) ##
    Create a file named **.env** under root folder and add below contents inside, like key=value pairs
    ```
    OPENAI_API_KEY=…
    GEMINI_API_KEY=…
    GROQ_API_KEY=…
    CHROMA_DB_PATH=resources/chromadb
    EMBEDDINGS_MODEL_NAME=all-MiniLM-L6-v2
    EMBEDDINGS_MODEL_PATH=resources/models/sentencetransformers
    USER_AGENT=C:/Program Files/Google/Chrome/Application/chrome.exe
    ```
    
# Code Snippets #
* ## Logging ##
    ```python
    from utils.MyUtils import clear_terminal, logger
    
    clear_terminal()
    ```

* ## Foundation Model ##
    ```python
    from utils.MyModels import BaseChatModel, LlmModel, init_llm
    
    llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)
    ```
 
* ## Embeddings ##
    ```python
    from utils.MyEmbeddingFunction import SentenceEmbeddingFunction
    
    my_embeddings = SentenceEmbeddingFunction()
    ```

 
* ## Create vector store ChromaDB from documents ##
    ```python
    from utils.MyVectorStore import chroma_from_documents
    
    vectorstore = chroma_from_documents(
        documents=all_splits, embedding=my_embeddings, collection_name="qa_retrieval_chain"
    )
    ```

* ## Get vector store, ChromaDB ##
    ```python
    from utils.MyVectorStore import chroma_get
    
    vectorstore = chroma_get(
        collection_name="qa_retrieval_chain", embedding_function=my_embeddings
    )
