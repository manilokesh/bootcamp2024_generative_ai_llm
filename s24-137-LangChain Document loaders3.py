

########################################################################
# region Initialise foundation LLM

from utils.MyModels import init_llm, LlmModel,BaseChatModel

llm:BaseChatModel = init_llm(LlmModel.MISTRAL)

# endregion Initialise foundation LLM

########################################################################
 
#  Web Base Loader
"""
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://aiaccelera.com/100-ai-startups-100-llm-apps-that-have-earned-500000-before-their-first-year-of-existence/")
docs = loader.load()
print(docs[0].page_content[:2000])
"""
########################################################################
 
# Unstructured HTML Loader
"""
from langchain.document_loaders import UnstructuredHTMLLoader

loader = UnstructuredHTMLLoader("data/_100 AI Startups__ 100 LLM Apps that have earned $500,000 before their first year of existence.html")
data = loader.load()
print(data)
"""
########################################################################
 
# Beautiful Soup

from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://analyticsindiamag.com/ai-news-updates/mistral-ai-unveils-mistral-large-2-beats-llama-3-1-on-code-and-math/")
data = loader.load()
print(data)
print(type(data))


from langchain.text_splitter import RecursiveCharacterTextSplitter

chunk_size =26
chunk_overlap = 4

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "(?<=\\. )", " ", ""]
)

result1 = recursive_splitter.split_text(data[0].page_content)

print(result1)
