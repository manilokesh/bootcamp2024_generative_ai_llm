

########################################################################  
# region Initialise foundation LLM
from utils.MyModels import init_llm, LlmModel

llm = init_llm(LlmModel.MISTRAL)
 
# endregion Initialise foundation LLM

######################################################################## 

# !pip install yt_dlp
# !pip install pydub
# !pip install openai-whisper

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import OpenAIWhisperParser
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader


url="https://www.youtube.com/watch?v=Rb9Bpw8yvTg"
save_dir="data/youtube/"
loader = GenericLoader(
    YoutubeAudioLoader([url],save_dir), 
    OpenAIWhisperParser()
)
docs = loader.load()

print(docs[0].page_content[0:500])