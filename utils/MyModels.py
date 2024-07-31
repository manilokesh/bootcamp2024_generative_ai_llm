from enum import Enum
import os

# region Initialise foundation LLM

# > pip install langchain
# > pip install langchain_google_genai
# > pip install langchain_openai


# Define the enum
class LlmModel(Enum):
    GEMINI = "gemini-1.5-pro"
    LLAMA = "llama3.1"
    MISTRAL = "mistral"
    OPENAI = "gpt-4o"
    GROQ_LLAMA3 = "llama3-70b-8192"
    GROQ_MIXTRAL = "mixtral-8x7b-32768"


# region google llm
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

# The `GOOGLE_API_KEY`` environment variable set with your API key, or
# Pass your API key using the google_api_key kwarg to the ChatGoogle constructor.


# Get the value of a specific environment variable
def init_llm_gemini(modelname: str, temperature: float):
    google_api_key = os.getenv("GEMINI_API_KEY")

    return ChatGoogleGenerativeAI(
        model=modelname, google_api_key=google_api_key, temperature=temperature
    )


# endregion google llm

# region ollama llm
from langchain_community.llms.ollama import Ollama


def init_llm_ollama(modelname: str, temperature: float):
    return Ollama(model=modelname, num_gpu=1, temperature=temperature)


# endregion ollama llm

# region openai llm
from langchain_openai import ChatOpenAI


def init_llm_openai(modelname: str, temperature: float):
    return ChatOpenAI(model=modelname, temperature=temperature)


# endregion openai llm

# region groq llm
from langchain_groq import ChatGroq


def init_llm_groq(modelname: str, temperature: float):
    return ChatGroq(model=modelname, temperature=temperature)


# endregion groq llm


def init_llm(llmmodel: LlmModel, temperature: float = 0.3) -> BaseChatModel:
    match llmmodel:
        case LlmModel.GEMINI:
            return init_llm_gemini(LlmModel.GEMINI.value, temperature)
        case LlmModel.LLAMA:
            return init_llm_ollama(LlmModel.LLAMA.value, temperature)
        case LlmModel.MISTRAL:
            return init_llm_ollama(LlmModel.MISTRAL.value, temperature)
        case LlmModel.OPENAI:
            return init_llm_openai(LlmModel.OPENAI.value, temperature)
        case LlmModel.GROQ_LLAMA3:
            return init_llm_groq(LlmModel.GROQ_LLAMA3.value, temperature)
        case LlmModel.GROQ_MIXTRAL:
            return init_llm_groq(LlmModel.GROQ_MIXTRAL.value, temperature)
        case _:
            raise ValueError(f"Unsupported LlmModel: {llmmodel}")
    return None


# endregion Initialise foundation LLM
