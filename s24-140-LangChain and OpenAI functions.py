
########################################################################
# region Initialise foundation LLM

from utils.MyModels import init_llm, LlmModel,BaseChatModel

llm:BaseChatModel = init_llm(LlmModel.MISTRAL , temperature=0)

# endregion Initialise foundation LLM

########################################################################

# https://python.langchain.com/v0.2/docs/integrations/chat/ollama_functions/
# https://medium.com/@mauryaanoop3/unleashing-structured-responses-functional-calling-with-langchain-ollama-and-phi-3-part-3-720b34203778

from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain.callbacks import StdOutCallbackHandler

# Loading Model
llm = OllamaFunctions(model=LlmModel.MISTRAL.value, keep_alive=-1)


# Schema for structured response
class Person(BaseModel):
    name: str = Field(description="The person's name", required=True)
    height: float = Field(description="The person's height", required=True)
    hair_color: str = Field(description="The person's hair color")


# Defining Prompt Template
prompt = PromptTemplate.from_template(
    """Alex is 5 feet tall. 
Claudia is 1 feet taller than Alex and jumps higher than him. 
Claudia is a brunette and Alex is blonde.

Human: {question}
AI: """
)

# Defining Chain
structured_llm = llm.with_structured_output(Person)
chain = prompt | structured_llm

# Inference
response = chain.invoke({"question": "Describe Alex"}, config={"callbacks": [StdOutCallbackHandler()]}) 

print(response)
