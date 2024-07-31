
########################################################################
# region Initialise foundation LLM

from utils.MyModels import init_llm, LlmModel,BaseChatModel

llm:BaseChatModel = init_llm(LlmModel.MISTRAL)

# endregion Initialise foundation LLM

########################################################################
 
# https://python.langchain.com/v0.2/docs/how_to/callbacks_runtime/

from typing import Any, Dict, List
 
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain.prompts import PromptTemplate
from langchain.callbacks import StdOutCallbackHandler

from langchain.callbacks.base import BaseCallbackHandler

class LoggingHandler(BaseCallbackHandler):
    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs
    ) -> None:
        print("Chat model started")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        print(f"Chat model ended, response: {response}")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ) -> None:
        print(f"Chain {serialized.get('name')} started")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        print(f"Chain ended, outputs: {outputs}")
 
callbacks = [LoggingHandler(), StdOutCallbackHandler() ]
 
prompt =  PromptTemplate(
    input_variables=["input"], 
    template="Tell me a joke about {input}")

chain = prompt | llm

result = chain.invoke({"input": "whale"}, config={"callbacks": callbacks})

print(result)