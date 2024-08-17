#!pip install "langserve[all]"

import os
import sys

sys.path.append(os.path.abspath(os.curdir))

import langchain

langchain.debug = True
langchain.verbose = True


from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

from utils.MyModels import LlmModel, init_llm

llm = init_llm(LlmModel.MISTRAL, temperature=0)

from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

chain = prompt_template | llm | parser

from fastapi import FastAPI

app = FastAPI(
    title="simpleTranslator",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

from langserve import add_routes

add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
