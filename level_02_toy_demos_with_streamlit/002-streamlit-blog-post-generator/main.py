# Blog Post Generator
"""
● Problem: write blog posts with a certain word count about a topic adopting a
specific role.
● Applications:  Writing marketing, sales content, etc.

We set the response length with max_tokens
"""
# Import package from parent folder
import os
import sys

sys.path.append(os.path.abspath(os.curdir))
 


from utils.MyModels import BaseChatModel, LlmModel, init_llm

llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)

import streamlit as st
from langchain import PromptTemplate

st.set_page_config(page_title="Blog Post Generator")

st.title("Blog Post Generator")


def generate_response(topic):
    template = """
    As experienced startup and venture capital writer, 
    generate a 400-word blog post about {topic}
    
    Your response should be in this format:
    First, print the blog post.
    Then, sum the total number of words on it and print the result like this: This post has X words.
    """
    prompt = PromptTemplate(input_variables=["topic"], template=template)
    query = prompt.format(topic=topic)
    response = llm(query, max_tokens=2048)
    return st.write(response)


"""
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

topic_text = st.text_input("Enter topic: ")
if not openai_api_key.startswith("sk-"):
    st.warning("Enter OpenAI API Key")
if openai_api_key.startswith("sk-"):
    generate_response(topic_text)
 """

topic_text = st.text_input("Enter topic: ")
generate_response(topic_text)
