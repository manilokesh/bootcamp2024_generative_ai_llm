# Advanced Output Parser
"""
To test an Output Parser that is capable of:
○ Using other types of inputs besides strings.
○ Validating the output format.

● Define the output structure we want.
● Use field_validators to validate the output format.
● Create the parser.
● Create the prompt template.
● Determine the user input.
● Apply the parser to obtain the desired output structure.
"""

from utils.MyUtils import clear_terminal, logger

clear_terminal()

# Define the desired output data structure

from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field, validator


class Suggestions_Output_Structure(BaseModel):
    words: List[str] = Field(
        description="list of substitute words based on the context"
    )
    reasons: List[str] = Field(
        description="the reasoning of why this word fits the context"
    )

    # Throw error if the substitute word starts with a number
    @validator("words")
    def not_start_with_number(cls, info):
        for item in info:
            if item[0].isnumeric():
                raise ValueError("ERROR: The word cannot start with a number")
        return info

    @validator("reasons")
    def end_with_dot(cls, info):
        for idx, item in enumerate(info):
            if item[-1] != ".":
                info[idx] += "."
        return info


# Create the parser
from langchain.output_parsers import PydanticOutputParser

my_parser = PydanticOutputParser(pydantic_object=Suggestions_Output_Structure)

# Determine the input

from langchain.prompts import PromptTemplate

my_template = """
Offer a list of suggestions to substitute the specified
target_word based on the present context and the reasoning
for each word.

{format_instructions}

target_word={target_word}
context={context}
"""

my_prompt = PromptTemplate(
    template=my_template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": my_parser.get_format_instructions()},
)

user_input = my_prompt.format_prompt(
    target_word="loyalty",
    context="""
    The loyalty of the soldier was so great that
    even under severe torture, he refused to betray
    his comrades.
    """,
)

# Initialise foundation LLM

from utils.MyModels import BaseChatModel, LlmModel, init_llm

llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)

"""
# DeprecationWarning: The method `BaseLLM.__call__` was deprecated in langchain-core 0.1.7 and will be removed in 0.3.0. 
# Use invoke instead.
output = llm(user_input.to_string())
"""
output = llm.invoke(user_input.to_string())

print(output)

# Apply the parser to get the desired output structure

output_parsed = my_parser.parse(output)

print(output_parsed)
