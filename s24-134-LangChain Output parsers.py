

########################################################################  
# region Initialise foundation LLM
 
from utils.MyModels import init_llm, LlmModel,BaseChatModel

llm:BaseChatModel = init_llm(LlmModel.MISTRAL)
 
# endregion Initialise foundation LLM

######################################################################## 
 
template_for_desired_output = """
Here is a book review: {book_review}.

I want you to output three things about the review in a JSON dictionary:

"sentiment": Is it positive or negative?
"positive": What is positively highlighted about the book in the review?
"negative": What is negatively highlighted about the book in the review?

"""

from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=["book_review"],
    template=template_for_desired_output
)

user_input = """
I bought the kindle edition of this book and I found it to be 
a terrible reading experience. Quote often it seemed like 
pieces of sentences were missing and occasionally when going 
from one page to another I would see the text of two lines 
superimposed on each other. I don't have a kindle so I used 
the android kindle app and kindle for the PC and both left 
much to be desired. Also I hated that if I flipped a few 
pages ahead that it would reset the location. Overall I would 
not recommend the e-book edition, the quality is horrendous. 
Also now that I see the prices, I paid more for the terrible 
kindle edition than the hardcover edition goes for. Anyway 
I would never recommend paying more for an e-book than a print 
book because you have less rights than with the print book. 
Also for a book like this I think there is value to being able 
to flip through it really quickly while the kindle interface 
is best for flipping through page by page.

That being said, the quality of the book was great. It was full 
of all sorts of insights and experiences. They can all be 
summarized as don't give up, watch out for VC's but don't 
write them off, listen to your customers, be willing to change, 
make sure the initial founding team works together well, etc... 
But just listening the values does not do it justice. You 
really have to read the experiences. The book is full of all 
sorts of insights too, not just about entrepreneurship but 
also about the individual companies. For example I was really 
impressed about PayPal and the fraud stuff they did and how 
valuable that was. I just never knew. Overall I think the book 
was very well put together. Although some of the founders liked 
to talk a lot more than others and it droned on and on. But 
others were brief and insightful. I would definitely recommend 
this.

If I bought the print edition I suspect I would be giving it 
5 stars. But really the kindle experience is probably worth 
0 stars. But the content is so good that I figure 4 stars is 
fair. Since at this time I see hardcover editions for $5 or 
$6 new I would say go grab one of those now!!! The book is 
definitely inspirational.
"""

question = prompt_template.format(book_review=user_input)

response = llm.invoke(question)

print(response)

print(type(response))

######################################################################## 

from langchain.output_parsers import PydanticOutputParser

from langchain_core.pydantic_v1 import BaseModel, Field, validator

from typing import List

class Reviews_Desired_Output_Structure(BaseModel):
    sentiment: List[str] = Field(
        description="Is it positive or negative?"
    )
    positive: List[str] = Field(
        description="What is positively highlighted about the book in the review?"
    )
    negative: List[str] = Field(
        description="What is negatively highlighted about the book in the review?"
    )

output_parser = PydanticOutputParser(
    pydantic_object=Reviews_Desired_Output_Structure
)

template_for_desired_output_with_parser = """
Here is a book review: {book_review}.

I want you to output three things about the review in a JSON dictionary:

{format_instructions}

"sentiment": Is it positive or negative?
"positive": What is positively highlighted about the book in the review?
"negative": What is negatively highlighted about the book in the review?

"""

prompt_template_with_parser = PromptTemplate(
    template=template_for_desired_output_with_parser,
    input_variables=["book_review"],
    partial_variables={
        "format_instructions": output_parser.get_format_instructions()
    }
)

question = prompt_template_with_parser.format(book_review=user_input)

response = llm.invoke(question)

formatted_output = output_parser.parse(response)

print(type(formatted_output))

json_output = formatted_output.json()

import json

python_dict = json.loads(json_output)

print(python_dict)