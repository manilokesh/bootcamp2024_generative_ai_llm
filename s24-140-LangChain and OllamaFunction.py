from langchain.callbacks import StdOutCallbackHandler

########################################################################
# region Initialise foundation LLM
#llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)
# endregion Initialise foundation LLM
########################################################################
# https://python.langchain.com/v0.2/docs/integrations/chat/ollama_functions/
# https://medium.com/@mauryaanoop3/unleashing-structured-responses-functional-calling-with-langchain-ollama-and-phi-3-part-3-720b34203778
# https://github.com/BassAzayda/ollama-function-calling/blob/main/function_call.py
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_experimental.llms.ollama_functions import OllamaFunctions

from utils.MyUtils import logger

# Inference
# response = chain.invoke({"human_input": "How much does Chop-Suey cost?"}, config={"callbacks": [StdOutCallbackHandler()]})
# response = chain.invoke({"human_input": "I want to order two Chop-Suey to 321 Street"}, config={"callbacks": [StdOutCallbackHandler()]})

fake_db = {
    "items": {
        "Chop-Suey": {"price": 15.00, "ingredients": ["chop", "suey", "cheese"]},
        "Lo-Main": {"price": 10.00, "ingredients": ["lo", "main", "basil"]},
        "Chin-Gun": {"price": 12.50, "ingredients": ["chin", "gunu", "tomato sauce"]},
        "Won-Ton": {"price": 11.00, "ingredients": ["won", "ton", "mushrooms"]},
    },
    "orders": [],
}

##################################################################

def get_item_info(item_name):
    item = fake_db["items"].get(item_name)

    if not item:
        return f"No information available for item: {item_name}"

    return {
        "name": item_name,
        "price": item["price"],
        "ingredients": item["ingredients"],
    }


def place_order(item_name, quantity, address):
    if item_name not in fake_db["items"]:
        return f"We don't have {item_name}!"

    if quantity < 1:
        return "You must order at least one item."

    order_id = len(fake_db["orders"]) + 1
    order = {
        "order_id": order_id,
        "item_name": item_name,
        "quantity": quantity,
        "address": address,
        "total_price": fake_db["items"][item_name]["price"] * quantity,
    }

    fake_db["orders"].append(order)

    return f"Order placed successfully! Your order ID is {order_id}. Total price is ${order['total_price']}."

##################################################################

tools = [
    {
        "name": "get_item_info",
        "description": "Get name and price of a menu item of the chinese restaurant",
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {
                    "type": "string",
                    "description": "The name of the menu item, e.g. Chop-Suey",
                },
            },
            "required": ["item_name"],
        },
    },
    {
        "name": "place_order",
        "description": "Place an order for a menu item from the restaurant",
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {
                    "type": "string",
                    "description": "The name of the item you want to order, e.g. Chop-Suey",
                },
                "quantity": {
                    "type": "integer",
                    "description": "The number of items you want to order",
                    "minimum": 1,
                },
                "address": {
                    "type": "string",
                    "description": "The address where the food should be delivered",
                },
            },
            "required": ["item_name", "quantity", "address"],
        },
    },
]

##################################################################

from utils.MyModels import BaseChatModel, LlmModel, init_llm

# Loading Model
llm = OllamaFunctions(model=LlmModel.MISTRAL.value, keep_alive=-1, temperature=0)
llm = llm.bind_tools(tools=tools)

# Defining Prompt Template
prompt = PromptTemplate.from_template(
    """You are an AI chatbot having a conversation with a human.

Human: {human_input}
AI: """
)

chain = prompt | llm

##################################################################

# response = chain.invoke({"human_input": "I want to order two Chop-Suey to 321 Street"}, config={"callbacks": [StdOutCallbackHandler()]})

# print(response)


# print(response.function_call.name)
# print(json.loads(response.function_call.arguments))

# print(response.tool_calls[0]['name'])
# print(response.tool_calls[0]['args'])

# function_name = response.function_call.name
# arguments = json.loads(response.function_call.arguments)

# function_name = response.tool_calls[0]['name']
# arguments =  response.tool_calls[0]['args']

# response = place_order(**arguments)
# print(response)


def process_query(query):

    result = chain.invoke(
        {"human_input": query}, config={"callbacks": [StdOutCallbackHandler()]}
    )

    # print(response)

    if result.tool_calls:
        for tool_call in result.tool_calls:
            function_name = tool_call["name"]
            args = tool_call["args"]
            logger.info(f"Function call: {function_name}, Args: {args}")

            if function_name == "get_item_info":
                return get_item_info(**args)
            elif function_name == "place_order":
                return place_order(**args)

    return result.content


result = process_query("I want to order two Chop-Suey to 321 Street")
print(result)
 