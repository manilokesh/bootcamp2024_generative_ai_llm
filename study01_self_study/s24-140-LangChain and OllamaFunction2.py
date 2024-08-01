
# Import package from parent folder
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

########################################################################
# region Initialise foundation LLM

from utils.MyModels import init_llm, LlmModel,BaseChatModel
 
llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)

# endregion Initialise foundation LLM

########################################################################

from openai import OpenAI
import json


class ChatBot:

    def __init__(self, database):
        self.fake_db = database

    def makellm(self):
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # required, but unused 
        )
        return client

    def chat(self, query):

        client = self.makellm()

        initial_response = self.make_openai_request(client,query)

        message = initial_response.choices[0].message

        if hasattr(message, "function_call") & (message.function_call != None):
            function_name = message.function_call.name
            arguments = json.loads(message.function_call.arguments)
            function_response = getattr(self, function_name)(**arguments)

            follow_up_response = self.make_follow_up_request(
                client,query, message, function_name, function_response
            )
            return follow_up_response.choices[0].message.content
        else:
            return message.content

    def make_openai_request(self,client, query):
        
        response = client.chat.completions.create(
            model= LlmModel.MISTRAL.value,
            messages=[{"role": "user", "content": query}],
           # functions=functions,
           temperature=0
        )
        return response

    def make_follow_up_request(
        self, client,query, initial_message, function_name, function_response
    ): 
        response = client.chat.completions.create(
            model= LlmModel.MISTRAL.value,
            messages=[
                {"role": "user", "content": query},
                initial_message,
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                },
            ],
        )
        return response

    def place_order(self, item_name, quantity, address):
        if item_name not in self.fake_db["items"]:
            return f"We don't have {item_name}!"

        if quantity < 1:
            return "You must order at least one item."

        order_id = len(self.fake_db["orders"]) + 1
        order = {
            "order_id": order_id,
            "item_name": item_name,
            "quantity": quantity,
            "address": address,
            "total_price": self.fake_db["items"][item_name]["price"] * quantity,
        }

        self.fake_db["orders"].append(order)

        return f"Order placed successfully! Your order ID is {order_id}. Total price is ${order['total_price']}."

    def get_item_info(self, item_name):
        if item_name in self.fake_db["items"]:
            item = self.fake_db["items"][item_name]
            return f"Item: {item['name']}, Price: ${item['price']}"
        else:
            return f"We don't have information about {item_name}."


database = {
    "items": {
        "Chop-Suey": {"name": "Chop-Suey", "price": 15.0},
        "Lo-Mein": {"name": "Lo-Mein", "price": 12.0},
    },
    "orders": [],
}


bot = ChatBot(database=database)

response = bot.chat("I want to order two Chop-Suey to 321 Street")
print(response)

response = bot.chat("I want to order one spring roll to 321 Street")
print(response)

response= bot.chat("Who is the current Pope?")
print(response)