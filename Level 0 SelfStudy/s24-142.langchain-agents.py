
# Import package from parent folder
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.MyUtils import clear_terminal, logger

clear_terminal()

# Initialise foundation LLM

from utils.MyModels import BaseChatModel, LlmModel, init_llm

llm: BaseChatModel = init_llm(LlmModel.MISTRAL, temperature=0)


from langchain import hub

prompt = hub.pull("hwchase17/react")

from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.chains import LLMMathChain
from langchain.tools import BaseTool

problem_chain = LLMMathChain.from_llm(llm=llm)
math_tool = Tool.from_function(
    name="Calculator",
    func=problem_chain.run,
    description="""Useful for when you need to answer questions about math. This tool is only for math questions and nothing else. Only input math expressions.""",
)

from playwright.sync_api import sync_playwright


class SearchWebTool(BaseTool):
    name = "search_from_web"
    description = "Searchs on the web for the specified query"

    def _run(self, query: str):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(f"https://www.google.com/search?q={query}")
            results = page.query_selector_all("h3")
            titles = [result.inner_text() for result in results]
            browser.close()
            return f"Top results for '{query}':\n" + "\n".join(titles[:5])

    def _arun(self, ticker: str):
        raise NotImplementedError("This tool does not support async")

import requests


class JokeGeneratorTool(BaseTool):
    name = "joke_generator"
    description = " Fetches a random joke from an API"

    def _run(self, query: str):
        joke_url = "https://official-joke-api.appspot.com/random_joke"

        try:
            response = requests.get(joke_url)
            response.raise_for_status()
            joke_data = response.json()
            joke = f"{joke_data['setup']} - {joke_data['punchline']}"
            logger.info(f"Random joke: {joke}")
            return joke
        except requests.exceptions.RequestException as e:
            logger.error(f"Error occurred while fetching joke: {str(e)}")
            return f"An error occurred while fetching a joke: {str(e)}"

    def _arun(self, ticker: str):
        raise NotImplementedError("This tool does not support async")


class WeatherFinderTool(BaseTool):
    name = "get_weather"
    description = "Fetches the current weather for a given location."

    def _run(self, location: str, unit="celsius"):
        logger.info(f"Getting weather for {location}")
        base_url = "https://api.open-meteo.com/v1/forecast"

        # Set up parameters for the weather API
        params = {
            "latitude": 0,
            "longitude": 0,
            "current_weather": "true",
            "temperature_unit": unit,
        }

        # Set up geocoding to convert location name to coordinates
        geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
        location_parts = location.split(",")
        city = location_parts[0].strip()
        country = location_parts[1].strip() if len(location_parts) > 1 else ""

        geo_params = {"name": city, "count": 1, "language": "en", "format": "json"}

        try:
            # First attempt to get coordinates
            logger.info(f"Fetching coordinates for {location}")
            geo_response = requests.get(geocoding_url, params=geo_params)
            geo_response.raise_for_status()
            geo_data = geo_response.json()
            logger.debug(f"Geocoding response: {geo_data}")

            # If first attempt fails, try with full location string
            if "results" not in geo_data or not geo_data["results"]:
                geo_params["name"] = location
                geo_response = requests.get(geocoding_url, params=geo_params)
                geo_response.raise_for_status()
                geo_data = geo_response.json()
                logger.debug(f"Second geocoding attempt response: {geo_data}")

            # Extract coordinates if found
            if "results" in geo_data and geo_data["results"]:
                params["latitude"] = geo_data["results"][0]["latitude"]
                params["longitude"] = geo_data["results"][0]["longitude"]
                logger.info(
                    f"Coordinates found: {params['latitude']}, {params['longitude']}"
                )
            else:
                logger.warning(f"No results found for location: {location}")
                return f"Sorry, I couldn't find the location: {location}"

            # Fetch weather data using coordinates
            logger.info("Fetching weather data")
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            weather_data = response.json()
            logger.debug(f"Weather data response: {weather_data}")

            # Extract and format weather information
            if "current_weather" in weather_data:
                current_weather = weather_data["current_weather"]
                temp = current_weather["temperature"]
                wind_speed = current_weather["windspeed"]

                result = f"The current weather in {location} is {temp}°{unit.upper()} with a wind speed of {wind_speed} km/h."
                logger.info(f"Weather result: {result}")
                return result
            else:
                logger.warning(f"No current weather data found for {location}")
                return f"Sorry, I couldn't retrieve weather data for {location}"
        except requests.exceptions.RequestException as e:
            logger.error(f"Error occurred while fetching weather data: {str(e)}")
            return f"An error occurred while fetching weather data: {str(e)}"

    def _arun(self, ticker: str):
        raise NotImplementedError("This tool does not support async")


tools = [math_tool, SearchWebTool(), JokeGeneratorTool(), WeatherFinderTool()]

agent = create_react_agent(tools=tools, llm=llm, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

from typing import Any, Dict, List

from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


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


from langchain.callbacks import StdOutCallbackHandler

# question = "What is the 22k gold rate in Erode today? according to goodreturns.in"
# question = "Compare Average Gold Rate in Erode for 22K & 24K"
# question = "say me a joke"
question = (
    "What is the capital city of TamilNadu and what is the weather forcast today ?"
)

response = agent_executor.invoke(
    {"input": question},
    config={"callbacks": [LoggingHandler(), StdOutCallbackHandler()]},
)

print(response)
