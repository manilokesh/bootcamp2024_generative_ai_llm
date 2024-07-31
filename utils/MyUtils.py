from enum import Enum

########################################################################
# region Logging

import coloredlogs, logging, sys

logger = logging.getLogger(__name__)
coloredlogs.install(
    level=logging.DEBUG,
    logger=logger,
    isatty=True,
    fmt="%(asctime)s %(levelname)-8s %(message)s",
    stream=sys.stdout,
    datefmt="%Y-%m-%d %H:%M:%S",
)
# endregion Logging

########################################################################
# region LOAD ENVIRONMENT VARIABLES

# > pip install pip install python-dotenv

from dotenv import load_dotenv, find_dotenv
import os

# Load the environment variables from the .env file
# find_dotenv() ensures the correct path to .env is used
dotenv_path = find_dotenv()
if dotenv_path == "":
    logger.error("No .env file found.")
else:
    logger.debug(f"Loading .env file from: {dotenv_path}")
    load_dotenv(dotenv_path)

# endregion LOAD ENVIRONMENT VARIABLES

embeddings_model_name = os.getenv("EMBEDDINGS_MODEL_NAME")
embeddings_model_path = os.getenv("EMBEDDINGS_MODEL_PATH")
chroma_db_path = os.getenv("CHROMA_DB_PATH")


########################################################################
# region add langchain logging

# > pip install langchain

import langchain

# langchain.debug = True
# langchain.verbose = True
# endregion add langchain logging

########################################################################
