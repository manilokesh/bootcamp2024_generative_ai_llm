
########################################################################
# region Logging

import logging
import sys

import coloredlogs

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

from dotenv import find_dotenv, load_dotenv

# Load the environment variables from the .env file
# find_dotenv() ensures the correct path to .env is used
dotenv_path = find_dotenv()
if dotenv_path == "":
    logger.error("No .env file found.")
else:
    logger.debug(f"Loading .env file from: {dotenv_path}")
    load_dotenv(dotenv_path)

# endregion LOAD ENVIRONMENT VARIABLES

########################################################################
# region add langchain logging

# > pip install langchain

# import langchain

# langchain.debug = True
# langchain.verbose = True
# endregion add langchain logging

########################################################################

# import only system from os
from os import name, system


# define our clear function
def clear_terminal():

    # for windows
    if name == "nt":
        _ = system("cls")
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system("clear")


########################################################################
