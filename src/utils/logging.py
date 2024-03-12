""" This module contains the function to get the logger for the script."""

import logging
import os
import sys


def get_logger():
    """Get the logger for the script."""
    file_name = os.path.basename(sys.argv[0]).replace(".py", "")
    file_handler = logging.FileHandler(filename=f"logs/{file_name}.log")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    format_ = "%(asctime)s L%(lineno)s %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=format_, handlers=handlers)
    logger = logging.getLogger(file_name)
    return logger
