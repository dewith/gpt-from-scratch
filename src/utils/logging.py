""" This module contains the function to get the logger for the script."""

import logging
import os
import sys
import datetime


def get_logger():
    """Get the logger for the script."""
    script_name = os.path.basename(sys.argv[0])
    file_name = script_name.replace(".py", "")
    timestamp = datetime.datetime.now().strftime("%y%m%d")
    log_path = f"logs/{timestamp}_{file_name}.log"

    file_handler = logging.FileHandler(filename=log_path, mode="a")
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    log_fmt = "[%(asctime)s %(levelname)8s] %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=logging.INFO, format=log_fmt, datefmt=date_fmt, handlers=handlers
    )
    logger = logging.getLogger(__name__)
    return logger
