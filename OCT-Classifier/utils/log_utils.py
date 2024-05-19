""" The Code is under Tencent Youtu Public Rule
"""
import logging
import os
import time
from datetime import datetime


def get_default_logger(
        args,
        logger_name,
        default_level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"):

    while not os.path.exists(args.out):
        time.sleep(0.1)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    logger = logging.getLogger(logger_name)
    logger.setLevel(default_level)
    console_handler.setFormatter(logging.Formatter(format, datefmt="%m/%d/%Y %H:%M:%S"))
    logger.addHandler(console_handler)

    filename = './result/' + 'train_info.log'

    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(default_level)
    file_handler.setFormatter(logging.Formatter(format, datefmt="%m/%d/%Y %H:%M:%S"))

    logger.addHandler(file_handler)

    return logger


def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'

    return datetime.today().strftime(fmt)
