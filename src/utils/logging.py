"""
logging.py

Setup project-wide logging.
"""
import logging

def get_logger(name):
    """Return a configured logger."""
    logger = logging.getLogger(name)
    return logger
