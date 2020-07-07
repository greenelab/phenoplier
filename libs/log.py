"""
Provides logging functions.
"""
import logging
import logging.config
import yaml

import conf


def _get_logger_config():
    with open(conf.GENERAL["LOG_CONFIG_FILE"], 'r') as f:
        return yaml.safe_load(f.read())


logging.config.dictConfig(_get_logger_config())


def get_logger(log_name: str = None) -> logging.Logger:
    """
    Returns a Logger instance.

    Args:
        log_name: logger name.

    Returns:
        A Logger instance configured with default settings. By default, it only print
        messages to the console.
    """
    return logging.getLogger(log_name)
