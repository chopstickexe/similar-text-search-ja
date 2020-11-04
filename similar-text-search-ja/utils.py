import configparser
from logging import Formatter, StreamHandler, getLogger
from pathlib import Path


def get_package_root(file: str = __file__) -> Path:
    """Return path to the parent directory of the given file

    Args:
        file (str, optional): Python script or other types of file. Defaults to __file__.

    Returns:
        Path: Parent path
    """
    return Path(file).parent


def set_logger(name: str, level: str = "INFO"):
    """Set a logger with the given name appropriately

    Args:
        name (str): logger name
        level (str, optional): logging level. Defaults to "INFO".
    """
    logger = getLogger(name)
    formatter = Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handler = StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False


def get_app_config(
    path: Path = get_package_root() / "config.ini",
) -> configparser.ConfigParser:
    """Parse config.ini

    Args:
        path (Path, optional): config.ini path. Defaults to get_package_root()/"config.ini".

    Returns:
        configparser.ConfigParser: Key-value pairs defined in the config
    """
    app_conf = configparser.ConfigParser()
    app_conf.read(path)
    return app_conf
