import json
from logging import Formatter, StreamHandler, getLogger
from pathlib import Path
from typing import Any, Dict


def get_dir(file: str = __file__) -> Path:
    """Return path to the directory of the given file

    Args:
        file (str, optional): Python script or other types of file. Defaults to __file__.

    Returns:
        Path: directory path
    """
    return Path(file).parent


def set_root_logger(level: str = "INFO"):
    """Set formatters and handlers to the root logger

    Args:
        level (str, optional): logging level. Defaults to "INFO".
    """
    logger = getLogger()
    formatter = Formatter("%(asctime)s %(name)-14s %(levelname)-8s %(message)s")
    handler = StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(handler)


def read_json_config(config_file: Path) -> Dict[str, Any]:
    with open(str(config_file), "r") as f:
        return json.load(f)
