import os
from box.exceptions import BoxValueError
import yaml
from ThyroidProject import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a yaml file and returns 

    Args:
        path_to_yaml(str): Path like input

    Raises:
        ValueError: if yaml is empty
        e: empty yaml file

    Returns:
        ConfigBox: ConfigBox type   
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Creates list of directories

    Args:
        path_to_directories(list): List of path of directories
        verbose(bool): Prints info if directories are created
        ignore_log(bool, optional): ignore if multiple dirs is to be created. Defaults to False

    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at :{path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file

    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info("json file saved at:{path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json file data


    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dictionary
    """
    with open(path) as f:
        content = json.load(f)
    logger.info("json file loaded successfully from : {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary file
        path (Path): path to binary file

    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at : {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data from file

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in binary file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from : {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """
    Returns the size of a file in kb

    Args:
        path(Path): Path to file

    Returns:
        str: Size of file in kb
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~{size_in_kb} KB"
