import configparser
import logging.config
import os


def get_package_root():
    return os.path.dirname(__file__)


def read_log_config():
    logging.config.fileConfig(os.path.join(get_package_root(), "logging.ini"))


def get_app_config() -> configparser.ConfigParser:
    app_conf = configparser.ConfigParser()
    app_conf.read(os.path.join(get_package_root(), "config.ini"))
    return app_conf
