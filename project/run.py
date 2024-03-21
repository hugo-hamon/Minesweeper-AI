from src.app import App
import argparse
import logging
import sys
import os

LOGGING_CONFIG = {
    'level': logging.INFO,
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'datefmt': '%d-%b-%y %H:%M:%S',
    'filename': 'log/log.log',
    'filemode': 'w'
}

DEFAULT_CONFIG_PATH = "config/default.toml"

if __name__ == "__main__":
    if not os.path.exists("log"):
        os.makedirs("log")

    logging.basicConfig(**LOGGING_CONFIG)
    logging.info("Starting run.py")

    parser = argparse.ArgumentParser(description="Minesweeper game")
    parser.add_argument("--config", type=str, help="Path to the config file")
    parser.add_argument("--test", action="store_true", help="Test the performance of the model")
    args = parser.parse_args()

    path = DEFAULT_CONFIG_PATH
    if args.config:
        path = args.config
        path = f"config/{path}.toml"
        logging.info(f"Using config path : {path}")
    else:
        logging.info(f"Using default config path : {path}")

    app = App(path)
    if args.test:
        app.test_performance()
    else:
        app.run()
