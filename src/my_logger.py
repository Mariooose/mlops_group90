import sys

from loguru import logger

logger.remove()  # Remove the default logger
logger.add(sys.stdout, level="WARNING")  # Add a new logger with WARNING level
logger.add("my_log.log", level="DEBUG", rotation="100 MB")
