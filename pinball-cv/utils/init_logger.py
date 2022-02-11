# """Creates a custom logging object to log to file and console as configured in the config.py file."""
# import logging

# from . import config

# # Create parent directory for log files if it doesn't exist
# config.LOG_PATH.parents[0].mkdir(parents=True, exist_ok=True)

# logger = logging.getLogger(__name__)
# """Logger: logging object for pinball-cv."""
# logger.setLevel(logging.DEBUG)

# # Sets formatting for loggers
# file_format = logging.Formatter("%(asctime)s:%(filename)s:%(levelname)s:%(message)s")
# stream_format = logging.Formatter("%(filename)s:%(levelname)s:%(message)s")

# # Creates handlers
# file_handler = logging.FileHandler(config.LOG_PATH)
# file_handler.setLevel(config.FILE_LOG_LEVEL)
# file_handler.setFormatter(file_format)

# stream_handler = logging.StreamHandler()
# stream_handler.setLevel(config.STREAM_LOG_LEVEL)
# stream_handler.setFormatter(stream_format)

# if config.LOG_TO_FILE:
#     logger.addHandler(file_handler)
# logger.addHandler(stream_handler)
