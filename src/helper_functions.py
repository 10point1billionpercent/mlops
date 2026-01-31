import logging
import os
from dotenv import load_dotenv

# Base directory (mlops/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load .env explicitly
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

# Logs directory
LOGS_DIR = os.path.join(BASE_DIR, os.getenv("LOGS_DIR", "logs"))
os.makedirs(LOGS_DIR, exist_ok=True)

# Log file
LOG_FILE = os.path.join(LOGS_DIR, "mlops_training.log")

# Logging config
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_logger():
    return logging.getLogger()

def log_info(message: str):
    logger = get_logger()
    logger.info(message)
    print(f"INFO: {message}")

def log_error(message: str):
    logger = get_logger()
    logger.error(message)
    print(f"ERROR: {message}")

def log_warning(message: str):
    logger = get_logger()
    logger.warning(message)
    print(f"WARNING: {message}")
