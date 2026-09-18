import logging
import sys
from src.config import settings

def setup_logging():
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    logger = logging.getLogger("traceiq")
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(ch)
        
    return logger

logger = setup_logging()
