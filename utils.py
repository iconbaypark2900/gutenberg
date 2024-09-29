import functools
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def error_handler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    return wrapper

def log_function_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Calling function: {func.__name__}")
        result = func(*args, **kwargs)
        logger.info(f"Function {func.__name__} completed")
        return result
    return wrapper