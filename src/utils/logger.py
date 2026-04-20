import logging
import os

def custom_logger():
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/training.log"),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)