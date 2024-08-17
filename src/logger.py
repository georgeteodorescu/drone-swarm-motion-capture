# ---------------------------------------------------------------------
# ----------------------------- logger.py -----------------------------
# ---------------------------------------------------------------------
import logging
import os
import sys
from datetime import datetime


# metaclasa singleton pentru a avea doar o instanta a clasei Logger
class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


# clasa Logger care foloseste metaclasa SingletonMeta
class Logger(metaclass=SingletonMeta):
    def __init__(self, log_to_file=False, filename=None):
        self.logger = None
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.log_to_file = log_to_file

            if self.log_to_file:
                # setare director pentru a loga fisierele
                log_dir = os.path.join(os.path.dirname(__file__), 'logs')
                print(log_dir)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                if filename is not None:
                    self.filename = filename
                else:
                    self.filename = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")
            self.setup_logger()

    def setup_logger(self):
        # crearea unui logger cu nivel de logare setat la DEBUG
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # logare in terminal (stdout)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        # logare in fisier
        if self.log_to_file and hasattr(self, 'filename'):
            file_handler = logging.FileHandler(self.filename)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    # metoda pentru logarea mesajelor de info
    def info(self, message):
        self.logger.info(message)

    # metoda pentru logarea mesajelor de debug
    def debug(self, message):
        self.logger.debug(message)

    # metoda pentru logarea mesajelor de warning
    def warning(self, message):
        self.logger.warning(message)

    # metoda pentru logarea mesajelor de eroare
    def error(self, message):
        self.logger.error(message)

    # metoda pentru logarea mesajelor critical
    def critical(self, message):
        self.logger.critical(message)
