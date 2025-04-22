import logging
from pathlib import Path
from .config import Config

class Logger:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance
    
    def _initialize_logger(self):
        config = Config()
        log_level = getattr(logging, config.get('logging.level', 'INFO'))
        log_file = config.get('logging.file', 'financial_analyzer.log')
        log_path = Path(__file__).parent.parent / 'logs' / log_file
        
        # Create logs directory if it doesn't exist
        log_path.parent.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('FinancialAnalyzer')
    
    def info(self, msg):
        self.logger.info(msg)
    
    def error(self, msg):
        self.logger.error(msg)
    
    def warning(self, msg):
        self.logger.warning(msg)
    
    def debug(self, msg):
        self.logger.debug(msg)
