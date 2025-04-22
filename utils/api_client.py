import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time
from ratelimit import limits, sleep_and_retry
from .config import Config

class APIClient:
    def __init__(self):
        self.config = Config()
        self.session = self._create_session()
        
    def _create_session(self):
        session = requests.Session()
        retries = Retry(
            total=self.config.get('api.retries', 3),
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session
    
    @sleep_and_retry
    @limits(calls=100, period=60)  # Rate limit: 100 calls per minute
    def request(self, method, url, **kwargs):
        timeout = self.config.get('api.timeout', 10)
        try:
            response = self.session.request(method, url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise APIError(f"API request failed: {str(e)}")

class APIError(Exception):
    pass
