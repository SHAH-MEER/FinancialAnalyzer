from .providers import YahooFinanceProvider, NewsAPIProvider, CryptoProvider

class DataGateway:
    def __init__(self):
        self.providers = {
            'market': YahooFinanceProvider(),
            'news': NewsAPIProvider(),
            'crypto': CryptoProvider()
        }
    
    def get_data(self, provider, **kwargs):
        return self.providers[provider].fetch(**kwargs)
