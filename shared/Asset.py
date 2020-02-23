
class Asset:

    def __init__(self, isin: str, weight: float, exchange: str):
        self.isin = isin
        self.target = weight 
        self.exchange = exchange

    