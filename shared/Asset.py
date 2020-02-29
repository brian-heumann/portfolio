import sqlalchemy
import pandas as pd 


class Asset:
    def __init__(self, isin: str) 
        self.isin = isin 

    def load_prices(con):
        self.data = pd.read_sql_table(self.isin, con)

    def to_returns():
        return self.data.pct_change().dropna()
