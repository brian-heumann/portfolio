import sqlalchemy
import pandas as pd 


class Asset:
    def __init__(self, isin: str): 
        self.isin = isin 

    def load_prices(self, con):
        self.data = pd.read_sql_table(self.isin, con, index_col='Date', parse_dates=True)
        return self.data

    def load_returns(self, con:sqlalchemy.engine):
        data = pd.read_sql_table(self.isin, con, index_col='Date', parse_dates=True, columns=['Close'])
        data.columns=[self.isin]
        return (1+data.pct_change().dropna())
