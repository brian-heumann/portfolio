import sys                          #nopep8
sys.path.append('configuration')    #nopep8
sys.path.append('shared')           #nopep8

import pandas as pd 
from functools import reduce 

from config import config, assets
from Database import Database 
from Asset import Asset

db = Database(config())
con = db.get_connection()

dataframes = []
for a in assets():
    asset = Asset(a['isin'])
    returns = asset.load_returns(con)
    dataframes.append(returns)

con.close()

merged = reduce( lambda left, right: pd.merge(left, right, on='Date', how='outer'), dataframes)
merged.dropna(inplace=True)
print(merged)