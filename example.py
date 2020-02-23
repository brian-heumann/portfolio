import investpy
import sqlalchemy
import psycopg2
import pandas as pd
from config.config import config, assets, save_assets

params = config()
print(params['user'])

engine = sqlalchemy.create_engine("postgresql://quotes:clue0QS-train@raspberrypi/quotes")
con = engine.connect()


def write_data(isin: str, df: pd.DataFrame):
    try:
        df.to_sql(isin, con, if_exists='append')

    except Exception as e:
        print(e)
        pass


def get_historical_data(etf, country='germany', from_date="01/01/2000", to_date="22/02/2020"):
    history = investpy.etfs.get_etf_historical_data(etf, country, from_date, to_date, as_json=False, order='ascending', interval='Daily')
    return history


universe = assets()
for asset in universe:
    isin = asset['isin']
    country = asset['country']
    print("PROCESSING {0}".format(isin) + ":")
    search_results = investpy.search_etfs(by='isin', value=isin)
    if search_results.shape[0] > 0:
        etf_name = search_results.iloc[0]['name'] 
        asset['name'] = etf_name
        asset['currency'] = search_results.iloc[0]['currency'] 
        asset['asset_class'] = search_results.iloc[0]['asset_class'] 
        asset['stock_exchange'] = search_results.iloc[0]['stock_exchange'] 
        try:
            print("  Downloading data...")            
            history = get_historical_data(etf_name, country)

            print("  Writing data into database...")
            write_data(isin, history)
        except Exception as error:
            print(error)
            pass

        print("COMPLETE")
        print("----------------------")

con.close()
save_assets(universe)