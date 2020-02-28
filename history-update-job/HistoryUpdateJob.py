import sys                # nopep8
sys.path.append('config')  # nopep8
sys.path.append('shared')  # nopep8
print(sys.path)           # nopep8

import pandas as pd
import psycopg2
import sqlalchemy
import investpy
import datetime
import logging

from config import config, assets, save_assets
from Database import Database 


class HistoryUpdateJob:
    """Updates the latest history for a list of assets."""

    def __init__(self,asset_config, connection):
        """Constructor"""
        self.asset_config = asset_config
        self.con = connection

    def __get_historical_data(self, etf, country, from_date, to_date):
        if to_date is None:
            now = datetime.datetime.now()
            to_date = "{day}/{month}/{year}".format(
                day=now.day, month=now.month, year=now.year)

        history = investpy.etfs.get_etf_historical_data(
            etf, country, from_date, to_date, as_json=False, order='ascending', interval='Daily')
        return history

    def __get_latest_Update_Date(self, etf_name):
        try:
            sql = 'SELECT MAX("Date") from "{0}"'.format(etf_name)
            return self.con.execute(sql).scalar()

        except Exception:
            pass

        return None

    def __filter__(self, df, latest_date):
        if latest_date:
            latest_date_str = "{year}-{month}-{day}".format(
                year=latest_date.year, month=latest_date.month, day=latest_date.day
            )
            later = df.index > latest_date_str
            return df[later]
        else:
            return df

    def __write_data(self, isin: str, df: pd.DataFrame):
        try:
            df.to_sql(isin, self.con, if_exists='append')

        except Exception as e:
            print(e)
            pass

    def run(self, from_date="01/01/1926", to_date=None):
        universe = self.asset_config
        for asset in universe:
            isin = asset['isin']
            print("PROCESSING {0}".format(isin) + ":")
            search_results = investpy.search_etfs(by='isin', value=isin)
            if search_results.shape[0] > 0:
                try:
                    etf_name = search_results.iloc[0]['name']
                    country = asset['country']
                    asset['name'] = etf_name
                    asset['currency'] = search_results.iloc[0]['currency']
                    asset['asset_class'] = search_results.iloc[0]['asset_class']
                    asset['stock_exchange'] = search_results.iloc[0]['stock_exchange']
                    print("  Downloading data...")
                    history = self.__get_historical_data(
                        etf_name, country, from_date, to_date)
                    # Only append the latest update
                    latest_date = self.__get_latest_Update_Date(isin)
                    history = self.__filter__(history, latest_date)
                    print("  Writing the latest data into database...")
                    self.__write_data(isin, history)

                except Exception as error:
                    print("  ERROR: {0}".format(error))
                    pass

                print("COMPLETE")
                print("----------------------")

asset_config = assets()
db_config = config()

database = Database(db_config)
connection = database.get_connection()

job = HistoryUpdateJob(asset_config, connection)
job.run(from_date="01/01/2000")
save_assets(asset_config)