import investpy
import psycopg2
import pandas as pd
from config.config import config

params = config()
print(params['user'])

conn = psycopg2.connect(**params)


def create_table(isin: str):
    cursor = conn.cursor()
    sql =  """create table if not exists "{table_name}" (
        Date varchar(10), 
        Open float, 
        High float, 
        Low float, 
        Close float, 
        Currency varchar(10))""".format(table_name=isin)
    cursor.execute(sql)
    conn.commit()
    cursor.close()

def write_data(isin: str, df: pd.DataFrame):
    print("Write data for {0}".format(isin))


universe = ['IE00B14X4T88', 'IE00BKM4GZ66', 'IE00BQN1K901', 'IE00B2QWDR12', 'IE00BKWQ0M75', 'IE00B78JSG98', 'LU1829218749', 'IE00B1FZS350']
for isin in universe:
    create_table(isin)

    search_results = investpy.search_etfs(by='isin', value=isin)
    if search_results.shape[0] > 0:
        # xetra = (search_results['stock_exchange'] == 'Xetra') 
        etf = search_results.iloc[0]['name'] 
        country='germany' 
        from_date = '01/01/2010'
        to_date = '20/02/2020'
        try:
            history = investpy.etfs.get_etf_historical_data(etf, country, from_date, to_date, as_json=False, order='ascending', interval='Daily')
            write_data(isin, history)
        except RuntimeError as error:
            print(error)
            pass

        print("----------------------")

conn.close()