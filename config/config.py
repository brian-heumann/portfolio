
from configparser import ConfigParser


def config(filename='config/database.ini', section='postgresql'):
    parser = ConfigParser()
    parser.read(filename)

    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception(
            "Section: {0} not found in file: {1}".format(section, filename))

    return db


def assets(filename='config/assets.ini'):
    parser = ConfigParser()
    parser.read(filename)

    assets = []
    sections = parser.sections()
    for section in sections:
        asset = {}
        asset['isin'] = section
        params = parser.items(section)
        for param in params:
            asset[param[0]] = param[1]
        assets.append(asset)

    return assets


def save_assets(assets, filename='config/assets.ini'):
    parser = ConfigParser()
    parser.read(filename)

    parameters = ['name', 'country', 'currency',
                  'asset_class', 'stock_exchange']
    for asset in assets:
        print(asset)
        for param in parameters:
            parser.set(asset['isin'], param, asset[param])

    with open(filename, 'w') as config_file:
        parser.write(config_file)
