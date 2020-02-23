from configparser import ConfigParser

filename = 'config/assets.ini'
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

