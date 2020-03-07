
import sqlalchemy

class Database:

    def __init__(self, config):
        """Constructor
            - Takes a configuration map with database parameters.
        """
        url = "postgresql://{user}:{password}@{host}/{database}".format(
            user=config['user'],
            password=config['password'],
            host=config['host'],
            database=config['database']
        )
        self.engine = sqlalchemy.create_engine(url)

    def get_connection(self):  
        """Get a new connecton to the database
            - Returns a connection object"""
        return self.engine.connect()

    def close(self):
        self.engine.dispose()
