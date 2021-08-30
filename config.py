import os

# SQL
SERVER_NAME = 'Bianalyzelab'
PORT = '1521'
SERVICE_NAME='ALP'
USER='AlexanderCh'
PASSWORD='Aa123456'

DEPTH = 5

# PATHS

STORAGE_PATH = 'Q://'
TRANSACTIONS_DF_PATH = os.path.join(STORAGE_PATH, 'transactions_df.pkl')
DATA_PATH = os.path.join(STORAGE_PATH, 'data.pkl')
PRODUCTS_PATH = os.path.join(STORAGE_PATH, 'products.pkl')
EMBEDDING_PATH = os.path.join(STORAGE_PATH, '/gensim.w2v')
CLUSTERING_PATH = os.path.join(STORAGE_PATH, 'clustering.pkl')

# QUERIES
PRODUCTS_QUERY = "select * from STRAUSS_PLUS.STORENEXT_CATALOG c"
USERS_QUERY = "select * from STRAUSS_PLUS.USERS"
TRANSACTIONS_QUERY = """select * from STRAUSS_PLUS.TEMP_TRANSACTION_TECHNION t
        left join (select ITEM_ID from STRAUSS_PLUS.STORENEXT_CATALOG) c ON t.barcode = c.item_id
        where  t.item_price between 0 and 300
        and c.item_id is  not null
        """