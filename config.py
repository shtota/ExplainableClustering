import os
TEST_MODE = False


# SQL
SERVER_NAME = 'Bianalyzelab'
PORT = '1521'
SERVICE_NAME='ALP'
with open('login.txt', 'r') as f:
    USER, PASSWORD  = f.read().split('\n')

DEPTH = 5

# PATHS

STORAGE_PATH = 'Q://'
TRANSACTIONS_CSV_PATH = os.path.join(STORAGE_PATH, 'transactions.csv')
DATA_PATH = os.path.join(STORAGE_PATH, 'data.pkl')
TEST_DATA_PATH = os.path.join(STORAGE_PATH, 'test_data.pkl')
EMBEDDING_PATH = os.path.join(STORAGE_PATH, 'gensim.w2v')
CLUSTERING_PATH = os.path.join(STORAGE_PATH, 'clustering.pkl')
CITY_PATH = './stats/'


# QUERIES
PRODUCTS_QUERY = "select * from STRAUSS_PLUS.STORENEXT_CATALOG c"
USERS_QUERY = "select USER_ID, CITY from STRAUSS_PLUS.V_ANALYZE_USERS"
TRANSACTIONS_QUERY = """select * from STRAUSS_PLUS.TEMP_TRANSACTION_TECHNION t
        left join (select ITEM_ID from STRAUSS_PLUS.STORENEXT_CATALOG) c ON t.barcode = c.item_id
        where  t.item_price between 0 and 300
        and c.item_id is  not null
        """

