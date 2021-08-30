import cx_Oracle
import numpy as np
import pandas as pd
from shutil import rmtree
from config import *
import pickle
from gensim.models.doc2vec import TaggedDocument
from collections import defaultdict, Counter
import math
from utils import convert_city_name


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Product():
    def __init__(self, row, index):
        self.usages = 0

        self.hierarchy_indices = []
        self.index = index
        self._representation = 4
        self.barcode = str(row[0])
        self.hierarchy_names = [str(i // 2) + '_' + row[1][i] for i in [1, 3, 5, 7, 0]]
        self.price = row[1][-4]
        self.sugar = row[1][-3]
        self.sodium = row[1][-1]
        self.fats = row[1][-2]

    def set_representation_level(self, r):
        self._representation = r

    @property
    def representation(self):
        return self.hierarchy_names[self._representation] + ('_child' if self._representation < 4 else '')

    @property
    def name(self):
        return self.hierarchy_names[-1]


class Transaction():
    def __init__(self, t_id, u_id, total=0, location=None, date=None):
        self.id = t_id
        self.user_id = u_id
        self.barcodes = []
        self.location = location
        self.total = total
        self._test_index = -1
        self.test_item = None
        self.date = date

    def create_test_item(self):
        if len(self.barcodes) > 10:
            index = np.random.randint(0, len(self.barcodes))
            self.test_item = (self.barcodes.pop(index))
            self._test_index = index

    def return_test_item(self):
        if self._test_index != -1:
            index = self._test_index
            self.barcodes.insert(self.test_item[0], index)
            self._test_index = -1
            self.test_item = None

    @staticmethod
    def to_sentence(transaction):
        return [Dataset().barcode_to_product[x].representation for x in transaction.barcodes] + \
               [transaction.user_id] * (transaction.user_id in Users().active_users)

    @staticmethod
    def to_doc(transaction):
        return TaggedDocument([Dataset().barcode_to_product[x].representation for x in transaction.barcodes],
                              [transaction.id, transaction.user_id])


class T2D:
    def __iter__(self):
        return map(Transaction.to_doc, Dataset().transactions)


class T2S:
    def __iter__(self):
        return map(Transaction.to_sentence, Dataset().transactions)


class Dataset(metaclass=Singleton):
    PRODUCT_USAGE_THRESHOLD = 3000

    def __init__(self):
        self.transactions = []
        self.products = []
        self.hierarchy_sizes = []
        self.index_of = {}
        self._load_data()

        self.barcode_to_product = {}
        for p in self.products:
            self.barcode_to_product[p.barcode] = p

    def _load_data(self):
        if os.path.exists(DATA_PATH):
            with open(DATA_PATH, 'rb') as f:
                self.transactions, self.products = pickle.load(f)
        else:
            if not os.path.exists(TRANSACTIONS_DF_PATH):
                print('fetching data from sql')
                transactions_df = self._load_sql_transactions()
            else:
                transactions_df = pd.read_pickle(TRANSACTIONS_DF_PATH)
            self._create_transactions(transactions_df)

            self._create_products(sorted(transactions_df.BARCODE.unique()))

            with open(DATA_PATH, 'wb') as f:
                pickle.dump([self.transactions, self.products], f)

    def _create_products(self, all_used_barcodes):
        dsn_tns = cx_Oracle.makedsn(SERVER_NAME, PORT, service_name=SERVICE_NAME)
        conn = cx_Oracle.connect(user=USER, password=PASSWORD, dsn=dsn_tns)
        c = conn.cursor()
        r = c.execute(PRODUCTS_QUERY)
        products_df = pd.DataFrame(r, columns=[str(x) for x in range(50, 81)]).set_index('50')
        self.products = []
        for i, row in enumerate(products_df.loc[all_used_barcodes].iterrows()):
            self.products.append(Product(row, i))
        print('created products')
        conn.close()
        do_again = True
        while do_again:
            do_again = False
            counts = self._get_representation_counts()
            for p in self.products:
                if counts[p.representation] < self.PRODUCT_USAGE_THRESHOLD and p._representation > 2:
                    p._representation -= 1
                    do_again = True

    def _create_transactions(self, transactions_df):
        print('loaded transactions df with {} transactions'.format(transactions_df.reset_index().OBJECT_ID.nunique()))
        transactions_dict = {}
        for barcode, t_id, quantity, item_price, u_id, total, long, lat, date in transactions_df.reset_index()[
            ['BARCODE', 'OBJECT_ID', 'QUANTITY', 'ITEM_PRICE', 'USER_ID', 'INVOICE_SUM', 'MERIDIAN', 'LATITUDE',
             'CREATION_DATE']].values:
            if t_id not in transactions_dict.keys():
                transactions_dict[t_id] = Transaction(t_id, u_id, total, (long, lat), date)
                if len(transactions_dict) % 1000000 == 0:
                    print(len(transactions_dict))
            transactions_dict[t_id].barcodes.append(barcode)

        self.transactions = list(transactions_dict.values())
        for t in self.transactions:
            t.barcodes = sorted(set(t.barcodes))
        self.transactions = [t for t in self.transactions if len(t.barcodes) > 4]
        for t in self.transactions:
            for b in t.barcodes:
                self.barcode_to_product[b].usages += 1
                if math.isnan(t.location[0]):
                    t.location = None

    @staticmethod
    def _load_sql_transactions():
        dsn_tns = cx_Oracle.makedsn(SERVER_NAME, PORT, service_name=SERVICE_NAME)
        conn = cx_Oracle.connect(user=USER, password=PASSWORD, dsn=dsn_tns)
        c = conn.cursor()
        r = c.execute(TRANSACTIONS_QUERY)
        csv_path = os.path.join(STORAGE_PATH, 'transactions_tmp.csv')

        with open(csv_path, 'w') as f:
            f.write(','.join(
                ['OBJECT_ID', 'INVOICE_SUM', 'USER_ID', 'BARCODE', 'QUANTITY', 'ITEM_PRICE', 'MERIDIAN', 'LATITUDE',
                 'CREATION_DATE']) + '\n')
            for i, row in enumerate(r):
                f.write(','.join([str(x) for x in row[:5] + row[6:9] + row[-3:-2]]) + '\n')
                if i % 10000000 == 0:
                    print(i)
        conn.close()

        dtype = {
            'OBJECT_ID': 'category',
            'USER_ID': 'category',
            'BARCODE': 'category',
            'INVOICE_SUM': 'float32',
            'QUANTITY': 'float16',
            'ITEM_PRICE': 'float32',
            'MERIDIAN': 'float32',
            'LATITUDE': 'float32'
        }
        transactions_df = pd.read_csv(csv_path, dtype=dtype, parse_dates=['CREATION_DATE'], na_values='None')
        transactions_df.to_pickle(TRANSACTIONS_DF_PATH)
        rmtree(csv_path)
        return transactions_df

    def _get_representation_counts(self):
        d = defaultdict(int)
        for p in self.products:
            d[p.representation] += p.usages
        return d
    
    def _create_indexing(self):
        # For each level of hierarchy count unique names, sort and give indices
        LEVEL_SIZE = [0 for i in range(DEPTH)]
        hierarchy_names = [set() for i in range(DEPTH)]
        for product in self.products:
            for i in range(DEPTH):
                hierarchy_names[i].add(product.hierarchy_names[i])

        #GROUP_self.index_of = {}  # Dict [name of hierarchy group/barcode: relative index in relevant hierarchy level]
        self.index_of = {}  # Dict {name of hierarchy group/barcode: absolute index in concatenated vector}
        offset = 0
        for i in range(DEPTH):
            for index, name in enumerate(sorted(hierarchy_names[i])):
                #GROUP_self.index_of[name] = index
                self.index_of[name] = index + offset
            offset += len(hierarchy_names[i])
            LEVEL_SIZE[i] = len(hierarchy_names[i])

            # Update products with relevant indices
        for product in self.products:
            product.hierarchy_indices = [self.index_of[x] for x in product.hierarchy_names]
        #return self.index_of, GROUP_self.index_of, LEVEL_SIZE


class Users(metaclass=Singleton):
    MIN_PURCHASES = 30
    MIN_TOTAL = 2000

    def __init__(self):
        self.tf_vectors = {}
        users_spent_money = defaultdict(int)
        users_transactions = defaultdict(int)
        for transaction in Dataset().transactions:
            user = transaction.user_id
            users_transactions[user] += 1
            users_spent_money[user] += transaction.total
        self.all_users = sorted(users_spent_money.keys())
        self.active_users = sorted([k for k in users_spent_money.keys()
                                    if (users_spent_money[k] >= self.MIN_TOTAL)
                                    and (users_transactions[k] >= self.MIN_PURCHASES)])
        print('Filtering users: {} out of {} remain. Percentage of transactions covered: {:.2f}'.format(
            len(self.active_users), 
            len(self.all_users), 
            100 * sum([users_transactions[k] for k in self.active_users]) / len(Dataset().transactions))
        )

        dsn_tns = cx_Oracle.makedsn(SERVER_NAME, PORT, service_name=SERVICE_NAME)
        conn = cx_Oracle.connect(user=USER, password=PASSWORD, dsn=dsn_tns)
        c = conn.cursor()
        r = c.execute(USERS_QUERY)
        users_df = pd.DataFrame(r, columns=['USER_ID', 'CITY']).set_index('USER_ID').loc[self.active_users]
        users_df = users_df[~users_df.CITY.isna()].reset_index()[['USER_ID', 'CITY']].sort_values('USER_ID')
        users_df.CITY = users_df.CITY.map(convert_city_name)
        self.active_users_cities = dict(users_df.values)


class CityStats(metaclass=Singleton):
    CITIES_TO_DROP = ['ערערה', 'כפר מנדא', 'ערערה-בנגב', 'אעבלין', 'אעבלין', 'טורעאן', 'כאבול', 'לקיה', 'נחף', 'עספיא']
    POPULATION_MIN = 12000
    
    def __init__(self):
        self.stats_df = pd.DataFrame(columns=['CITY', 'n_users'], data=Counter(Users().active_users_cities.values()).items())
        
        population_df = pd.read_csv('C://analytics/stats/population_age.csv', encoding='windows-1255',
                                    dtype='str').fillna(0)
        lfunc = lambda e: int(e.replace(',', '')) if e.find(',') != -1 else e
        population_df['total'] = population_df['סך הכל']
        population_df = population_df.applymap(lfunc).set_index('שם יישוב')
        population_df = population_df[population_df.total > self.POPULATION_MIN]
        population_df = population_df.drop(self.CITIES_TO_DROP, axis=0)

        self.used_cities = sorted([x for x in population_df.index.values])
        population_df = population_df.loc[self.used_cities]
        self.stats_df = self.stats_df.loc[self.used_cities]

        ages = [0, 18, 28, 38, 48, 58, 68, -1]
        v = population_df.values[:, 2:].astype(float)
        totals = population_df.total.values
        self.stats_df['population'] = totals
        for i in range(1, len(ages)):
            start, finish = ages[i - 1], ages[i]
            total = v[:, start:finish].sum(axis=1)
            if finish == -1:
                finish = 100
            self.stats_df[str(start) + '-' + str(finish - 1)] = total / totals * 100

        rel_df = pd.read_csv('C://analytics/stats/haredi.txt', encoding='windows-1255', dtype='str', delimiter=' ',
                             header=None)
        rel_df.columns = ['socio', 'a', 'rel_proportion', 'total', 'rel', 'name']

        def convert_name(s):
            if 'אביב' in s:
                return 'תל אביב-יפו'
            return ' '.join(s.split('$')[::-1]).strip()

        rel_df.name = rel_df.name.map(convert_name)
        rel_df.total = rel_df.total.map(lambda x: int(x.replace(',', '')))
        rel_df.rel = rel_df.rel.map(lambda x: min(int(x.replace(',', '')), 100))
        rel_df = rel_df[rel_df.total > 30000]
        self.stats_df['haredim'] = 0.0
        self.stats_df.loc[rel_df.name, 'haredim'] = np.array(rel_df.rel_proportion.map(lambda x: float(x[:-1])))

        diabetes_df = pd.read_csv('C://analytics/stats/diabetes.csv', delimiter=';', header=None)
        diabetes_df.columns = ['a', 'name', 'cases', 'proportion', 'fixed_proportion', 'error']
        diabetes_df = diabetes_df.set_index('name')
        diabetes_df = diabetes_df.rename(index=lambda x: x.replace('נצרת עילית', "נוף הגליל"))
        diabetes_df = diabetes_df.rename(index=lambda x: x.replace('תל אביב -יפו', "תל אביב-יפו"))

        self.stats_df['diabetes_ratio'] = self.stats_df.index.map(
            lambda x: diabetes_df.loc[x].fixed_proportion if x != 'חריש' else diabetes_df.fixed_proportion.mean())

        big_df = pd.read_csv('C://analytics/stats/big_table.csv', encoding='windows-1255')
        big_df = big_df[['Name', 'Total Population', 'Average Salary', 'Socioeconomic Value', 'Periphery Value',
                         'Jews Rate']].set_index('Name')
        big_df.columns = ['population', 'salary', 'socio', 'periphery', 'jews_ratio']
        big_df = big_df.rename(index=lambda x: x.replace('נצרת עילית', "נוף הגליל"))
        big_df = big_df.loc[self.used_cities]

        self.stats_df['salary'] = big_df['salary']
        self.stats_df['socio'] = big_df['socio']
        self.stats_df['periphery'] = big_df['periphery']
        self.stats_df['non_jews_ratio'] = 100 - big_df['jews_ratio'].fillna(0)
        self.stats_df['user_percentage'] = self.stats_df.n_users / self.stats_df.population * 100
        self.stats_df.head()

        self.stats_df.to_excel('stats.xls')


if __name__ == '__main__':
    Dataset()
    Users()
    CityStats()