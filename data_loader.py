import cx_Oracle
import numpy as np
import pandas as pd
from config import *
import pickle
from gensim.models.doc2vec import TaggedDocument
from collections import defaultdict, Counter
from utils import convert_city_name


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Product:
    # Python object representing line from SQL products table.
    DEFAULT_PRICE = 10

    def __init__(self, row, index):
        self.usages = 0

        self.hierarchy_indices = []  # 5 indices in long vector representing the full product hierarchy (FMCG, CLASS, CAT, SUBCAT, BARCODE)
        self.index = index  # Number representing the product
        self.representation_level = 4  # Representation level(from 0 to 4, 4 meaning BARCODE). Used when the product has low frequency and we use it's hierarchy predcessors instead
        self.barcode = str(row[0])
        self.hierarchy_names = [str(i // 2) + '_' + row[1][i] for i in [1, 3, 5, 7]]  # 5 names of products hierarchy: from FMCG to BARCODE.
        self.hierarchy_names.append(row[1][0])
        self.price = row[1][-6]
        if self.price is None or self.price == 0:
            self.price = row[1][-5]
        if self.price is None or self.price == 0:
            self.price = self.DEFAULT_PRICE

        self.sugar = row[1][-3]
        self.sodium = row[1][-1]
        self.fats = row[1][-2]

    def decrease_representation_level(self):
        self.representation_level -= 1

    @property
    def representation(self):
        if self.representation_level == 4:
            return self.barcode
        return self.hierarchy_names[self.representation_level] + '_child'

    @property
    def name(self):
        return self.hierarchy_names[-1]


class Transaction:
    def __init__(self, t_id, u_id, total=0, location=None, date=None):
        self.id = t_id
        self.user_id = u_id
        self.barcodes = [] # disabled by default due to the memory limitations
        self.length = 0 # number of barcodes
        self.location = location
        if self.location[0] is None:
            self.location = None
        self.total = total
        self.date = date

    def to_sentence(self, barcodes):
        uid = self.user_id
        sentence = [Dataset().barcode_to_product[x].representation for x in barcodes]
        if uid in Users().active_users_set:
            sentence.append(uid)
        if uid in Users().active_users_cities.keys():
            sentence.append(Users().active_users_cities[uid])
        return sentence

    def to_document(self, barcodes):
        tags = [self.id]
        if self.user_id in Users().active_users:
            tags.append(self.user_id)
        if self.user_id in Users().active_users_cities.keys():
            tags.append(Users().active_users_cities[self.user_id])
        return TaggedDocument([Dataset().barcode_to_product[x].representation for x in barcodes], tags)


class Dataset(metaclass=Singleton):
    PRODUCT_USAGE_THRESHOLD = 3000

    def __init__(self):
        self.transactions = [] # list of Transaction objects
        self.products = [] # list of Product objects
        self.barcode_to_product = {} # dict barcode -> Product
        self.id_to_transaction = {} # dict t_id -> Transaction
        self._load_data()
        
        self.hierarchy_sizes = [] # list of 5 integers, each one represents size of relevant hierarchy level
        self.index_of = {} # dict mapping hierarchy names and barcodes to indices in long vector
        self._create_indexing()
    
    def load_barcodes_to_memory(self):
        with open(BARCODE_PATH, 'r') as f:
            line = f.readline()
            while line:
                data = line[:-1].split(' ')
                self.id_to_transaction[data[0]].barcodes = data[1:]
                line = f.readline()

    def unload_barcodes_from_memory(self):
        for t in self.transactions:
            t.barcodes = None

    def barcode_iterator(self):
        with open(BARCODE_PATH, 'r') as f:
            line = f.readline()
            while line:
                data = line[:-1].split(' ')
                yield data[0], data[1:]
                line = f.readline()
    
    def _load_data(self):
        """
        Main loading function. Loads transactions and products. Overall flow goes as follows:
        0) Check if pickled data exists, load and exit if yes
        1) Check if CSV file with transactions exists. If not - load it from database.
        2) Go over csv, create Transaction objects, filter out transactions with less that 5 items.
        3) Create product objects for purchased products, count usages, update product representations
        4) DUMP ASIDE information about barcodes
        """
        if os.path.exists(DATA_PATH):
            print('Loading pickled data')
            with open(DATA_PATH, 'rb') as f:
                self.transactions, self.products = pickle.load(f)
            for p in self.products:
                self.barcode_to_product[p.barcode] = p
            for t in self.transactions:
                self.id_to_transaction[t.id] = t
            print(len(self.transactions), 'transactions loaded')
        else:
            if not os.path.exists(TRANSACTIONS_CSV_PATH):
                print('fetching data from sql')
                self._load_sql_transactions()
            
            # Go over CSV, create transaction objects
            all_barcodes = self._create_transactions()
            for t in self.transactions:
                self.id_to_transaction[t.id] = t
            print('created transactions')
            
            # Create products
            self._create_products(all_barcodes)
            for p in self.products:
                self.barcode_to_product[p.barcode] = p
            
            # COunt usages
            for t in self.transactions:
                for b in t.barcodes:
                    self.barcode_to_product[b].usages += 1
            
            # Update representations
            for i in range(2):
                counts = defaultdict(int)
                for p in self.products:
                    counts[p.representation] += p.usages
                for p in self.products:
                    if counts[p.representation] < self.PRODUCT_USAGE_THRESHOLD:
                        p.decrease_representation_level()
            print('created products')

            with open(BARCODE_PATH, 'w') as f:
                for t in self.transactions:
                    line = [str(t.id)] + [str(x) for x in t.barcodes]
                    f.write(' '.join(line)+'\n')
                    t.barcodes = None
            
            with open(DATA_PATH, 'wb') as f:
                pickle.dump([self.transactions, self.products], f)

    def _create_transactions(self):
        print('Parsing csv file')
        id_to_index = {}
        all_barcodes = set()
        with open(TRANSACTIONS_CSV_PATH, 'r') as f:
            line = f.readline()
            line = f.readline()
            while line:
                t_id, total, u_id, barcode, quantity, item_price, long, lat, date = line[:-1].split(',')
                if t_id not in id_to_index.keys():
                    self.transactions.append(Transaction(t_id, u_id, float(total), (long, lat), date))
                    id_to_index[t_id] = len(self.transactions) - 1
                    if len(id_to_index) % 1000000 == 0:
                        print('Finished', len(id_to_index), 'transactions')
                self.transactions[id_to_index[t_id]].barcodes.append(barcode)
                line = f.readline()
                all_barcodes.add(barcode)
        print('finished parsing csv')
        
        for t in self.transactions:
            t.barcodes = sorted(set(t.barcodes))
            t.length = len(t.barcodes)
        self.transactions = [t for t in self.transactions if t.length > 4]
            
        return all_barcodes
                
    def _create_products(self, all_used_barcodes):
        dsn_tns = cx_Oracle.makedsn(SERVER_NAME, PORT, service_name=SERVICE_NAME)
        conn = cx_Oracle.connect(user=USER, password=PASSWORD, dsn=dsn_tns)
        c = conn.cursor()
        r = c.execute(PRODUCTS_QUERY)
        products_df = pd.DataFrame(r, columns=[str(x) for x in range(50, 81)]).set_index('50')
        self.products = []
        for i, row in enumerate(products_df.loc[all_used_barcodes].iterrows()):
            self.products.append(Product(row, i))
        conn.close()
      
    @staticmethod
    def _load_sql_transactions():
        dsn_tns = cx_Oracle.makedsn(SERVER_NAME, PORT, service_name=SERVICE_NAME)
        conn = cx_Oracle.connect(user=USER, password=PASSWORD, dsn=dsn_tns)
        c = conn.cursor()
        r = c.execute(TRANSACTIONS_QUERY)

        with open(TRANSACTIONS_CSV_PATH, 'w') as f:
            f.write(','.join(
                ['OBJECT_ID', 'INVOICE_SUM', 'USER_ID', 'BARCODE', 'QUANTITY', 'ITEM_PRICE', 'MERIDIAN', 'LATITUDE',
                 'CREATION_DATE']) + '\n')
            for i, row in enumerate(r):
                f.write(','.join([str(x) for x in row[:5] + row[6:9] + row[-3:-2]]) + '\n')
                if i % 10000000 == 0:
                    print(i)
        conn.close()

    def _create_indexing(self):
        # For each level of hierarchy count unique names, sort and give indices
        self.hierarchy_sizes = [0 for i in range(DEPTH)]
        hierarchy_names = [set() for i in range(DEPTH)]
        for product in self.products:
            for i in range(DEPTH):
                hierarchy_names[i].add(product.hierarchy_names[i])

        self.index_of = {}  # Dict {name of hierarchy group/barcode: absolute index in concatenated vector}
        offset = 0
        for i in range(DEPTH):
            for index, name in enumerate(sorted(hierarchy_names[i])):
                self.index_of[name] = index + offset
            offset += len(hierarchy_names[i])
            self.hierarchy_sizes[i] = len(hierarchy_names[i])
        # Update products with relevant indices
        for product in self.products:
            product.hierarchy_indices = [self.index_of[x] for x in product.hierarchy_names]


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
        self.active_users_set = set(self.active_users)
        print('Filtering users: {} out of {} remain. Percentage of transactions covered: {:.2f}'.format(
            len(self.active_users), 
            len(self.all_users), 
            100 * sum([users_transactions[k] for k in self.active_users]) / len(Dataset().transactions))
        )

        dsn_tns = cx_Oracle.makedsn(SERVER_NAME, PORT, service_name=SERVICE_NAME)
        conn = cx_Oracle.connect(user=USER, password=PASSWORD, dsn=dsn_tns)
        c = conn.cursor()
        r = c.execute(USERS_QUERY)
        users_df = pd.DataFrame(r, columns=['USER_ID', 'CITY'],dtype=str).set_index('USER_ID').loc[self.active_users]
        users_df = users_df[~users_df.CITY.isna()].reset_index()[['USER_ID', 'CITY']].sort_values('USER_ID')
        users_df.CITY = users_df.CITY.map(convert_city_name)
        self.active_users_cities = dict(users_df.values)


class CityStats(metaclass=Singleton):
    CITIES_TO_DROP = ['ערערה', 'כפר מנדא', 'ערערה-בנגב', 'אעבלין', 'אעבלין', 'טורעאן', 'כאבול', 'לקיה', 'נחף', 'עספיא']
    POPULATION_MIN = 12000
    
    def __init__(self):
        self.stats_df = pd.DataFrame(columns=['CITY', 'n_users'],
                                     data=Counter(Users().active_users_cities.values()).items()).set_index('CITY')
        
        population_df = pd.read_csv(os.path.join(CITY_PATH, 'population_age.csv'), encoding='windows-1255',
                                    dtype='str').fillna(0)
        lfunc = lambda e: int(e.replace(',', '')) if e.find(',') != -1 else e
        population_df['total'] = population_df['סך הכל']
        population_df = population_df.applymap(lfunc).set_index('שם יישוב')
        population_df = population_df[population_df.total > self.POPULATION_MIN]
        population_df = population_df.drop(self.CITIES_TO_DROP, axis=0)

        self.used_cities = sorted(population_df.index.values)
        self.population_df = population_df.loc[self.used_cities]
        #print(self.used_cities,'\n', self.stats_df.index.values)
        self.stats_df = self.stats_df.loc[self.used_cities]

        ages = [0, 4,9,14, 19, 22, 31, 51, 66, -1]
        v = self.population_df.values[:, 2:78].astype(float)
        totals = self.population_df.total.values
        self.stats_df['population'] = totals
        for i in range(1, len(ages)):
            start, finish = ages[i - 1], ages[i]
            if finish == -1:
                total = v[:, start:].sum(axis=1)
                self.stats_df['66+'] = total / totals * 100
            else:
                total = v[:, start:finish].sum(axis=1)
                self.stats_df['{:02d}-{:02d}'.format(start,finish-1)] = total / totals * 100

        rel_df = pd.read_csv(os.path.join(CITY_PATH, 'haredi.txt'), encoding='windows-1255', dtype='str', delimiter=' ',
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

        big_df = pd.read_csv(os.path.join(CITY_PATH, 'big_table.csv'), encoding='windows-1255')
        big_df = big_df[['Name', 'Total Population', 'Average Salary', 'Socioeconomic Value', 'Periphery Value',
                         'Jews Rate', 'Eligible to Receive a Bagrut Among 12th Grage Rate', 'College Graduation Rate Among 35-55 Rate', 'Students Rate']].set_index('Name')
        big_df.columns = ['population', 'salary', 'socio', 'periphery', 'jews_ratio', 'bagrut', 'college grads rate', 'students rate']
        big_df = big_df.rename(index=lambda x: x.replace('נצרת עילית', "נוף הגליל"))
        big_df = big_df.loc[self.used_cities]

        self.stats_df['salary'] = big_df['salary']
        self.stats_df['socio'] = big_df['socio']
        self.stats_df['periphery'] = big_df['periphery']
        self.stats_df['non_jews_ratio'] = 100 - big_df['jews_ratio'].fillna(0)
        self.stats_df['user_percentage'] = self.stats_df.n_users / self.stats_df.population * 100
        
        #self.stats_df.to_excel('stats.xls')


# class TransactionsToDocuments:
#     def __iter__(self):
#         return map(Transaction.to_document, Dataset().transactions)


# class TransactionsToSentences:
#     def __iter__(self):
#         return map(Transaction.to_sentence, Dataset().transactions)


if __name__ == '__main__':
    Dataset()
    Users()
    CityStats()
