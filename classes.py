class Product():
    def __init__(self, row, prices, index):
        self.usages = 0

        self.hierarchy_indices = []
        self.index = index
        self._representation = 4
        self.barcode = str(row[0])
        self.hierarchy_names = [str(i // 2) + '_' + row[1][i] for i in [1, 3, 5, 7, 0]]
        self.price = min(prices.get(self.barcode, 10), 100)
        self.sugar = self._red_sticker_converter(row[1][-3])
        self.sodium = self._red_sticker_converter(row[1][-1])
        self.fats = self._red_sticker_converter(row[1][-2])

    def set_representation_level(self, r):
        self._representation = r

    @property
    def representation(self):
        return self.hierarchy_names[self._representation] + ('_child' if self._representation < 4 else '')

    @property
    def name(self):
        return self.hierarchy_names[-1]

    def _red_sticker_converter(self, x):
        if x is None:
            x = 'לא ידוע'
        if x.startswith('עם'):
            return 'yes'
        if x.startswith('ללא'):
            return 'no'
        return 'unknown'


class Transaction():
    def __init__(self, t_id, u_id, total=0, location=None, date=None):
        self.id = t_id
        self.user_id = u_id
        self.barcodes = []
        self.location = location
        # self.indices = []
        # self.quantities = []
        # self.prices = []
        self.total = total  # only for strauss dataset
        self._test_index = -1
        self.test_item = None
        self.date = date

    def create_test_item(self):  # need to refactor this function.
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

    def to_sentence(self):
        return [barcode_to_product[x].representation for x in self.barcodes] + [self.user_id] * (
                    self.user_id in filtered_users)
