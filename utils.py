from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def smart_inverse(x):
    if x is None:
        return "None"
    x = x[::-1]
    new = list(x)
    in_number = False
    start = 0
    for i, char in enumerate(x):
        if in_number:
            if char.isalpha() or char == ' ':
                in_number = False
                new[start:i] = new[i-1:start-1:-1]
        else:
            if not char.isalpha() and char != ' ':
                in_number = True
                start = i
    return ''.join(new)


def decompose(X, show=0, hue=None, annotations=None,perplexity=30, learning_rate=200, ee=12, metric='cosine', init='pca', title='', figsize=(20,10), transformed_X=None, return_coordinates=False, **kwargs):
    #X = normalize(np.vstack([gensim_skipgram.wv[products[x].barcode + '_child'] for x in sorted_indices]), axis=1)
    if transformed_X is None:
        pca = TSNE(n_components=2, init=init, metric=metric, perplexity=perplexity,learning_rate=learning_rate, early_exaggeration=ee, random_state=0, square_distances=True)
        transformed_X = pca.fit_transform(X)
    plt.figure(figsize=figsize)
    if show:
        transformed_X = transformed_X[:show, :]
    if hue is not None:
        p = sns.scatterplot(x=transformed_X[:,0], y=transformed_X[:,1], hue=hue, s=250, **kwargs)
    else:
        p = sns.scatterplot(x=transformed_X[:,0], y=transformed_X[:,1], s=250,**kwargs)
    if annotations is not None:
        for i,name in enumerate(annotations):
            p.annotate(name, (transformed_X[i,0], transformed_X[i,1]), fontsize=13)
    p.set_title(f"TSNE decomposition {title}")
    if return_coordinates:
        return p, transformed_X
    return p


def convert_city_name(city_name):
    bridge = {'הרצליה': 'הרצלייה',
              'יהוד-מונוסון': 'יהוד',
              'נהריה': 'נהרייה',
              'פתח תקוה': 'פתח תקווה',
              'קרית אונו': 'קריית אונו',
              'קרית אתא': 'קריית אתא',
              'קרית ביאליק': 'קריית ביאליק',
              'קרית גת': 'קריית גת',
              'קרית ים': 'קריית ים',
              'קרית מוצקין': 'קריית מוצקין',
              'קרית מלאכי': 'קריית מלאכי',
              'קרית שמונה': 'קריית שמונה',
              'סכנין': "סח'נין",
              'נצרת עילית': 'נוף הגליל',
              'מגד אל כרום': "מג'ד אל-כרום",
             'קדימה':'קדימה-צורן'
             }
    same_names = [
        ('מודיעין-מכבים-רעות*', 'מודיעין-מכבים-רעות', 'מודיען', 'מודיעין- מכבים- רעות', 'מודיעין. דירה 81',
         'מודיעין מכבים-רעות', 'מודיעין מכבים ראות', 'מודעין', 'מודיעין מכבים רעות', 'מודיעין-מכבים-רעות'),
        ('מודיעין עילית', 'מודעין עלית', 'מודיעין עלית', 'מודיעים עילית'),
        ('בנימינה-גבעת עדה*', 'בנימינה-גבעת עדה', 'בנימינה', 'בנימינה גבעת עדה'),
        ('תל אביב-יפו', 'תל אביב - יפו 6', 'תל אבי יפו', 'תל אבי', 'תל אבב', 'תל אב - יפו', 'תל אביה', 'תל-אביב -יפו',
         'תל-אביב', 'תל אביב יפו', 'תל אביב-יפו', 'תל אביב - יפו')
    ]
    for group in same_names:
        for name in group[1:]:
            bridge[name] = group[0]

    city_name = city_name.strip()
    if city_name in bridge.keys():
        return bridge[city_name]
    return city_name

def translate_district(x):
    d = {
        'צפון': 'North',
        'דרום': 'South',
        'חיפה': 'Haifa',
        'שרון-שומרון': 'Sharon - Shomron',
        'מרכז': 'Central',
        'ירושלים': 'Jerusalem',
        'דן - פ"ת': 'Dan - Petach Tikva',
        'תל אביב-יפו': 'Tel Aviv',
        'אילת': 'Eilat'
    }
    return d.get(x, 'Unknown')