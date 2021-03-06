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


def decompose(X, tsne=False, show=0, hue=None, annotations=None):
    #X = normalize(np.vstack([gensim_skipgram.wv[products[x].barcode + '_child'] for x in sorted_indices]), axis=1)
    pca = PCA(n_components=2)
    if tsne:
        pca = TSNE(n_components=2)
    transformed = pca.fit_transform(X)
    plt.figure(figsize=(20,10))
    if show:
        transformed = transformed[:show, :]
    if hue is not None:
        p = sns.scatterplot(x=transformed[:,0], y=transformed[:,1], hue=hue, s=250)
    else:
        p = sns.scatterplot(x=transformed[:,0], y=transformed[:,1], s=250)
    if annotations is not None:
        for i,name in enumerate(annotations):
            p.annotate(name, (transformed[i,0], transformed[i,1]), fontsize=13)
    p.set_title('TSNE decomposition' if tsne else 'PCA decomposition')


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
              'מגד אל כרום': "מג'ד אל-כרום"}
    same_names = [
        ('מודיעין-מכבים-רעות*', 'מודיעין-מכבים-רעות', 'מודיען', 'מודיעין- מכבים- רעות', 'מודיעין. דירה 81',
         'מודיעין מכבים-רעות', 'מודיעין מכבים ראות', 'מודעין', 'מודיעין מכבים רעות', 'מודיעין-מכבים-רעות'),
        ('מודיעין עילית', 'מודעין עלית', 'מודיעין עלית', 'מודיעים עילית'),
        ('בנימינה-גבעת עדה*', 'בנימינה-גבעת עדה', 'בנימינה', 'בנימינה גבעת עדה'),
        ('תל אביב-יפו', 'תל אביב - יפו 6', 'תל אבי יפו', 'תל אבי', 'תל אבב', 'תל אב - יפו', 'תל אביה', 'תל-אביב -יפו',
         'תל-אביב', 'תל אביב יפו', 'תל אביב-יפו')
    ]
    for group in same_names:
        for name in group[1:]:
            bridge[name] = group[0]

    city_name = city_name.strip()
    if city_name in bridge.keys():
        return bridge[city_name]
    return city_name