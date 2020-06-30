import ast
from torch.utils.data import Dataset

def intify(x):
    try:
        return int(x)
    except:
        return float('nan')

def cast_list(l):
    l = ast.literal_eval(l)
    names = []
    for idx in range(3):
        try:
            d = l[idx]
            name = d['name']
            names.append(name)
        except:
            names.append('None')
    return names

def floatify(x):
    try:
        return float(x)
    except:
        return float('nan')

def get_genres(l):
    names = []
    l = ast.literal_eval(l)
    for idx in range(3):
        try:
            d = l[idx]
            name = d['name']
            names.append(name)
        except:
            names.append('None')
    return names

def get_production(l):
    try:
        l = ast.literal_eval(l)
    except:
        return 'None'
    try:
        d = l[0]
        name = d['name']
        return name
    except:
        return 'None'

class EmbedDataset(Dataset):
    def __init__(self, user_array, genre_array, cast_array, comp_array, cont_array, rating_array):
        self.user = user_array
        self.genre = genre_array
        self.cast = cast_array
        self.comp = comp_array
        self.cont = cont_array
        self.rating = rating_array

    def __len__(self):
        return len(self.comp)

    def __getitem__(self, idx):
        cont = self.cont[idx]

        genres = self.genre[idx]

        cast = self.cast[idx]

        comp = self.comp[idx]

        user = self.user[idx]

        y = self.rating[idx]

        return user, cont, genres, cast, comp, y

def get_numpy(df):
    genre_cols = ['genre_1','genre_2','genre_3']
    cast_cols = ['cast_1','cast_2','cast_3']
    comp_cols = ['production_company']
    cont_cols = ['vote_count','vote_average','revenue','popularity','budget']

    genre = df[genre_cols].values
    cast = df[cast_cols].values
    comp = df[comp_cols].values
    cont = df[cont_cols].values
    user = df['userId'].values
    rating = df['min_max_rating'].values

    return user,genre,cast,comp,cont,rating