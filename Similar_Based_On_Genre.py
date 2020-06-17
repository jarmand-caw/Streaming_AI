import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import difflib
import ast

dataset = pd.read_csv('/Users/jackarmand/PycharmProjects/Streaming_AI/dataset.csv',index_col=0)

dataset = dataset.loc[dataset['popularity']>15]
vectors = []
for x in list(dataset['vector']):
    l = ast.literal_eval(x)
    vectors.append(l)
vectors = np.array(vectors)

distances = cosine_distances(vectors)
distance_dataframe = pd.DataFrame(distances,index=[str(x).lower() for x in list(dataset['title'])],columns=[str(x).lower() for x in list(dataset['title'])])

title = input('Enter Movie Name: ')
title = title.lower().strip()
def get_movie(title):
    ret = False
    out = 'No movies found!'
    while ret==False:
        try:
            results = distance_dataframe.loc[title].sort_values().iloc[1:11]
            out = list(results.index)
            ret = True
        except KeyError:
            close = difflib.get_close_matches(title,list(distance_dataframe.index))
            print('Did you mean one of these?',close)
            title = input('If so, copy and paste name from above and enter. If not, enter "no": ' )
            title = title.lower().strip()
            if title=='no':
                ret==True
    return out

out = get_movie(title)
if type(out)==list:
    print('')
    print(*out,sep='\n')
else:
    print('')
    print(out)