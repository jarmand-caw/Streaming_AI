import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from DLRM.dlrm import DLRMEngine
import torch
import torch.nn as nn
import copy
from DLRM.prep import *


META_DATA_PATH = ''
CREDITS_PATH = ''
LINK_PATH = ''
RATINGS_PATH = ''
df = pd.read_csv(META_DATA_PATH)
cred = pd.read_csv(CREDITS_PATH)


df['id'] = df['id'].apply(lambda x: intify(x))
df = df.loc[~df['id'].isna()]

cred['cast_names'] = cred['cast'].apply(lambda x: cast_list(x))

data = pd.merge(df,cred,on='id')

data['cast_1'] = data['cast_names'].apply(lambda x: x[0])
data['cast_2'] = data['cast_names'].apply(lambda x: x[1])
data['cast_3'] = data['cast_names'].apply(lambda x: x[2])

actors = set(list(data['cast_1'])+list(data['cast_2'])+list(data['cast_3']))
actors = list(actors)
actors.remove('None')
actors.insert(0,'None')

actor_id_map = dict(zip(actors,list(np.arange(len(actors)))))

data['cast_1'] = data['cast_1'].map(actor_id_map)
data['cast_2'] = data['cast_2'].map(actor_id_map)
data['cast_3'] = data['cast_3'].map(actor_id_map)




data['popularity'] = data['popularity'].apply(floatify)


data['genre_names'] = data['genres'].apply(lambda x: get_genres(x))

data['genre_1'] = data['genre_names'].apply(lambda x: x[0])
data['genre_2'] = data['genre_names'].apply(lambda x: x[1])
data['genre_3'] = data['genre_names'].apply(lambda x: x[2])

genres = set(list(data['genre_1']) + list(data['genre_2']) + list(data['genre_3']))

genres = list(genres)
genres.remove('None')
genres.insert(0, 'None')

genre_id_map = dict(zip(genres, list(np.arange(len(genres)))))

data['genre_1'] = data['genre_1'].map(genre_id_map)
data['genre_2'] = data['genre_2'].map(genre_id_map)
data['genre_3'] = data['genre_3'].map(genre_id_map)



data['production_company'] = data['production_companies'].apply(get_production)
comps = set(list(data['production_company']))
comps = list(comps)
comps.remove('None')
comps.insert(0,'None')

comp_id_map = dict(zip(comps,list(np.arange(len(comps)))))
data['production_company'] = data['production_company'].map(comp_id_map)

continuous = ['vote_count','vote_average','revenue','popularity','budget']
categorial = ['genre_1','genre_2','genre_3','cast_1','cast_2','cast_3','production_company']
data_to_load = data[continuous+categorial+['title','id']].dropna(how='any')
data_to_load = data_to_load.groupby('title').first().reset_index()

links = pd.read_csv(LINK_PATH)
id_map = dict(zip(list(links['movieId']),list(links['tmdbId'])))

user_movie_df = pd.read_csv(RATINGS_PATH)
user_movie_df['movieId'] = user_movie_df['movieId'].map(id_map)
user_movie_df['userId'] = user_movie_df['userId'].astype(int).map(dict(zip(list(set(user_movie_df['userId'])),list(np.arange(len(list(set(user_movie_df['userId']))))))))
user_movie_df.dropna(inplace=True)
user_movie_df.rename(columns={'movieId':'id'},inplace=True)

ratings_dataframe = pd.merge(user_movie_df,data_to_load,on='id')

scaler = MinMaxScaler()
ratings_dataframe['min_max_rating'] = scaler.fit_transform(ratings_dataframe['rating'].values.reshape(-1,1))
ratings_dataframe['budget'] = ratings_dataframe['budget'].astype(int)

train,test = train_test_split(ratings_dataframe,test_size=.1,random_state=1)

train_dataset = EmbedDataset(*get_numpy(train))
test_dataset = EmbedDataset(*get_numpy(test))

train_loader = DataLoader(train_dataset,batch_size=256,shuffle=True,num_workers=1)
test_loader = DataLoader(test_dataset,batch_size=256,num_workers=1)



config = {
    'cat_layers':[128],
    'cont_layers':[5,256,512,256,128,64],
    'movie_output_layers':[256,256,128],
    'bottom_mlp_layers':[512,256,128],
    'num_users':len(set(user_movie_df['userId']))+1,
    'num_cast':len(actors),
    'num_genre':len(genres),
    'num_companies':len(comps),
    'latent_dim_user':100,
    'latent_dim_genre':20,
    'latent_dim_cast':20,
    'latent_dim_comp':20,
          'use_cuda':True,
          'model_dir':'/content/drive/My Drive/Movie_Categorization/Models/',
          'optimizer':'adam','adam_lr':1e-2,'l2_regularization':0,
          'model_name':'DLRM_HUGE','crit':nn.MSELoss(),'implicit':False,'explicit':True,'pretrain':False}

torch.manual_seed(0)
epochs = 150
engine = DLRMEngine(config)
best_score=1e10
count = 0
patience=5
for epoch in range(epochs):
    engine.train_an_epoch(train_loader,epoch)
    score = engine.evaluate(test_loader,epoch)
    if score<best_score:
        best_model = copy.deepcopy(engine.model.state_dict())
        best_score=score
    else:
        count+=1
    if count>patience:
        break
model_name = 'DLRM_HUGE'
model_dir = str(config['model_dir']) + model_name + "_" + str(round(best_score, 3)) + '.pt'
torch.save(best_model,model_dir)