import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

dataset = pd.read_csv('/Users/jackarmand/PycharmProjects/Streaming_AI/dataset.csv',index_col=0)
#vectors = dataset['vector']
print(dataset.columns)

distances = cosine_distances(vectors)
distance_dataframe = pd.DataFrame(distances,index=[str(x) for x in list(dataset['title'])],columns=[str(x) for x in list(dataset['title'])])
ids = list(distance_dataframe[862].sort_values().head(1000).index)