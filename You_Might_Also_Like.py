import pandas as pd
import numpy as np
popular_movies = []#enter movie names here
print("Hi! We need you to rate a few movies for us. Seen any of these?")
print("If you haven't seen a movie, just hit enter")
print("If you liked the movie, enter 1")
print("If you didn't like the movie, enter 0")

subset = np.random.choice(popular_movies,len(popular_movies),replace=False)
start = 0
end = 20
respones = []
while count<10:
    small_set = subset[start:end]
    for movie in small_set:
        response = input(movie)
        responses.append(response)
    count = 0
    no_none = [x for x in responses if x is not None]
    count = len(no_none)
    start+=20
    end+=20

movies = subset[:len(responses)]

indexer = pd.read_csv('...',index_col=0)
indexer
