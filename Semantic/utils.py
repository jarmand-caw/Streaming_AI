import torch
from gensim.models import KeyedVectors
import numpy as np
import ast
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
import re
import string
from nltk import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer


# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir):
    state_dict = torch.load(model_dir)  # ensure all storage are on gpu
    model.load_state_dict(state_dict)


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(),
                                     lr=params['adam_lr'],
                                     weight_decay=params['l2_regularization'])

    return optimizer


class PreprocessDataFrame:
    def __init__(self, df):
        assert 'overview', 'genres' in list(df.columns)
        self.df = df
        nltk.download('stopwords')
        self.stop = set(stopwords.words('english'))
        self.punctuation = string.punctuation
        nltk.download('punkt')
        nltk.download('wordnet')

    def get_genres(self, string):
        invalid_genres = ['Aniplex', 'BROSTA TV', 'Carousel Productions', 'GoHands',
                          'Mardock Scramble Production Committee', 'Odyssey Media', 'Pulser Productions', 'Rogue State',
                          'Sentai Filmworks', 'Telescene Film Group Productions', 'The Cartel',
                          'Vision View Entertainment', 'TV Movie', 'Foreign']
        l = ast.literal_eval(string)
        names = []
        for d in l:
            name = d['name']
            if name not in invalid_genres:
                names.append(d['name'])
        return names

    def clean_text(self, text):
        punctuation = string.punctuation
        lemmatizer = WordNetLemmatizer()
        text = text.translate(str.maketrans('', '', punctuation))
        text = text.lower().strip()
        text = ' '.join([i if i not in self.stop and i.isalpha() else '' for i in text.lower().split()])
        text = ' '.join([lemmatizer.lemmatize(w) for w in word_tokenize(text)])
        text = re.sub(r"\s{2,}", " ", text)
        return text

    def sequence_ify(self):
        MAX_NB_WORDS = 50000
        MAX_SEQUENCE_LENGTH = self.df['overview'].map(len).max()
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True)
        tokenizer.fit_on_texts(self.df['overview'].values)
        word_index = tokenizer.word_index
        X = tokenizer.texts_to_sequences(self.df['overview'].values)
        X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
        self.X = X
        self.word_index = word_index

    def multi_label_binarizer(self):
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(list(self.df['genres']))
        self.y = y

    def get_embedding_matrix(self, file_dir):
        word2vecDict = KeyedVectors.load_word2vec_format(
            file_dir, binary=True)
        embed_size = 300

        embeddings_index = dict()
        for word in word2vecDict.wv.vocab:
            embeddings_index[word] = word2vecDict.word_vec(word)
        print("Loaded " + str(len(embeddings_index)) + " word vectors.")

        embedding_matrix = 1 * np.random.randn(len(self.word_index) + 1, embed_size)

        embeddedCount = 0
        for word, i in self.word_index.items():
            i -= 1
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                embeddedCount += 1
        print("total embedded:", embeddedCount, "common words")

        del (embeddings_index)

        self.w2v_embedding = embedding_matrix

    def process(self, embedding_file_dir):
        self.df['genres'] = self.df['genres'].apply(lambda x: self.get_genres(x))
        self.df = self.df.loc[~(self.df['genres'].str.len() == 0)]
        self.df = self.df[['id', 'title', 'overview', 'genres']]
        self.df = self.df.dropna()
        self.df['overview'] = self.df['overview'].apply(lambda x: self.clean_text(x))
        self.sequence_ify()
        self.multi_label_binarizer()
        self.get_embedding_matrix(embedding_file_dir)

        return self.X, self.y

def convert_user_to_tensors(user_df, movie_tensor_df, user_movie_df, rat=True):
    """Function for converting users into tensors of each movie they watch
    Note: user_df is a dataframe indexed by user ID that contains columns with lists of movies a user has watched and ratings for each movie
    Note: movie_tensor_df is a pandas dataframe indexed by movie ID that has a column named 'tensor'
    Note: user_movie_df is a the dataframe containing each user and movie"""
    user_tensors = []
    movie_tensors = []
    for idx in range(len(user_movie_df)):
        user = user_movie_df.iloc[idx]['userId']
        movie = user_movie_df.iloc[idx]['movieId']
        movies_watched = list(user_df.loc[user]['movieId'])  # get the list of movies the user has watched
        if rat:
            ratings = list(user_df.loc[user]['ratings'])
            ratings_dict = dict(zip(movies_watched, ratings))

        movies_watched_not_including = list(set(movies_watched) - set([movie]))
        if rat:
            ratings = [ratings_dict[x] for x in movies_watched_not_including]
        try:
            utens = list(movie_tensor_df.loc[movies_watched_not_including]['tensors'])
        except:
            print(movies_watched_not_including)
            print(idx)
            return
        utens = [x.unsqueeze(0) for x in utens]
        user_tensor = torch.cat(utens,
                                dim=0)  # get the vectors for each movie and concat them
        if rat:
            ratings = torch.Tensor(ratings)
            try:
                user_tensor = (user_tensor.t() * ratings).t()
            except:
                return user_tensor,ratings,movies_watched_not_including
            user_tensors.append(list(user_tensor.numpy()))
        else:
            user_tensors.append(list(user_tensor.numpy()))
        movie_tensor = movie_tensor_df.loc[movie]['tensors']
        movie_tensors.append(list(movie_tensor.numpy()))
    user_movie_df['user_tensors'] = user_tensors
    user_movie_df['movie_tensors'] = movie_tensors
    return user_movie_df
