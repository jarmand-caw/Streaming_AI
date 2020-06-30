import torch
from DLRM.dlrm_engine import Engine
from Benchmark.utils import use_cuda


class DLRM(torch.nn.Module):
    def __init__(self, config):
        super(DLRM, self).__init__()
        self.config = config
        self.cat_layers = config['cat_layers']
        self.cont_layers = config['cont_layers']
        self.movie_output_layers = config['movie_output_layers']
        self.bottom_mlp_layers = config['bottom_mlp_layers']

        self.num_users = config['num_users']
        self.num_genres = config['num_genre']
        self.num_cast = config['num_cast']
        self.num_comp = config['num_companies']

        self.latent_dim_user = config['latent_dim_user']
        self.latent_dim_genre = config['latent_dim_genre']
        self.latent_dim_cast = config['latent_dim_cast']
        self.latent_dim_comp = config['latent_dim_comp']

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_user)
        self.embedding_genre = torch.nn.EmbeddingBag(num_embeddings=self.num_genres,
                                                     embedding_dim=self.latent_dim_genre, mode='sum')
        self.embedding_cast = torch.nn.EmbeddingBag(num_embeddings=self.num_cast, embedding_dim=self.latent_dim_cast,
                                                    mode='sum')
        self.embedding_comp = torch.nn.Embedding(num_embeddings=self.num_comp, embedding_dim=self.latent_dim_comp)

        self.cont_mlp = torch.nn.ModuleList()
        for idx in range(len(config['cont_layers']) - 1):
            in_size = self.cont_layers[idx]
            out_size = self.cont_layers[idx + 1]
            self.cont_mlp.append(torch.nn.Linear(in_size, out_size))
            self.cont_mlp.append(torch.nn.ReLU())
            self.cont_mlp.append(torch.nn.BatchNorm1d(out_size))
        self.cont_mlp_seq = torch.nn.Sequential(*self.cont_mlp)

        self.embedding_mlp = torch.nn.ModuleList()
        self.embedding_mlp.append(torch.nn.Linear(self.latent_dim_cast + self.latent_dim_comp + self.latent_dim_genre,
                                                  self.cat_layers[0]))
        self.embedding_mlp.append(torch.nn.ReLU())
        self.embedding_mlp.append(torch.nn.BatchNorm1d(self.cat_layers[0]))
        for idx in range(len(config['cat_layers']) - 1):
            in_size = self.cat_layers[idx]
            out_size = self.cat_layers[idx + 1]
            self.embedding_mlp.append(torch.nn.Linear(in_size, out_size))
        self.embedding_mlp_seq = torch.nn.Sequential(*self.embedding_mlp)

        self.movie_output_mlp = torch.nn.ModuleList()
        self.movie_output_mlp.append(
            torch.nn.Linear(self.latent_dim_cast + self.latent_dim_comp + self.latent_dim_genre + self.cont_layers[-1],
                            self.movie_output_layers[0]))
        # self.movie_output_mlp.append(torch.nn.Linear(self.cont_layers[-1]+self.cat_layers[-1], self.movie_output_layers[0]))
        self.movie_output_mlp.append(torch.nn.ReLU())
        self.movie_output_mlp.append(torch.nn.BatchNorm1d(self.movie_output_layers[0]))
        for idx in range(len(config['movie_output_layers']) - 1):
            in_size = self.movie_output_layers[idx]
            out_size = self.movie_output_layers[idx + 1]
            self.movie_output_mlp.append(torch.nn.Linear(in_size, out_size))
            self.movie_output_mlp.append(torch.nn.ReLU())
            self.movie_output_mlp.append(torch.nn.BatchNorm1d(out_size))
        self.movie_output_mlp_seq = torch.nn.Sequential(*self.movie_output_mlp)

        self.bottom_mlp = torch.nn.ModuleList()
        self.bottom_mlp.append(
            torch.nn.Linear(self.movie_output_layers[-1] + self.latent_dim_user, self.bottom_mlp_layers[0]))
        self.bottom_mlp.append(torch.nn.ReLU())
        self.bottom_mlp.append(torch.nn.BatchNorm1d(self.bottom_mlp_layers[0]))
        for idx in range(len(self.bottom_mlp_layers) - 1):
            in_size = self.bottom_mlp_layers[idx]
            out_size = self.bottom_mlp_layers[idx + 1]
            self.bottom_mlp.append(torch.nn.Linear(in_size, out_size))
            self.bottom_mlp.append(torch.nn.ReLU())
            self.bottom_mlp.append(torch.nn.BatchNorm1d(out_size))
        self.bottom_mlp_seq = torch.nn.Sequential(*self.bottom_mlp)

        self.relu = torch.nn.ReLU()
        self.affine_output = torch.nn.Linear(self.bottom_mlp_layers[-1], 1)

        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, cont, genres, cast, comp):
        batch_size = user_indices.shape[0]
        user_embedding = self.embedding_user(user_indices)

        comp_embedding = self.embedding_comp(comp).view(batch_size, -1)
        genre_embedding = self.embedding_genre(genres).view(batch_size, -1)
        cast_embedding = self.embedding_cast(cast).view(batch_size, -1)

        cat_vector = torch.cat([comp_embedding, genre_embedding, cast_embedding], dim=-1)
        # cat_vector = self.embedding_mlp_seq(cat_vector)

        cont = self.cont_mlp_seq(cont)

        movie_vector = torch.cat([cat_vector, cont], dim=-1)

        movie_vector = self.movie_output_mlp_seq(movie_vector)

        vector = torch.cat([user_embedding, movie_vector], dim=-1)

        vector = self.bottom_mlp_seq(vector)

        vector = self.affine_output(vector)
        out = self.logistic(vector)
        return out.view(-1)

    def init_weight(self):
        pass


class DLRMEngine(Engine):
    """Engine for training & evaluating GMF model"""

    def __init__(self, config):
        self.model = DLRM(config)
        if config['use_cuda'] is True:
            use_cuda(True)
            self.model.cuda()
        super(DLRMEngine, self).__init__(config)
        print(self.model)

        if config['pretrain']:
            self.model.load_pretrain_weights()