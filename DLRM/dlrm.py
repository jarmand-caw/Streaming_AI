import torch
from DLRM.dlrm_engine import Engine
from utils import use_cuda


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
        self.embedding_genre = torch.nn.Embedding(num_embeddings=self.num_genres, embedding_dim=self.latent_dim_genre)
        self.embedding_cast = torch.nn.Embedding(num_embeddings=self.num_cast, embedding_dim=self.latent_dim_cast)
        self.embedding_comp = torch.nn.Embedding(num_embeddings=self.num_comp, embedding_dim=self.latent_dim_comp)



        self.cont_mlp = torch.nn.ModuleList()
        for idx in range(len(config['cont_layers'])-1):
            in_size = self.cont_layers[idx]
            out_size = self.cont_layers[idx+1]
            self.cont_mlp.append(torch.nn.Linear(in_size, out_size))

        self.embedding_mlp = torch.nn.ModuleList()
        self.embedding_mlp.append(torch.nn.Linear(self.latent_dim_cast*3+self.latent_dim_comp+self.latent_dim_genre*3,
                                                  self.cat_layers[0]))
        for idx in range(len(config['cat_layers'])-1):
            in_size = self.cat_layers[idx]
            out_size = self.cat_layers[idx+1]
            self.embedding_mlp.append(torch.nn.Linear(in_size, out_size))

        self.movie_output_mlp = torch.nn.ModuleList()
        self.movie_output_mlp.append(torch.nn.Linear(self.cat_layers[-1]+self.cont_layers[-1], self.movie_output_layers[0]))
        for idx in range(len(config['movie_output_layers']) - 1):
            in_size = self.movie_output_layers[idx]
            out_size = self.movie_output_layers[idx + 1]
            self.movie_output_mlp.append(torch.nn.Linear(in_size, out_size))

        self.bottom_mlp = torch.nn.ModuleList()
        self.bottom_mlp.append(torch.nn.Linear(self.movie_output_layers[-1]+self.latent_dim_user, self.bottom_mlp_layers[0]))
        for idx in range(len(self.bottom_mlp_layers)-1):
            in_size = self.bottom_mlp_layers[idx]
            out_size = self.bottom_mlp_layers[idx+1]
            self.bottom_mlp.append(torch.nn.Linear(in_size,out_size))

        self.relu = torch.nn.ReLU()
        self.affine_output = torch.nn.Linear(self.bottom_mlp_layers[-1],1)

        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, cont, genres, cast, comp):
        batch_size = user_indices.shape[0]
        user_embedding = self.embedding_user(user_indices)


        comp_embedding = self.embedding_comp(comp).view(batch_size,-1)
        genre_embedding = self.embedding_genre(genres).view(batch_size,-1)
        cast_embedding = self.embedding_cast(cast).view(batch_size,-1)

        cat_vector = torch.cat([comp_embedding,genre_embedding,cast_embedding],dim=-1)
        for idx in range(len(self.embedding_mlp)):
            cat_vector = self.embedding_mlp[idx](cat_vector)
            cat_vector = self.relu(cat_vector)

        for idx in range(len(self.cont_mlp)):
            cont = self.cont_mlp[idx](cont)
            cont = self.relu(cont)

        movie_vector = torch.cat([cat_vector,cont],dim=-1)
        for idx in range(len(self.movie_output_mlp)):
            movie_vector = self.movie_output_mlp[idx](movie_vector)
            movie_vector = self.relu(movie_vector)

        vector = torch.cat([user_embedding,movie_vector],dim=-1)
        for idx in range(len(self.bottom_mlp)):
            vector = self.bottom_mlp[idx](vector)
            vector = self.relu(vector)

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