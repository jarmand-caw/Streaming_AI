from engine import Engine
import torch
from matrix_factorization import GMF
# from engine import Engine
from utils import use_cuda, resume_checkpoint


class MF_MLP(torch.nn.Module):
    def __init__(self, config):
        super(MF_MLP, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.mf_embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.mf_embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.fc_layers = torch.nn.ModuleList()
        for idx in range(len(config['layers']) - 1):
            layer = config['layers'][idx]
            in_size = layer[0]
            out_size = layer[1]
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.mf_layers = torch.nn.ModuleList()
        for idx in range(len(config['mf_layers']) - 1):
            layer = config['mf_layers'][idx]
            in_size = layer[0]
            out_size = layer[1]
            self.mf_layers.append(torch.nn.Linear(in_size, out_size))

        self.relu = torch.nn.ReLU()
        self.affine_output = torch.nn.Linear(in_features=config['layers'][-1][0] + config['mf_layers'][-1][0],
                                             out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector
        for idx in range(len(self.fc_layers)):
            vector = self.fc_layers[idx](vector)
            vector = self.relu(vector)
            # vector = torch.nn.BatchNorm1d()(vector)
            # vector = torch.nn.Dropout(p=0.5)(vector)

        mf_user_embedding = self.mf_embedding_user(user_indices)
        mf_item_embedding = self.mf_embedding_item(item_indices)
        mf_vector = user_embedding * item_embedding
        for idx in range(len(self.mf_layers)):
            mf_vector = self.mf_layers[idx](mf_vector)
            mf_vector = self.relu(mf_vector)

        logits = self.affine_output(torch.cat([vector, mf_vector], dim=-1))
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        """Loading weights from trained GMF model"""
        config = self.config
        gmf_model = GMF(config)
        if config['use_cuda'] is True:
            gmf_model.cuda()
        resume_checkpoint(gmf_model, model_dir=config['pretrain_mf'], device_id=config['device_id'])
        self.embedding_user.weight.data = gmf_model.embedding_user.weight.data
        self.embedding_item.weight.data = gmf_model.embedding_item.weight.data


class MF_MLPEngine(Engine):
    """Engine for training & evaluating GMF model"""

    def __init__(self, config):
        self.model = MF_MLP(config)
        if config['use_cuda'] is True:
            use_cuda(True)
            self.model.cuda()
        super(MF_MLPEngine, self).__init__(config)
        print(self.model)

        if config['pretrain']:
            self.model.load_pretrain_weights()