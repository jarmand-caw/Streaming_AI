import torch.nn as nn
import torch
from Semantic.semantic_engine import Engine
from utils import use_cuda


class GenreAttn(nn.Module):
    def __init__(self, config):

        """LSTM model with encoder used to predict genre or tag classes
            from movie overview text
        """
        super(GenreAttn, self).__init__()

        self.config = config

        # LSTM for the text overview
        self.vocab_size = config['vocab_size']
        self.n_hidden = config['n_hidden']
        self.n_out = config['n_out']
        self.weights_matrix = config['w2v_weights_matrix']

        num_embeddings, embedding_dim = self.weights_matrix.shape[0], self.weights_matrix.shape[1]
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.emb.weight.data.copy_(torch.from_numpy(self.weights_matrix))
        self.emb.weight.requires_grad = True
        self.attn = nn.Linear(self.weights_matrix.shape[1],self.weights_matrix.shape[1])
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(self.weights_matrix.shape[1], 500)
        self.linear2 = nn.Linear(500, 500)
        self.linear3 = nn.Linear(500, 250)
        self.output_fc = nn.Linear(250, self.n_out)
        self.relu = nn.ReLU()

    def forward(self, inp):
        inp = inp.long()
        embeds = self.emb(inp)
        attention = self.attn(embeds)
        attention = self.softmax(attention)
        out = attention*embeds
        out = self.relu(self.linear1(out))
        out = self.relu(self.linear2(out))
        out = self.relu(self.linear2(out))
        out = self.relu(self.linear2(out))
        out = self.relu(self.linear3(out))
        out = self.output_fc(out)

        return out


class GenreAttnEngine(Engine):
    def __init__(self, config):
        self.model = GenreAttn(config)
        if config['use_cuda'] is True:
            use_cuda(True)
            self.model.cuda()
        super(GenreAttnEngine, self).__init__(config)
        print(self.model)
