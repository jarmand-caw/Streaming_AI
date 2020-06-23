import torch.nn as nn
import torch
from Semantic.semantic_engine import Engine
from utils import use_cuda


class GenreNet(nn.Module):
    def __init__(self, config):

        """LSTM model with encoder used to predict genre or tag classes
            from movie overview text
        """
        super(GenreNet, self).__init__()

        self.config = config

        # LSTM for the text overview
        self.vocab_size = config['vocab_size']
        self.n_hidden = config['n_hidden']
        self.n_out = config['n_out']
        self.n_layers = config['n_layers']
        self.weights_matrix = config['w2v_weights_matrix']

        self.num_embeddings, self.embedding_dim = self.weights_matrix.shape[0], self.weights_matrix.shape[1]
        self.emb = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.emb.weight.data.copy_(torch.from_numpy(self.weights_matrix))
        self.emb.weight.requires_grad = True
        self.lstm = nn.LSTM(self.embedding_dim, self.n_hidden, self.n_layers, dropout=0.2, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.lstm_fc = nn.Linear(self.embedding_dim, 500)
        self.linear1 = nn.Linear(500, 500)
        self.linear2 = nn.Linear(500, 500)
        self.linear3 = nn.Linear(500, 250)
        self.output_fc = nn.Linear(250, self.n_out)
        self.relu = nn.ReLU()

    def forward(self, lstm_inp):
        batch_size = lstm_inp.size(0)
        hidden = self.init_hidden(batch_size)
        lstm_inp = lstm_inp.long()
        embeds = self.emb(lstm_inp)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = self.dropout(lstm_out[:, -1])  # last column of LSTM is one we want since this is many to one
        lstm_out = self.relu(self.lstm_fc(lstm_out))

        out = self.relu(self.linear1(lstm_out))
        out = self.relu(self.linear2(out))
        out = self.relu(self.linear2(out))
        out = self.relu(self.linear2(out))

        out = self.relu(self.linear3(out))
        out = self.output_fc(out)

        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.config['use_cuda'] is True:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        return hidden


class GenreEngine(Engine):
    def __init__(self, config):
        self.model = GenreNet(config)
        if config['use_cuda'] is True:
            use_cuda(True)
            self.model.cuda()
        super(GenreEngine, self).__init__(config)
        print(self.model)
