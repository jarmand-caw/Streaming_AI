import torch.nn as nn
import torch
from Semantic.semantic_engine import Engine
from utils import use_cuda


class CFNet(nn.Module):
    def __init__(self, config):

        """LSTM model with encoder used to predict genre or tag classes
            from movie overview text
        """
        super(CFNet, self).__init__()

        self.config = config
        self.vector_dict = config['vector_dict']
        self.vector_length = config['vector_length']
        self.hidden_size = config['hidden_size']
        self.n_layers = config['n_layers']
        self.fixed_size = config['fixed_size']
        self.lstm = nn.LSTM(self.vector_length, self.hidden_size, self.n_layers, batch_first=True)


        # LSTM for the text overview
        self.dropout = nn.Dropout(0.1)
        self.user_embedding = nn.Linear(self.hidden_size, self.fixed_size)
        self.item_embedding = nn.Linear(self.vector_length,self.fixed_size)
        self.linear1 = nn.Linear(self.fixed_size*2, 200)
        self.linear2 = nn.Linear(200, 200)
        self.linear3 = nn.Linear(200, 100)
        self.output_fc = nn.Linear(100, self.n_out)
        self.relu = nn.ReLU()

    def forward(self, lstm_inp, movie_vector):
        #lstm_inp is a vector size (batch_size, num_movies, movie_vector_len) that describes all the movies a user watched
        batch_size = lstm_inp.size(0)
        hidden = self.init_hidden(batch_size)
        lstm_inp = lstm_inp.long()
        lstm_out, hidden = self.lstm(lstm_inp, hidden)
        lstm_out = self.dropout(lstm_out[:, -1])  # last column of LSTM is one we want since this is many to one
        user = self.user_embedding(lstm_out)
        item = self.item_embedding(movie_vector)

        out = torch.cat(user, item)

        out = self.relu(self.linear1(out))
        out = self.relu(self.linear2(out))
        out = self.relu(self.linear2(out))
        out = self.relu(self.linear2(out))

        out = self.relu(self.linear3(out))
        out = self.output_fc(out)

        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.config['use_cuda'] is True:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_size).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_size).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_size).zero_())
        return hidden


class CFEngine(Engine):
    def __init__(self, config):
        self.model = CFNet(config)
        if config['use_cuda'] is True:
            use_cuda(True)
            self.model.cuda()
        super(CFEngine, self).__init__(config)
        print(self.model)