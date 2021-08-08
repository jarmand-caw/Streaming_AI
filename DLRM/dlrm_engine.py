import torch
from torch.utils.tensorboard import SummaryWriter
from Benchmark.utils import use_optimizer, save_checkpoint
from sklearn.metrics import f1_score, precision_score, mean_absolute_error, r2_score
import numpy as np


class Engine(object):
    """Meta Engine for training & evaluating DLRM model
    Note: Subclass should implement self.model !
    Note: MinMaxScaler should be defined and fit on ratings 0-5
    """

    def __init__(self, config):
        self.config = config  # model configuration dictionary
        self._writer = SummaryWriter()  # tensorboard writer
        self.opt = use_optimizer(self.model, config)

        self.crit = config['crit']

    def train_single_batch(self, user, cont, genres, cast, comp, rating):
        assert hasattr(self, 'model'), 'Please specify the exact model !'

        if self.config['use_cuda'] is True:
            user, cont, genres, cast, comp, rating = user.cuda(), cont.cuda(), genres.cuda(), cast.cuda(), comp.cuda(), rating.cuda()
        self.opt.zero_grad()
        ratings_pred = self.model(user, cont, genres, cast, comp)
        loss = self.crit(ratings_pred, rating)
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            user = batch[0].long()
            cont = batch[1].float()
            genres = batch[2].long()
            cast = batch[3].long()
            comp = batch[4].long()
            rating = batch[5].float()

            loss = self.train_single_batch(user, cont, genres, cast, comp, rating)
            total_loss += loss
        self._writer.add_scalar('model/loss', total_loss, epoch_id)
        print('Epoch', epoch_id, 'Total Train Loss:', total_loss)

    def evaluate(self, test_loader, epoch_id, scaler=None):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        output_list = []
        target_list = []
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                user = batch[0].long()
                cont = batch[1].float()
                genres = batch[2].long()
                cast = batch[3].long()
                comp = batch[4].long()
                rating = batch[5].float()
                if self.config['use_cuda'] is True:
                    user, cont, genres, cast, comp, rating = user.cuda(), cont.cuda(), genres.cuda(), cast.cuda(), comp.cuda(), rating.cuda()
                output = self.model(user, cont, genres, cast, comp)
                if self.config['use_cuda'] is True:
                    output = output.cpu()
                    rating = rating.cpu()

                output_list += list(output.view(-1).numpy())
                target_list += list(rating.view(-1).numpy())

            if self.config['implicit'] is True:
                f1 = f1_score(target_list, output_list)
                prec = precision_score(target_list, output_list)
                self._writer.add_scalar('test/F1', f1, epoch_id)
                self._writer.add_scalar('test/Prec', prec, epoch_id)
                scores = [f1, prec]
            if self.config['explicit'] is True:
                target_list = scaler.inverse_transform(np.array(target_list).reshape(-1, 1))
                output_list = scaler.inverse_transform(np.array(output_list).reshape(-1, 1))
                mae = mean_absolute_error(target_list, output_list)
                r2 = r2_score(target_list, output_list)
                scores = [mae, r2]
                self._writer.add_scalar('test/MAE', mae, epoch_id)
                self._writer.add_scalar('test/R2', r2, epoch_id)
            print('Evaluating Epoch', epoch_id, '...')
            if self.config['implicit'] is True:
                print('F1:', f1, 'Prec:', prec)
            if self.config['explicit'] is True:
                print('MAE:', mae, 'R2:', r2)
                print('')
            return scores[0]

    def save(self, model_name, epoch_id, scores):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = str(self.config['model_dir']) + model_name + "_" + str(epoch_id) + "_" + str(
            round(scores, 3)) + '.pt'
        save_checkpoint(self.model, model_dir)