import torch
from torch.utils.tensorboard import SummaryWriter
from utils import use_optimizer, save_checkpoint
from sklearn.metrics import f1_score, precision_score, mean_absolute_error, r2_score
import numpy as np


class Engine(object):
    """Meta Engine for training & evaluating NCF model
    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._writer = SummaryWriter()  # tensorboard writer
        self.opt = use_optimizer(self.model, config)

        self.crit = config['crit']

    def train_single_batch(self, users, items, ratings):
        assert hasattr(self, 'model'), 'Please specify the exact model !'

        if self.config['use_cuda'] is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        self.opt.zero_grad()
        ratings_pred = self.model(users, items)
        loss = self.crit(ratings_pred, ratings)
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            user, item, rating = batch[0].long(), batch[1].long(), batch[2]
            rating = rating.float()
            loss = self.train_single_batch(user, item, rating)
            total_loss += loss
        self._writer.add_scalar('model/loss', total_loss, epoch_id)
        print('Epoch', epoch_id, 'Total Train Loss:', total_loss)

    def evaluate(self, test_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        output_list = []
        target_list = []
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                users, items, targets = batch[0].long(), batch[1].long(), batch[2]
                if self.config['use_cuda'] is True:
                    users, items, targets = users.cuda(), items.cuda(), targets.cuda()
                output = self.model(users, items)
                if self.config['use_cuda'] is True:
                    output = output.cpu()
                    targets = targets.cpu()

                output_list += list(output.view(-1).numpy())
                target_list += list(targets.view(-1).numpy())

            if self.config['implicit'] is True:
                f1 = f1_score(target_list, output_list)
                prec = precision_score(target_list, output_list)
                self._writer.add_scalar('test/F1', f1, epoch_id)
                self._writer.add_scalar('test/Prec', prec, epoch_id)
                scores = [f1, prec]
            if self.config['explicit'] is True:
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
            return scores[0]

    def save(self, model_name, epoch_id, scores):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = str(self.config['model_dir']) + model_name + "_" + str(epoch_id) + "_" + str(round(scores, 3))
        save_checkpoint(self.model, model_dir)
