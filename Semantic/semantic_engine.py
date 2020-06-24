import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils import use_optimizer, save_checkpoint
from sklearn.metrics import f1_score, precision_score
import numpy as np

class Engine(object):
    """Meta Engine for training & evaluating LSTM model
    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._writer = SummaryWriter()  # tensorboard writer
        if config['lstm'] is True:
            self.clip = self.config['clip']
        self.opt = use_optimizer(self.model, config)
        self.crit = torch.nn.BCEWithLogitsLoss()

    def train_single_batch(self, text, labels):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        if self.config['use_cuda'] is True:
            text, labels = text.cuda(), labels.cuda()
        self.opt.zero_grad()
        pred = self.model(text)
        loss = self.crit(pred, labels)
        loss.backward()
        if self.config['lstm'] is True:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.opt.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.train()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor), 'Train loader must contain type torch.LongTensor'
            text, labels = batch[0], batch[1]
            labels = labels.float()
            loss = self.train_single_batch(text, labels)
            total_loss += loss
        average_loss = total_loss/(batch_id+1)
        self._writer.add_scalar('model/loss', average_loss, epoch_id)
        print('Epoch', epoch_id, 'Average Train Loss:', average_loss)

    def evaluate(self, test_loader, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            for i, batch in enumerate(test_loader):
                text, labels = batch[0], batch[1]
                if self.config['use_cuda'] is True:
                    text, labels = text.cuda(), labels.cuda()
                output = self.model(text)
                loss = self.crit(output,labels.float()).item()
                total_loss+=loss
                if self.config['use_cuda'] is True:
                    output = output.cpu()
                    labels = labels.cpu()
                output = torch.sigmoid(output).numpy()>.5
                labels = labels.numpy()
                if i==0:
                    output_list = output
                    target_list = labels
                else:
                    np.vstack((output_list,output))
                    np.vstack((target_list,labels))
            f1 = f1_score(target_list, output_list, average='micro')
            prec = precision_score(target_list, output_list, average='micro')
            average_loss = total_loss/(i+1)
            self._writer.add_scalar('test/F1', f1, epoch_id)
            self._writer.add_scalar('test/Prec', prec, epoch_id)
            self._writer.add_scalar('test/Loss', average_loss, epoch_id)
            scores = [f1, prec, average_loss]

            print('Evaluating Epoch', epoch_id, '...')
            print('F1:', f1, 'Prec:', prec, 'Loss:',average_loss)
            return scores[-1]

    def save(self, model_name, epoch_id, scores):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = str(self.config['model_dir']) + model_name + "_" + str(epoch_id) + "_" + str(round(scores[0], 3))
        save_checkpoint(self.model, model_dir)