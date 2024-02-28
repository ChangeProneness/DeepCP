import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.train_loss_min = np.Inf

    def __call__(self, train_loss, model):

        score = train_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(train_loss, model)
        elif (score >= self.best_score):  # <
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(train_loss, model)
            self.counter = 0

    def save_checkpoint(self, train_loss, model):
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.train_loss_min = train_loss
