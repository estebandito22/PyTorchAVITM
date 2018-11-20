"""Class to train AVITM models."""

import os
import multiprocessing as mp

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from avitm.decoder_network import DecoderNetwork


class AVITM(object):

    """Class to train AVITM model."""

    def __init__(self, input_size, n_components=10, model_type='prodLDA',
                 hidden_sizes=(100,), activation='softplus', batch_size=64,
                 lr=1e-3, momentum=0.99, solver='adam', num_epochs=100,
                 reduce_on_plateau=True, weight_decay=0.0):
        """
        Initialize AVITM model.

        Args
            input_size : int, dimension of input
            n_components : int, number of topic components, (default 10)
            model_type : string, 'prodLDA' or 'LDA' (default 'prodLDA')
            hidden_sizes : tuple, length = n_layers - 2, (default (100, ))
            activation : string, 'softplus', 'relu', (default 'softplus')
            batch_size : int, size of batch to use for training (default 64)
            lr : float, learning rate to use for training (default 1e-3)
            momentum : float, momentum to use for training (default 0.99)
            solver : string, optimizer 'adam' or 'sgd' (default 'adam')
            num_epochs : int, number of epochs to train for, (default 100)
            reduce_on_plateau : bool, reduce learning rate by 10x on plateau of 10 epochs (default True)
            weight_decay : float, weight decay parameter for optimizer, (default 0.0)
        """
        assert isinstance(input_size, int) and input_size > 0,\
            "input_size must by type int > 0."
        assert isinstance(n_components, int) and input_size > 0,\
            "n_components must by type int > 0."
        assert model_type in ['LDA', 'prodLDA'],\
            "model must be 'LDA' or 'prodLDA'."
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu'], \
            "activation must be 'softplus' or 'relu'."
        assert isinstance(batch_size, int) and batch_size > 0,\
            "batch_size must be int > 0."
        assert isinstance(lr, float) and lr > 0, "lr must be float > 0."
        assert isinstance(momentum, float) and momentum > 0 and momentum <= 1,\
            "momentum must be 0 < float <= 1."
        assert solver in ['adam', 'sgd'], "solver must be 'adam' or 'sgd'."
        assert isinstance(reduce_on_plateau, bool),\
            "reduce_on_plateau must be type bool."
        assert isinstance(weight_decay, float) and weight_decay >= 0,\
            "weight_decay must be float >= 0."

        self.input_size = input_size
        self.n_components = n_components
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.solver = solver
        self.num_epochs = num_epochs
        self.reduce_on_plateau = reduce_on_plateau
        self.weight_decay = weight_decay

        # init inference avitm network
        self.model = DecoderNetwork(
            input_size, n_components, model_type, hidden_sizes, activation)

        # init optimizer
        if self.solver == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr, betas=(self.momentum, 0.99),
                weight_decay=weight_decay)
        elif self.solver == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, momentum=self.momentum,
                weight_decay=weight_decay)

        # init lr scheduler
        if self.reduce_on_plateau:
            self.scheduler = ReduceLROnPlateau(self.optimizer, patience=10)

        # performance attributes
        self.best_score = 0
        self.best_loss = float('inf')
        self.best_loss_train = float('inf')

        # training atributes
        self.model_dir = None
        self.train_data = None
        self.val_data = None
        self.nn_epoch = None

        # Use cuda if available
        if torch.cuda.is_available():
            self.USE_CUDA = True
        else:
            self.USE_CUDA = False

    def _loss(self, inputs, word_dists, prior_mean, prior_variance,
              posterior_mean, posterior_variance):

        # KL term
        prior_variance_det = prior_variance.cumprod(dim=0)[-1]
        prior_variance_inv = 1 / prior_variance

        posterior_variance_det = posterior_variance.cumprod(dim=1)[:, -1]

        diff_means = prior_mean - posterior_mean

        # trace(\Sigma_1^-1 \Sigma_0)
        tr_sigs = torch.sum(prior_variance_inv * posterior_variance, dim=1)
        # (\mu_1 - \mu_0)^T \Sigm_1^-1 (\mu_1 - \mu_0)
        mdiff_sig1inv_mdiff = torch.sum(
            diff_means * prior_variance_inv * diff_means, dim=1)
        # log |\Sigma_1| / |\Sigma_0|
        sig_det_ratio = prior_variance_det / posterior_variance_det

        KL = 0.5 * (tr_sigs + mdiff_sig1inv_mdiff - self.n_components
                    + torch.log(sig_det_ratio + 1e-10))

        # Reconstruction term
        RL = -torch.sum(inputs * torch.log(word_dists + 1e-10), dim=1)

        loss = KL + RL

        return loss.sum()

    def _train_epoch(self, loader):
        """Train epoch."""
        self.model.train()
        train_loss = 0
        samples_processed = 0

        for batch_samples in loader:
            # batch_size x vocab_size
            X = batch_samples['X']

            if self.USE_CUDA:
                X = X.cuda()

            # forward pass
            self.model.zero_grad()
            prior_mean, prior_variance, \
                posterior_mean, posterior_variance, word_dists = self.model(X)

            # backward pass
            loss = self._loss(
                X, word_dists, prior_mean, prior_variance,
                posterior_mean, posterior_variance)
            loss.backward()
            self.optimizer.step()

            # compute train loss
            samples_processed += X.size()[0]
            train_loss += loss.item()

        return samples_processed, train_loss

    def _eval_epoch(self, loader):
        """Eval epoch."""
        self.model.eval()
        eval_loss = 0
        samples_processed = 0

        with torch.no_grad():
            for batch_samples in loader:
                # batch_size x vocab_size
                X = batch_samples['X']

                if self.USE_CUDA:
                    X = X.cuda()

                # forward pass
                self.model.zero_grad()
                prior_mean, prior_variance, \
                    posterior_mean, posterior_variance, \
                    word_dists = self.model(X)

                # backward pass
                loss = self._loss(
                    X, word_dists, prior_mean, prior_variance,
                    posterior_mean, posterior_variance)

                # compute train loss
                samples_processed += X.size()[0]
                eval_loss += loss.item()

        return samples_processed, eval_loss

    def fit(self, train_dataset, val_dataset, save_dir):
        """
        Train the AVITM model.

        Args
            train_dataset : PyTorch Dataset classs for training data.
            val_dataset : PyTorch Dataset classs for validation data.
            save_dir : directory to save checkpoint models to.
        """
        # Print settings to output file
        print("Settings: \n\
               N Components: {}\n\
               Topic Prior Mean: {}\n\
               Topic Prior Variance: {}\n\
               Model Type: {}\n\
               Hidden Sizes: {}\n\
               Activation: {}\n\
               Learning Rate: {}\n\
               Momentum: {}\n\
               Reduce On Plateau: {}\n\
               Weight Decay: {}\n\
               Save Dir: {}".format(
                   self.n_components, 0.0,
                   1. - (1./self.n_components), self.model_type,
                   self.hidden_sizes, self.activation, self.lr, self.momentum,
                   self.reduce_on_plateau, self.weight_decay, save_dir))

        self.model_dir = save_dir
        self.train_data = train_dataset
        self.val_data = val_dataset

        train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True,
            num_workers=mp.cpu_count())
        val_loader = DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=True,
            num_workers=mp.cpu_count())

        # init training variables
        train_loss = 0
        samples_processed = 0

        # train loop
        for epoch in range(self.num_epochs + 1):
            self.nn_epoch = epoch
            if epoch > 0:
                # train epoch
                sp, train_loss = self._train_epoch(train_loader)
                samples_processed += sp

            # eval epoch
            _, val_loss = self._eval_epoch(val_loader)

            # val_score = self.score(self.val_data)
            val_score = 0

            # report
            print("Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss: {}\tValidation Loss: {}\tValidation Score: {}".format(
                epoch, self.num_epochs, samples_processed,
                len(self.train_data)*self.num_epochs, train_loss,
                val_loss, val_score))

            # save best
            if val_score > self.best_score:
                self.best_score = val_score
                self.best_loss = val_loss

                self.best_loss_train = train_loss

                self.save(save_dir)

    def predict(self, dataset, k=10):
        """Predict input."""
        self.model.eval()

        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=mp.cpu_count())

        preds = []

        with torch.no_grad():
            for batch_samples in loader:
                # batch_size x vocab_size
                X = batch_samples['X']

                if self.USE_CUDA:
                    X = X.cuda()

                # forward pass
                self.model.zero_grad()
                _, _, _, _, word_dists = self.model(X)

                _, indices = torch.sort(word_dists, dim=1)
                preds += [indices[:, :k]]

            preds = torch.cat(preds, dim=0)

        return preds

    def score(self, dataset, scorer='coherence', k=10):
        """Score model."""
        preds = self.predict(dataset, k)
        if scorer == 'perplexity':
            # score = perplexity_score(truth, preds)
            raise NotImplementedError("Not implemented yet.")
        elif scorer == 'coherence':
            # score = coherence_score(truth, preds)
            raise NotImplementedError("Not implemented yet.")
        else:
            raise ValueError("Unknown score type!")

        return score

    def _format_file(self):
        model_dir = "AVITM_nc_{}_tpm_{}_tpv_{}_hs_{}_ac_{}_lr_{}_mo_{}_rp_{}_wd_{}".\
            format(self.n_components, 0.0, 1 - (1./self.n_components),
                   self.model_type, self.hidden_sizes, self.activation,
                   self.lr, self.momentum, self.reduce_on_plateau,
                   self.weight_decay)
        return model_dir

    def save(self, models_dir=None):
        """
        Save model.

        Args
            models_dir: path to directory for saving NN models.
        """
        if (self.model is not None) and (models_dir is not None):

            model_dir = self._format_file()
            if not os.path.isdir(os.path.join(models_dir, model_dir)):
                os.makedirs(os.path.join(models_dir, model_dir))

            filename = "epoch_{}".format(self.nn_epoch) + '.pth'
            fileloc = os.path.join(models_dir, model_dir, filename)
            with open(fileloc, 'wb') as file:
                torch.save({'state_dict': self.model.state_dict(),
                            'dcue_dict': self.__dict__}, file)

    def load(self, model_dir, epoch):
        """
        Load a previously trained model.

        Args
            model_dir: directory where models are saved.
            epoch: epoch of model to load.
        """
        epoch_file = "epoch_"+str(epoch)+".pth"
        model_file = os.path.join(model_dir, epoch_file)
        with open(model_file, 'rb') as model_dict:
            checkpoint = torch.load(model_dict)

        for (k, v) in checkpoint['dcue_dict'].items():
            setattr(self, k, v)

        self._init_nn()
        self.model.load_state_dict(checkpoint['state_dict'])
