"""PyTorch class for feed foward inference network."""

from collections import OrderedDict
from torch import nn
import torch


class InferenceNetwork(nn.Module):

    """Inference Network."""

    def __init__(self, input_size, output_size, hidden_sizes,
                 activation='softplus', dropout=0.2):
        """
        Initialize InferenceNetwork.

        Args
            input_size : int, dimension of input
            output_size : int, dimension of output
            hidden_sizes : tuple, length = n_layers - 2
            activation : string, 'softplus' or 'relu', default 'softplus'
            dropout : float, default 0.2, default 0.2
        """
        super(InferenceNetwork, self).__init__()
        assert isinstance(input_size, int), "input_size must by type int."
        assert isinstance(output_size, int), "output_size must be type int."
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu'], \
            "activation must be 'softplus' or 'relu'."
        assert dropout >= 0, "dropout must be >= 0."

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout

        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'relu':
            self.activation = nn.ReLU()

        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.hiddens = nn.Sequential(OrderedDict([
            ('l_{}'.format(i // 2), nn.Linear(h_in, h_out)) if i % 2 == 0 else
            ('a_{}'.format(i // 2), self.activation) for i, (h_in, h_out) in
            enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:]))]))
        self.f_mu = nn.Linear(hidden_sizes[-1], output_size)
        self.f_mu_batchnorm = nn.BatchNorm1d(output_size, affine=False)
        self.f_sigma = nn.Linear(hidden_sizes[-1], output_size)
        self.f_sigma_batchnorm = nn.BatchNorm1d(output_size, affine=False)
        self.dropout_enc = nn.Dropout(p=self.dropout)

    def forward(self, x):
        """Forward pass."""
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.hiddens(x)
        x = self.dropout_enc(x)
        mu = self.f_mu_batchnorm(self.f_mu(x))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))

        return mu, log_sigma
