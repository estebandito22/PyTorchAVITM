"""PyTorch class for feed foward AVITM network."""

import torch
from torch import nn
from torch.nn import functional as F
from avitm.inference_network import InferenceNetwork


class DecoderNetwork(nn.Module):

    """AVITM Network."""

    def __init__(self, input_size, n_components=10, model_type='prodLDA',
                 hidden_sizes=(100,), activation='relu'):
        """
        Initialize InferenceNetwork.

        Args
            input_size : int, dimension of input
            n_components : int, number of topic components, (default 10)
            model_type : string, 'prodLDA' or 'LDA' (default 'prodLDA')
            hidden_sizes : tuple, length = n_layers - 2, (default (100, ))
            activation : string, 'softplus', 'relu', (default 'softplus')
        """
        super(DecoderNetwork, self).__init__()
        self.input_size = input_size
        self.n_components = n_components
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation

        self.inf_net = InferenceNetwork(
            input_size, n_components, hidden_sizes, activation)

        # init prior parameters
        # \mu_1k = log \alpha_k + 1/K \sum_i log \alpha_i;
        # \alpha = 1 \forall \alpha
        topic_prior_mean = 0.0
        self.prior_mean = torch.tensor(
            [topic_prior_mean] * n_components, requires_grad=True)
        if torch.cuda.is_available():
            self.prior_mean = self.prior_mean.cuda()

        # \Sigma_1kk = 1 / \alpha_k (1 - 2/K) + 1/K^2 \sum_i 1 / \alpha_k;
        # \alpha = 1 \forall \alpha
        topic_prior_variance = 1. - (1. / self.n_components)
        self.prior_variance = torch.tensor(
            [topic_prior_variance] * n_components, requires_grad=True)
        if torch.cuda.is_available():
            self.prior_variance = self.prior_variance.cuda()

        self.beta = nn.Linear(n_components, input_size, bias=False)
        nn.init.xavier_uniform_(self.beta.weight)

        # dropout on theta
        self.drop_theta = nn.Dropout(p=0.2)

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(torch.sqrt(logvar))
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        """Forward pass."""
        # batch_size x n_components
        posterior_mu, posterior_sigma = self.inf_net(x)

        # generate samples from theta
        theta = F.softmax(
            self.reparameterize(posterior_mu, posterior_sigma), dim=1)
        theta = self.drop_theta(theta)

        # prodLDA vs LDA
        if self.model_type == 'prodLDA':
            # in: batch_size x input_size x n_components
            word_dist = F.softmax(self.beta(theta), dim=1)
            # word_dist: batch_size x input_size
        elif self.model_type == 'LDA':
            # simplex constrain on Beta
            self.beta.weight = nn.Parameter(F.softmax(self.beta.weight, dim=0))
            word_dist = self.beta(theta)
            # word_dist: batch_size x input_size

        return self.prior_mean, self.prior_variance, \
            posterior_mu, posterior_sigma, word_dist
