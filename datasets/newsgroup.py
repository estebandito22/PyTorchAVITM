"""Class for loading NMT dataset."""

import torch
from torch.utils.data import Dataset


class NewsGroupDataset(Dataset):

    """Class to load 20 News Groups dataset."""

    def __init__(self, X, idx2token):
        """
        Initialize NewsGroupDataset.

        Args
            X : array-like, shape=(n_samples, n_features)
                Document word matrix.
        """
        self.X = X
        self.idx2token = idx2token

    def __len__(self):
        """Return length of dataset."""
        return len(self.X)

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        X = torch.FloatTensor(self.X[i])

        return {'X': X}
