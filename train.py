import os
import json

import numpy as np
import pandas as pd

from avitm.avitm import AVITM
from datasets.newsgroup import NewsGroupDataset


def to_bow(data, min_length):
    """Convert index lists to bag of words representation of documents."""
    vect = [np.bincount(x[x != np.array(None)].astype('int'), minlength=min_length)
            for x in data if np.sum(x[x != np.array(None)]) != 0]
    return np.array(vect)


cwd = os.getcwd()
vocab_size = 1995
vocab = os.path.join(cwd, 'data', 'vocab.pkl')
vocab = json.load(open(vocab, 'r'))
idx2token = {v: k for (k, v) in vocab.items()}

train = np.load(os.path.join(cwd, 'data', 'train.txt.npy'), encoding='latin1')
train_bow = to_bow(train, vocab_size)
train_data = NewsGroupDataset(train_bow, idx2token)

avitm = AVITM(input_size=1995, n_components=50, model_type='prodLDA',
              hidden_sizes=(100, 100), activation='softplus', dropout=0.2,
              learn_priors=False, batch_size=64, lr=2e-3, momentum=0.99,
              solver='adam', num_epochs=100, reduce_on_plateau=False)

avitm.fit(train_data)

topics = pd.DataFrame(avitm.get_topics(10)).T
topics

avitm.score(10)
