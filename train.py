import os
import numpy as np

from sklearn.datasets import fetch_20newsgroups_vectorized

from avitm.avitm import AVITM
from datasets.newsgroup import NewsGroupDataset

newsgroups_train = fetch_20newsgroups_vectorized(subset='train')
input = newsgroups_train.data[:,:1995].toarray()
train_mask = np.random.rand(len(input)) < 0.8
train = newsgroups_train.data[:,:1995].toarray()[train_mask]
val = newsgroups_train.data[:,:1995].toarray()[~train_mask]

avitm = AVITM(input_size=1995, n_components=10, model_type='prodLDA',
              hidden_sizes=(100,), activation='softplus', batch_size=64,
              lr=2e-3, momentum=0.99, solver='adam', num_epochs=50,
              reduce_on_plateau=True, weight_decay=1e-5)


train_data = NewsGroupDataset(train)
val_data = NewsGroupDataset(val)
avitm.fit(train_data, val_data, os.path.join(os.getcwd(), 'outputs'))

avitm.predict(val_data, 10)
