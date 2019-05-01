import os
import json
import glob

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from pytorchavitm import AVITM
from pytorchavitm.datasets import BOWDataset


cwd = os.getcwd()

part1 = glob.glob(os.path.join(cwd, 'data', 'Part1/*/*/*.txt'))
part2 = glob.glob(os.path.join(cwd, 'data', 'Part 2/*/*/*.txt'))
part3 = glob.glob(os.path.join(cwd, 'data', 'Part 3/*/*/*.txt'))

filenames = part1 + part2 + part3

docs = []
for file in filenames:
    with open(file, 'r', encoding='latin1') as f:
        docs += [f.read()]

cv = CountVectorizer(input='content', lowercase=True, stop_words='english',
                     max_df=0.99, min_df=0.01, binary=False)

train_bow = cv.fit_transform(docs)
train_bow = train_bow.toarray()

idx2token = cv.get_feature_names()
input_size = len(idx2token)

train_data = BOWDataset(train_bow, idx2token)

avitm = AVITM(input_size=input_size, n_components=50, model_type='prodLDA',
              hidden_sizes=(100, 100), activation='softplus', dropout=0.2,
              learn_priors=True, batch_size=64, lr=2e-3, momentum=0.99,
              solver='adam', num_epochs=100, reduce_on_plateau=False)

avitm.fit(train_data)

topics = pd.DataFrame(avitm.get_topics(10)).T
topics

avitm.score(k=10, topics=10)


# w/o learned priors, prodLDA, 2layer, 0.447
# w/  learned priors, prodLDA, 2layer, 0.436
# w/  learned priors, prodLDA, 1layer, 0.458
# w/o learned priors, LDA, 2layer 0.433
