import os
import csv

import numpy as np
import pandas as pd

from avitm.avitm import AVITM
from datasets.abstracts import AbstractsDataset

def vocab_to_dict(csv_path, fnames, split):
	with open(csv_path) as csvfile:
		reader = csv.DictReader(csvfile, delimiter = split, 
					fieldnames = fnames)
		mydict = {int(row[fnames[0]]) - 1:row[fnames[1]] for row in reader}
	return mydict

def split_documents(csv_path):
	result = {}
	with open(csv_path) as csvfile:
		reader = csv.reader(csvfile, delimiter = ' ')
		for row in reader:
			if row[0] in result:
				result[row[0]][int(row[1])] = int(row[2])
			else:
				result[row[0]] = {int(row[1]):int(row[2])}
	return result

cwd = os.getcwd()
vocab_size = 30799
vocab_path = os.path.join(cwd, 'data/abstracts/bow/', 'words.txt')
idx2token = vocab_to_dict(vocab_path, ['index','token'], " ")

p1_path=os.path.join(cwd,'data/abstracts/bow/nsfabs_part1_out/','docwords.txt')
p2_path=os.path.join(cwd,'data/abstracts/bow/nsfabs_part2_out/','docwords.txt')
p3_path=os.path.join(cwd,'data/abstracts/bow/nsfabs_part3_out/','docwords.txt')

train_docs_1 = split_documents(p1_path)
train_docs_2 = split_documents(p2_path)
train_docs_3 = split_documents(p3_path)

train_docs = train_docs_1
train_docs.update(train_docs_2)
train_docs.update(train_docs_3)

print(train_docs['102'])

train = np.load(os.path.join(cwd, 'data', 'train.txt.npy'), encoding='latin1')
train_bow = to_bow(train, vocab_size)
train_data = AbstractsDataset(train_bow, idx2token)

avitm = AVITM(input_size=1995, n_components=50, model_type='prodLDA',
              hidden_sizes=(100, 100), activation='softplus', dropout=0.2,
              learn_priors=False, batch_size=64, lr=2e-3, momentum=0.99,
              solver='adam', num_epochs=100, reduce_on_plateau=False)

avitm.fit(train_data)

topics = pd.DataFrame(avitm.get_topics(10)).T
topics

avitm.score(10)

