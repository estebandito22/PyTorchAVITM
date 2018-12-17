import os
import csv
import pickle

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

pickle.dump(train_docs, open('abs.p', 'wb'))
pickle.dump(train_docs, open('vocab.p', 'wb'))
