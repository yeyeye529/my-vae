import os
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import stop_words
from sklearn.metrics import euclidean_distances
from pyemd import emd
import json

with open("wmd/dictionary.json","r") as f:
	model = json.load(f)
# for k,v in model.items():
# 	print(k,v)


my_stopwords_set = ['<s>', '</s>', '<pad>', '<unk>']
my_stopwords_set = set(my_stopwords_set).union(stop_words.ENGLISH_STOP_WORDS)


def get_distance(d1, d2, min_vocab=2, verbose=False, mode = 0):
	# d1 = d1.decode('utf-8')
	# d2 = d2.decode('utf-8')
	vocabulary = [w for w in set(d1.lower().split() + d2.lower().split()) if w in model and w not in my_stopwords_set]
	# print("vocabulary:",vocabulary)
	if len(vocabulary) == 0:
		return 1

	vect = CountVectorizer(vocabulary=vocabulary).fit([d1, d2])
	W_ = np.array([model[w] for w in vect.get_feature_names() if w in model])
	#print("w_",W_)

	D_ = euclidean_distances(W_)
	#print("D_",D_)
	D_ = D_.astype(np.double)
	

	D_ /= D_.max()  # just for comparison purposes

	v_1, v_2 = vect.transform([d1, d2])
	
	v_1 = v_1.toarray().ravel()
	v_2 = v_2.toarray().ravel()
	# pyemd needs double precision input
	v_1 = v_1.astype(np.double)
	v_2 = v_2.astype(np.double)
	v_1 /= v_1.sum()
	v_2 /= v_2.sum()
	if    verbose:
		print(vocabulary)
		print(v_1, v_2)
		print(D_)

	
	# if mode == 0:
	doc_distance = get_wmd_distance(v_1, v_2, D_)
	# elif mode == 1:
	# 	doc_distance = get_wcd_distance(v_1, v_2, D_)
	# else:
	# 	doc_distance = get_rwmd_distance(v_1, v_2, D_)

	return doc_distance

		

def get_wmd_distance(v1, v2, distance_matrix):

	return emd(v1, v2, distance_matrix)

# def get_wcd_distance(v1, v2, vocabulary):

# 	d1_sum = (vocabulary * v1[:, numpy.newaxis]).sum(axis = 0)
# 	d2_sum = (vocabulary * v2[:, numpy.newaxis]).sum(axis = 0)

# 	return np.linalg.norm(d1_sum - d2_sum)
	

# def get_rwmd_distance(v1, v2, distance_matrix):
# 	pass
	


if __name__ == '__main__':
	d1 = "what are some of your favorite saying"
	d2 = "what are some of the best riddle"
	print (get_distance(d1, d2))

	d1 = "how do swimming pool filter work"
	d2 = "how do liquid tea help work"
	print (get_distance(d1, d2))

	d1 = "how do swimming pool filter work"
	d2 = "how do liquid tea work"
	print (get_distance(d1, d2))
