from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import numpy as np


glove_input_file = 'F:/Codes/pythonproject/word_embeddings/glove.840B.300d.txt'
word2vec_output_file = 'F:/Codes/pythonproject/word_embeddings/glove.840B.300d.word2vec.txt'
# (count, dimensions) = glove2word2vec(glove_input_file, word2vec_output_file)

# print(count, '\n', dimensions)


glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

cat_vec = glove_model['cat']
print(cat_vec)
print(glove_model.most_similar('frog'))


# result = word_vectors.most_similar(positive=['woman', 'king'], negative=['man'])
# print(result)

# U, s, Vh = la.svd(X, full_matrices=False)

# for i in range(matrix.shape[0]):
#               plt.text(U[i, 0], U[i, 1], words[i])
              
# coord = U[:, 0:2]
# plt.xlim(np.min(coord[:, 0]) - 0.1, np.max(coord[:, 0]) + 0.1)
# plt.ylim(np.min(coord[:, 1]) - 0.1, np.max(coord[:, 1]) + 0.1)