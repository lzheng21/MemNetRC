from keras.preprocessing import sequence
from keras.models import Sequential,Model, Graph
from keras.layers.core import *
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import *
from keras.layers.convolutional import *
from keras.utils import np_utils
from keras.optimizers import *
from keras.layers.wrappers import TimeDistributed
from keras.layers import Input, Dense, merge
import numpy as np
from keras.optimizers import *
import pickle
from keras.layers.normalization import *
import theano
from keras.regularizers import l2, activity_l2
from keras.layers.pooling import *
from keras import backend as K
def build(config,num_docs,doc_max_len,vocab_size):

    doc_input = Input(shape=(num_docs,doc_max_len),dtype='int32')
    embedding = Embedding(output_dim=config['word_dim'], input_length=doc_max_len, input_dim=vocab_size, mask_zero=True)
    embedding_1 = TimeDistributed(embedding, input_shape=(num_docs, doc_max_len), input_dtype='int32')(doc_input)
