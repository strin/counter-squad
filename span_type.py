# a sanity task to classify the type of questions.

from __future__ import print_function
#from __init__ import *
from pprint import pprint
from data import create_vocab, filter_vocab
from utils import load_json, write_json, create_idict, choice, locate
from db import KeyValueStore
from evaluate import Evaluator
import numpy as np
import traceback
from keras.layers import Input, merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Lambda, Dense
from keras.engine.topology import Merge
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
import keras.backend as K
import random

word2vec = KeyValueStore('word2vec')


def get_question_type(question):
    question = ' '.join(question)
    types = ['how many', 'what', 'who', 'which', 'when', 'how often',
             'where', 'how much', 'how large', 'how', 'hoe', 'why']
    for _type in types:
        if question.find(_type) != -1:
            return _type
    print(question)
    import pdb; pdb.set_trace();



if __name__ == '__main__':
    data = load_json('output/train-v1.1.small.json')
    dev_data = load_json('output/dev-v1.1.json')


    for paragraph in data:
        for qa in paragraph['qas']:
            qtype = get_question_type(qa['question.tokens'])
            print(qtype)




