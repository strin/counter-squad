from __future__ import print_function
from __init__ import *
from pprint import pprint
from data import create_vocab, filter_vocab
from utils import load_json, write_json, create_idict, choice, locate
from db import KeyValueStore
import numpy as np
from keras.layers import Input, merge
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda, Dense
from keras.engine.topology import Merge
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
import keras.backend as K

word2vec = KeyValueStore('word2vec')

def create_x_y(data, vocab, stats, verbose=False):
    ''' create input-output pairs for neural network training
    the input is (#examples,
    '''
    #data = filter_vocab(data, vocab)
    max_span = stats['max_span']
    max_q = stats['max_q']
    surround_size = stats['surround_size']
    neg_samples = stats['neg_samples']
    ivocab = create_idict(vocab)
    X = []
    Y = []

    def print_sentence(name, sen):
        if verbose:
            print(name, [ivocab[v] for v in sen])

    for paragraph in data:
        context = paragraph['context.tokens']
        all_spans = sum(paragraph['spans'], [])
        for qa in paragraph['qas']:
            # extract question.
            q = np.zeros(max_q)
            for (i, word) in enumerate(qa['question.tokens']):
                q[i] = vocab[word]
            for answer in qa['answers']:
                def extract(pos):
                    print_sentence('question', q)
                    # extract span.
                    s = np.zeros(max_span)
                    for (i, word) in enumerate(answer['text.tokens']):
                        s[i] = vocab[word]
                    print_sentence('span', s)
                    # extract context left.
                    answer_start = pos
                    cl = np.zeros(surround_size)
                    cr = np.zeros(surround_size)
                    for i in range(surround_size):
                        ind = answer_start - 1 - i
                        if ind >= 0:
                            cl[i] = vocab[context[ind]]
                    print_sentence('cl', cl)
                    for i in range(surround_size):
                        ind = answer_start + len(answer['text.tokens']) + i
                        if ind < len(context):
                            cr[i] = vocab[context[ind]]
                    print_sentence('cr', cr)
                    return (s, q, cl, cr)

                try:
                    X.append(extract(answer['answer_start']))
                    Y.append(1.)
                    for span in choice(all_spans, neg_samples, replace=True):
                        def replace(l, ws, wt):
                            new_l = []
                            for w in l:
                                if w == ws:
                                    new_l.append(wt)
                                else:
                                    new_l.append(w)
                            return new_l
                        span = replace(span, '-LRB-', '(')
                        span = replace(span, '-RRB-', ')')
                        span = replace(context, '.', '')
                        pos = locate(context, span)
                        X.append(extract(pos))
                        Y.append(0.)

                except Exception as e:
                    print('context', ' '.join(context))
                    print('answer', answer)
                    print('span', span)
                    print(e.message)
                    import pdb; pdb.set_trace();
                    #raise e

    S = np.array([x[0] for x in X])
    Q = np.array([x[1] for x in X])
    CL = np.array([x[2] for x in X])
    CR = np.array([x[3] for x in X])
    Y = np.array(Y)
    return (S, Q, CL, CR, Y)


def compile(config, vocab):
    hidden_dim = config['hidden_dim']
    lr = config['lr']
    vocab_size = len(vocab)
    print('vocab size', vocab_size)
    print('initializing weights from db')
    word_weights = np.zeros((vocab_size, hidden_dim))
    for (wi, word) in enumerate(vocab):
        word_weights[wi] = np.array(word2vec[word])
    embed_layer = Embedding(vocab_size, hidden_dim, weights=[word_weights])
    sum_layer = Lambda(lambda emb: K.sum(emb, axis=1), output_shape=lambda input_shape: (input_shape[0], input_shape[2]))
    model_cl = Sequential()
    input_cl = Input(shape=(config['surround_size'],), name='in_cl')
    x_cl = embed_layer(input_cl)
    x_cl = sum_layer(x_cl)
    #model_cl.add(Lambda(lambda emb: K.sum(emb, axis=1),
    #                    output_shape=lambda input_shape: (input_shape[0], input_shape[2])
    #            ))
    input_cr = Input(shape=(config['surround_size'],), name='in_cr')
    x_cr = embed_layer(input_cr)
    x_cr = sum_layer(x_cr)

    input_q = Input(shape=(config['max_q'],), name='in_q')
    x_q = embed_layer(input_q)
    x_q = sum_layer(x_q)

    x = merge([x_cl, x_cr, x_q], mode='concat')
    x = Dense(100, activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(input=[input_cl, input_cr, input_q], output=x)

    model.summary()

    optimizer = RMSprop(lr=config['lr'], rho=0.9, epsilon=1e-8)
    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

    return model


if __name__ == '__main__':
    data = load_json('output/train-v1.1.small.json')
    (vocab, stats) = create_vocab(data)
    config = {
        'hidden_dim': 300,
        'lr': 1e-3,
        'neg_samples': 10,
        'surround_size': 10
    }
    config.update(stats)
    print('config = ')
    pprint(config)

    print('creating x y')
    (S, Q, CL, CR, Y) = create_x_y(data, vocab, config)

    model = compile(config, vocab)

    print('training starts')
    history = model.fit([CL, CR, Q], Y,
                        batch_size=64, nb_epoch=3,
                        verbose=1
                        )




