from __future__ import print_function
from __init__ import *
from pprint import pprint
from data import create_vocab, filter_vocab
from utils import load_json, write_json, create_idict, choice, locate
from db import KeyValueStore
import numpy as np
import traceback
from keras.layers import Input, merge
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda, Dense
from keras.engine.topology import Merge
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
import keras.backend as K
import random

word2vec = KeyValueStore('word2vec')

def create_x_y(data, vocab, stats, test=False, verbose=False):
    ''' create input-output pairs for neural network training
    the input is (#examples,
    '''
    data = filter_vocab(data, vocab, stats)
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

    try:
        for paragraph in data:
            context = paragraph['context.tokens']
            all_spans = sum(paragraph['spans'], [])
            for qa in paragraph['qas']:
                # extract question.
                q = np.zeros(max_q)
                for (i, word) in enumerate(qa['question.tokens']):
                    q[i] = vocab[word]

                def extract(pos, span):
                    print_sentence('question', q)
                    # extract span.
                    s = np.zeros(max_span)
                    for (i, word) in enumerate(span):
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
                        ind = answer_start + len(span) + i
                        if ind < len(context):
                            cr[i] = vocab[context[ind]]
                    print_sentence('cr', cr)
                    return (s, q, cl, cr)

                if not test:
                    for answer in qa['answers']:
                        verbose=True
                        X.append(extract(answer['answer_start'], answer['text.tokens']))
                        Y.append(1.)
                        verbose=False
                    # spans = choice(all_spans, neg_samples, replace=True)
                    spans = all_spans
                if test:
                    spans = all_spans
                for span in spans:
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
                    span = replace(span, '-LSB-', '[')
                    span = replace(span, '-RSB-', ']')
                    pos = locate(context, span)
                    X.append(extract(pos, span))
                    Y.append(0.)

    except Exception as e:
        print('context', context)
        print('answer', answer)
        print('span', span)
        print(e.message)
        traceback.print_exc()
        import pdb; pdb.set_trace();
        #raise e

    if not test:
        random.shuffle(X)
    S = np.array([x[0] for x in X])
    Q = np.array([x[1] for x in X])
    CL = np.array([x[2] for x in X])
    CR = np.array([x[3] for x in X])
    Y = np.array(Y)
    return (S, Q, CL, CR, Y)


def predict_span(model, data, vocab, config):
    data = filter_vocab(data, vocab, config)
    import pdb; pdb.set_trace();
    (S, Q, CL, CR, Y) = create_x_y(data, vocab, config, test=True)
    all_probs = model.predict([CL, CR, Q])
    pt = 0
    predictions = {}
    for paragraph in data:
        context = paragraph['context.tokens']
        all_spans = sum(paragraph['spans'], [])
        for qa in paragraph['qas']:
            probs = all_probs[pt : pt + len(all_spans)]
            pred_ind = np.argmax(probs, axis=0)
            pred_span = all_spans[pred_ind]
            pt += len(all_spans)
            qid = qa['id']
            predictions[qid] = pred_span
    return predictions


def compile(config, vocab):
    hidden_dim = config['hidden_dim']
    lr = config['lr']
    vocab_size = len(vocab)
    print('vocab size', vocab_size)
    print('initializing weights from db')
    word_weights = np.zeros((vocab_size, hidden_dim))
    for word in vocab:
        wi = vocab[word]
        if word == '<none>':
            continue
        vec = word2vec[word]
        if not vec:
            print('warning: word2vec OOV', word)
            word_weights[wi] = np.array(word2vec['<unk>'])
        else:
            word_weights[wi] = np.array(vec)
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
    #x = merge([x_cl, x_cr, x_q], mode=lambda (cl, cr, q): K.sum((cl + cr) * q, axis=1, keepdims=True),
    #          output_shape=lambda input_shape: (input_shape[0], 1))
    x = Dense(1000, activation='relu')(x)
    x = Dense(1000, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(input=[input_cl, input_cr, input_q], output=x)

    model.summary()

    optimizer = RMSprop(lr=config['lr'], rho=0.9, epsilon=1e-8)
    model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

    return model


if __name__ == '__main__':
    data = load_json('output/train-v1.1.small.json')
    data = [data[0]]
    (vocab, stats) = create_vocab(data)
    config = {
        'hidden_dim': 300,
        'lr': 1e-4,
        'neg_samples': 1,
        'surround_size': 10
    }
    config.update(stats)
    print('config = ')
    pprint(config)

    print('creating x y')
    (S, Q, CL, CR, Y) = create_x_y(data, vocab, config)
    print(Y)

    model = compile(config, vocab)

    print('training starts')
    history = model.fit([CL, CR, Q], Y,
                        batch_size=64, nb_epoch=500,
                        verbose=1
                        )

    predictions = predict_span(model, data, vocab, config)
    print('predicing')
    pprint(predictions)





