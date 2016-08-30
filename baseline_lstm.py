from __future__ import print_function
from __init__ import *
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
import argparse

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
    verbose=0
    def print_sentence(name, sen):
        if verbose:
            print(name, ' '.join([ivocab[v] for v in sen if v]))

    def map_vocab(word):
        if word in vocab:
            return vocab[word]
        else:
            return vocab['<unk>']

    try:
        for paragraph in data:
            context = paragraph['context.tokens']
            all_spans = sum(paragraph['spans'], [])
            for qa in paragraph['qas']:
                # extract question.
                q = np.zeros(max_q)
                for (i, word) in enumerate(qa['question.tokens']):
                    if i >= len(q):
                        break
                    q[i] = map_vocab(word)

                def extract(pos, span, is_answer=False):
                    if verbose:
                        print('is_answer', is_answer)
                    print_sentence('question', q)
                    # extract span.
                    s = np.zeros(max_span)
                    for (i, word) in enumerate(span):
                        if i >= len(s):
                            break
                        s[i] = map_vocab(word)
                    print_sentence('span', s)
                    # extract context left.
                    answer_start = pos
                    cl = np.zeros(surround_size)
                    cr = np.zeros(surround_size)
                    for i in range(surround_size):
                        ind = answer_start - surround_size + i
                        if ind >= 0:
                            cl[i] = map_vocab(context[ind])
                    print_sentence('cl', cl)
                    for i in range(surround_size):
                        ind = answer_start + len(span) + i
                        if ind < len(context):
                            cr[i] = map_vocab(context[ind])
                    print_sentence('cr', cr)
                    if verbose:
                        print()

                    return (s, q, cl, cr)

                if not test:
                    for answer in qa['answers']:
                        X.append(extract(answer['answer_start'],
                                answer['text.tokens'], is_answer=True) + (1.,))
                    spans = choice(all_spans, neg_samples, replace=True)
                    #spans = all_spans
                if test:
                    spans = all_spans
                for span in spans:
                    pos = locate(context, span)
                    X.append(extract(pos, span) + (0.,))

    except Exception as e:
        print(e.message)
        traceback.print_exc()
        import pdb; pdb.set_trace();
        #raise e

    if not test:
        random.shuffle(X)

    return X


def unpack_x_y(batch):
    S = np.array([x[0] for x in batch])
    Q = np.array([x[1] for x in batch])
    CL = np.array([x[2] for x in batch])
    CR = np.array([x[3] for x in batch])
    Y = np.array([x[4] for x in batch])
    return (S, Q, CL, CR, Y)


def lexicalize(span):
    span = ' '.join(span)
    span = span.replace(' \'s', '\'s')
    return span


def predict_span(model, data, vocab, config):
    data = filter_vocab(data, vocab, config)
    (S, Q, CL, CR, Y) = unpack_x_y(create_x_y(data, vocab, config, test=True))
    all_probs = model.predict([S, CL, CR, Q])
    pt = 0
    predictions = {}
    for paragraph in data:
        context = paragraph['context.tokens']
        all_spans = sum(paragraph['spans'], [])
        for qa in paragraph['qas']:
            probs = all_probs[pt : pt + len(all_spans)]
            pred_ind = np.argmax(probs, axis=0)
            pred_span = all_spans[pred_ind]
            pred_span = lexicalize(pred_span)
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
            #print('warning: word2vec OOV', word)
            word_weights[wi] = np.array(word2vec['<unk>'])
        else:
            word_weights[wi] = np.array(vec)
    embed_layer = Embedding(vocab_size, hidden_dim, weights=[word_weights], mask_zero=True)
    sum_layer = Lambda(lambda emb: K.sum(emb, axis=1), output_shape=lambda input_shape: (input_shape[0], input_shape[2]))
    model_cl = Sequential()
    input_cl = Input(shape=(config['surround_size'],), name='in_cl')
    x_cl = embed_layer(input_cl)
    # x_cl = sum_layer(x_cl)
    x_cl = LSTM(hidden_dim)(x_cl)
    #model_cl.add(Lambda(lambda emb: K.sum(emb, axis=1),
    #                    output_shape=lambda input_shape: (input_shape[0], input_shape[2])
    #            ))
    input_cr = Input(shape=(config['surround_size'],), name='in_cr')
    x_cr = embed_layer(input_cr)
    # x_cr = sum_layer(x_cr)
    x_cr = LSTM(hidden_dim)(x_cr)

    input_q = Input(shape=(config['max_q'],), name='in_q')
    x_q = embed_layer(input_q)
    # x_q = sum_layer(x_q)
    x_qs = LSTM(hidden_dim)(x_q)
    x_qc = LSTM(hidden_dim)(x_q)

    input_s = Input(shape=(config['max_span'],), name='in_s')
    x_s = embed_layer(input_s)
    x_s = LSTM(hidden_dim)(x_s)

    # use fc layers.
    #x = merge([x_s, x_cl, x_cr, x_q], mode='concat')
    #x = Dense(300, activation='relu')(x)
    #x = Dense(300, activation='relu')(x)
    #x = Dense(1, activation='sigmoid')(x)

    # similarity between span and question.
    comp_s = merge([x_s, x_qs], mode=lambda (s, q): K.sum(s * q, axis=1, keepdims=True),
              output_shape=lambda input_shape: (input_shape[0][0], 1))
    comp_c = merge([x_cl, x_cr, x_qc], mode=lambda (cl, cr, q): K.sum((cl + cr) * q, axis=1, keepdims=True),
              output_shape=lambda input_shape: (input_shape[0][0], 1))
    comp = merge([comp_s, comp_c], mode='concat')
    x = Dense(1, activation='sigmoid')(comp)

    # construct the model.
    model = Model(input=[input_s, input_cl, input_cr, input_q], output=x)

    model.summary()

    optimizer = RMSprop(lr=config['lr'], rho=0.9, epsilon=1e-8)
    model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])

    return model, {
        'input_cl': input_cl,
        'input_cr': input_cr,
        'input_q': input_q,
        'x_cr': x_cr,
        'x_cl': x_cl,
        'x_q': x_q
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lstm baseline for QA on squad dataset.')
    parser.add_argument('--mode', type=str, default='full')
    args = parser.parse_args()

    #data = load_json('output/train-v1.1.small.json')
    data = load_json('output/train-v1.1.json')
    dev_data = load_json('output/dev-v1.1.json')

    if args.mode == 'tiny':
        data = data[:10]
        dev_data = dev_data[:10]
    if args.mode == 'small':
        data = data[:100]
        dev_data = dev_data[:100]
    evaluator = Evaluator(dev_data)

    (vocab, stats) = create_vocab(data)
    (vocab, _) = create_vocab(dev_data, vocab)

    config = {
        'hidden_dim': 300,
        'lr': 1e-4,
        'neg_samples': 10,
        'surround_size': 10
    }
    config.update(stats)
    print('config = ')
    pprint(config)

    print('creating x y')

    model, meta = compile(config, vocab)

    print('training starts')
    num_epoch = 100
    for epoch in range(num_epoch):
        print('epoch', epoch)
        batch = create_x_y(data, vocab, config)
        random.shuffle(batch)
        (S, Q, CL, CR, Y) = unpack_x_y(batch)
        history = model.fit([S, CL, CR, Q], Y,
                            batch_size=64, nb_epoch=1,
                            verbose=1
                            )
        print('num examples seen', epoch * len(Y))

        if (epoch + 1) % 20 == 0:
            predictions = predict_span(model, dev_data, vocab, config)
            print('predicing')
            pprint(predictions)
            print('F1', evaluator.F1(predictions), 'Exact Match', evaluator.ExactMatch(predictions))





