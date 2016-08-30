from __future__ import print_function
from pprint import pprint
from data import create_vocab, filter_vocab
from utils import load_json, write_json, create_idict, choice, locate
from db import KeyValueStore
from evaluate import Evaluator
import numpy as np
import traceback
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
                        ind = answer_start - 1 - i
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


def predict_span(data, vocab, config):
    data = filter_vocab(data, vocab, config)
    dataset = create_x_y(data, vocab, config, test=True)
    embedding = np.zeros((len(vocab), config['hidden_dim']))
    for word in vocab:
        if word == '<none>':
            continue
        if word not in word2vec:
            word = '<unk>'
        embedding[vocab[word]] = np.array(word2vec[word])

    pt = 0
    predictions = {}
    datap = 0

    ivocab = create_idict(vocab)
    X = []
    verbose=0
    def print_sentence(name, sen):
        print(name, ' '.join([ivocab[v] for v in sen if v]))

    for paragraph in data:
        context = paragraph['context.tokens']
        all_spans = sum(paragraph['spans'], [])
        for qa in paragraph['qas']:
            scores = []
            for (si, span) in enumerate(all_spans):
                (s, q, cl, cr, y) = dataset[datap + si]
                emb_q = np.mean(embedding[q.astype(int)], axis=0)
                emb_cl = np.mean(embedding[cl.astype(int)], axis=0)
                emb_cr = np.mean(embedding[cr.astype(int)], axis=0)
                emb_c = (emb_cl + emb_cr) / 2.
                sim = np.dot(emb_q, emb_c)
                scores.append(sim)
            pred_ind = np.argmax(scores, axis=0)
            pred_span = all_spans[pred_ind]
            pred_span = lexicalize(pred_span)
            for (si, (span, score)) in enumerate(zip(all_spans, scores)):
                print('[', lexicalize(span), ']', score)
                (s, q, cl, cr, y) = dataset[datap + si]
                print_sentence('cl', cl)
                print_sentence('cr', cr)
                print_sentence('cl', s)
                print()

            pprint(qa)
            print('span', pred_span)
            import pdb; pdb.set_trace();
            pt += len(all_spans)
            qid = qa['id']
            predictions[qid] = pred_span
            datap += len(all_spans)
    return predictions


if __name__ == '__main__':
    data = load_json('output/train-v1.1.small.json')
    dev_data = load_json('output/dev-v1.1.json')

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

    predictions = predict_span(dev_data, vocab, config)
    print('predicing')
    pprint(predictions)
    print('F1', evaluator.F1(predictions), 'Exact Match', evaluator.ExactMatch(predictions))





