from __future__ import print_function
from pprint import pprint
from data import create_vocab, filter_vocab
from utils import load_json, write_json, create_idict
import numpy as np


def create_x_y(data, vocab, stats,
               surround_size=10):
    ''' create input-output pairs for neural network training
    the input is (#examples,
    '''
    data = filter_vocab(data, vocab)
    max_span = stats['max_span']
    max_q = stats['max_q']
    ivocab = create_idict(vocab)
    S = []
    CL = []
    CR = []
    Q = []

    def print_sentence(name, sen):
        print(name, [ivocab[v] for v in sen])

    for paragraph in data:
        context = paragraph['context.tokens']
        for qa in paragraph['qas']:
            # extract question.
            q = np.zeros(max_q)
            for (i, word) in enumerate(qa['question.tokens']):
                q[i] = vocab[word]
            for answer in qa['answers']:
                try:
                    print_sentence('question', q)
                    # extract span.
                    s = np.zeros(max_span)
                    for (i, word) in enumerate(answer['text.tokens']):
                        s[i] = vocab[word]
                    print_sentence('span', s)
                    # extract context left.
                    answer_start = answer['answer_start']
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
                    print()

                    S.append(s)
                    CL.append(cl)
                    CR.append(cr)
                    Q.append(q)

                except Exception as e:
                    print('context', context)
                    print('answer', answer)
                    print('ind', ind)
                    print(e.message)
                    import pdb; pdb.set_trace();
                    raise e

    S = np.array(S)
    Q = np.array(Q)
    CL = np.array(CL)
    CR = np.array(CR)
    return (S, Q, CL, CR)


if __name__ == '__main__':
    data = load_json('output/train-v1.1.small.json')
    (vocab, stats) = create_vocab(data)
    print('vocab stats:')
    pprint(stats)
    create_x_y(data, vocab, stats)



