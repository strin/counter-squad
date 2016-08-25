from __future__ import print_function
from pprint import pprint
from data import create_vocab, filter_vocab
from utils import load_json, write_json, locate


def create_x_y(data, vocab, stats):
    ''' create input-output pairs for neural network training
    the input is (#examples,
    '''
    data = filter_vocab(data, vocab)
    for paragraph in data:
        context = sum(paragraph['sentences'], [])
        for qa in paragraph['qas']:
            for answer in qa['answers']:
                try:
                    ind = locate(context, answer['text.sent'])
                except ValueError as e:
                    print('context', context)
                    print('answer', answer)
                    print('ind', ind)
                    print(e.message)
                    raise e




if __name__ == '__main__':
    data = load_json('output/dev-v1.0.small.json')
    (vocab, stats) = create_vocab(data)
    print('vocab stats:')
    pprint(stats)
    create_x_y(data, vocab, stats)



