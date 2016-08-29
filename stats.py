from __future__ import print_function
from pprint import pprint
from data import create_vocab, filter_vocab
from collections import Counter
from utils import load_json, write_json, create_idict, choice, locate
import sys


if __name__ == '__main__':
    name = sys.argv[1]

    data = load_json(sys.argv[1])

    span_dist = Counter()

    for paragraph in data:
        for qa in paragraph['qas']:
            for answer in qa['answers']:
                span_len = len(answer['text.tokens'])
                span_dist[span_len] += 1

    span_dist_total = float(sum(span_dist.values()))
    span_dist = {k : v / span_dist_total for (k, v) in span_dist.items()}
    print('span distribution')
    pprint(span_dist)

