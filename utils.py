from __future__ import print_function
import json
import numpy.random as npr

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def write_json(obj, path):
    with open(path, 'w') as f:
        return json.dump(obj, f)

def create_idict(dict):
    return {v: k for (k, v) in dict.items()}


def choice(objs, size, replace=True, p=None):
    all_inds = range(len(objs))
    inds = npr.choice(all_inds, size=size, replace=replace, p=p)
    return [objs[ind] for ind in inds]


def locate(context, span):
    for i in range(len(context) - len(span) + 1):
        if context[i:i+len(span)] == span:
            return i
    print(context)
    print(span)
    raise Exception('error, cannot match span in context')
