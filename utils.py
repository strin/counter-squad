from __future__ import print_function
import json
import numpy.random as npr
import os

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


def replace(l, ws, wt):
    new_l = []
    for w in l:
        if w == ws:
            new_l.append(wt)
        else:
            new_l.append(w)
    return new_l


def mkdir_if_not_exist(path):
    if path == '':
        return
    if not os.path.exists(path):
        os.makedirs(path)



class Checkpoint(object):
    def __init__(self, dirname):
        self.dirname = dirname
        mkdir_if_not_exist(dirname)


    def log(self, it, obj):
        write_json(obj, os.path.join(self.dirname, 'checkpoint_%d' % it))

