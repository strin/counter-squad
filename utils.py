import json

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def write_json(obj, path):
    with open(path, 'w') as f:
        return json.dump(obj, f)


def locate(sent, span):
    start = 0
    while True:
        ind = sent.index(span[0], start)
        if sent[ind:ind + len(span)] == span:
            break
        start = ind + 1
    return ind

