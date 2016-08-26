import json

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def write_json(obj, path):
    with open(path, 'w') as f:
        return json.dump(obj, f)

def create_idict(dict):
    return {v: k for (k, v) in dict.items()}
