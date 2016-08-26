# Load data from file.
# painter
from __future__ import print_function
from pprint import pprint
import json
import re
from copy import deepcopy
from utils import write_json
import traceback
import nltk
from nltk import word_tokenize as nltk_word_tokenize, sent_tokenize
nltk.internals.config_java(options='-xmx16G')


WHITESPACE_AND_PUNCTUATION = set([' ', '.', ',', ':', ';', '!', '?', '$', '%', '(', ')', '[', ']', '-', '`', '\'', '"'])
ARTICLES = set(['the', 'a', 'an'])

def word_tokenize(sent):
    tokens = nltk.word_tokenize(sent)
    new_tokens = []
    for token in tokens:
        if token != '.' and token[-1] in ['.']:
            token = token[:-1]
        new_tokens.append(token)
    return new_tokens


class Loader:
    """Load the dataset."""

    def __init__(self, filepath):
        """Init."""
        self.data = None
        self.version = None
        self._load(filepath)

    def _load(self, filepath):
        with open(filepath) as data_file:
            parsed_file = json.load(data_file)
            self.data = parsed_file['data']
            self.version = parsed_file['version']

    def get_data(self):
        """Get the loaded data."""
        return self.data

    def get_article_titles(self):
        """Generator for article titles."""
        if (self.version != '1.0'):
            raise ValueError('Dataset version unrecognized.')
        return map(lambda x: x['title'], self.data)


# what follows are basic (immutable) pre-processing operators on data.
def merge_paragraphs(data):
    ''' assume QA is only dependent on the current paragraph.
    so we merge paragraphs from articles as a new dataset.
    '''
    output = []
    for article in data:
        output.extend(article['paragraphs'])
    return deepcopy(output)


def clean_text(text):
    if isinstance(text, unicode):
        text = text.replace(u'\u00a0', ' ')
    else:
        text = text.replace('\xc2\xa0', ' ')
    text = text.replace(u'\u2019', '\'')
    text = text.replace(u'\u2013', '-')

    while text and text[0] == ' ':
        text = text[1:]
    if text == '':
        return text

    is_unicode = lambda x: x.encode('ascii', 'ignore') == ''
    words = text.split(' ')
    num_space = 0
    for i in range(len(words)-1, -1, -1):
        if is_unicode(words[i]):
            num_space = 1
        else:
            break
    words = [word for word in words if not is_unicode(word)]
    text = u' '.join(words)
    text = text.encode('ascii', 'ignore') # TODO: warning utf-8 ignore
    text = text.lower() # TODO: does parse rely on this?
    text += ' ' * num_space
    return text


def clean_answer(answer):
    ''' based on the evaluator.
    '''
    shift_start = 0
    answer = clean_text(answer)
    while len(answer) > 1 and answer[0] in WHITESPACE_AND_PUNCTUATION:
        answer = answer[1:]
        shift_start += 1
    while len(answer) > 1 and answer[-1] in WHITESPACE_AND_PUNCTUATION:
        answer = answer[:-1]
    answer = answer.split()
    if len(answer) > 1 and answer[0] in ARTICLES:
        shift_start += len(answer[0]) + 1
        answer = answer[1:]
    answer = ' '.join(answer)
    return answer, shift_start


def clean_paragraphs(data):
    ''' take in dirty paragraphs and clean them up.
    '''
    data = deepcopy(data)
    for paragraph in data:
        context = paragraph['context']
        paragraph['context'] = clean_text(paragraph['context'])
        for qa in paragraph['qas']:
            qa['question'] = clean_text(qa['question'])
            for answer in qa['answers']:
                answer_start = answer['answer_start']
                answer['text'], shift_start = clean_answer(answer['text'])
                answer['answer_start'] = len(clean_text(context[:answer_start])) + shift_start # compute new start.
                try:
                    assert(answer['text'] == paragraph['context'][answer['answer_start']:answer['answer_start']+len(answer['text'])])
                except AssertionError as e:
                    print(answer['text'])
                    print(paragraph['context'][answer['answer_start']:answer['answer_start']+len(answer['text'])])
                    print(clean_text(context) + '\n')
                    print(clean_text(context[:answer_start]))
                    import pdb; pdb.set_trace();
                    raise e

    return data


def tokenize_paragraph(paragraph):
    raw_sentences = sent_tokenize(paragraph)
    sentences = []
    for raw_sentence in raw_sentences:
        sentences.append(word_tokenize(raw_sentence))
    return sentences


def tokenize_paragraphs(data):
    ''' After paragraphs have been cleaned up.
    '''
    data = deepcopy(data)
    log = open('log/tokenize.error', 'w')
    error_count = 0
    for paragraph in data:
        raw_context = paragraph['context']
        context = tokenize_paragraph(raw_context)
        paragraph['context.sents'] = context
        # context = word_tokenize(raw_context)
        context = sum(context, [])
        paragraph['context.tokens'] = context
        for qa in paragraph['qas']:
            qa['question.tokens'] = word_tokenize(qa['question'])
            for raw_answer in qa['answers']:
                answer_start = raw_answer['answer_start']
                context_pre = word_tokenize(raw_context[:answer_start])
                answer = word_tokenize(raw_answer['text'])
                ind = len(context_pre)
                raw_answer['text.tokens'] = answer
                raw_answer['answer_start'] = ind
                try:
                    assert(answer == context[ind:ind+len(answer)])
                except AssertionError as e:
                    print('\n%d' % error_count, file=log)
                    print('context', context[ind:ind+len(answer)], file=log)
                    print('answer', answer, file=log)
                    error_count += 1
                    #raise e

    return data



def validate_answer_start(data):
    error_count = 0
    for paragraph in data:
        context = paragraph['context']
        for qa in paragraph['qas']:
            for answer in qa['answers']:
                answer_start = answer['answer_start']
                try:
                    assert(context[answer_start:answer_start + len(answer['text'])]
                        == answer['text'])
                except AssertionError as e:
                    error_count += 1
                    print(answer['text'])
                    print(context[answer_start:answer_start + len(answer['text'])])
                    print(qa['id'])
                    # import pdb; pdb.set_trace();
                    # raise e
    print('total_error', error_count)



def constituents_in_tree(parsetree):
    parsetree = parsetree.replace('\n', '')
    while parsetree.find('  ') != -1:
        parsetree = parsetree.replace('  ', ' ')
    def find_constituents(parsetree, results=[]):
        if parsetree.startswith('('):
            eles = re.findall(r'\(([^ ]*) (.*)\)', parsetree)
            try:
                assert(len(eles) == 1)
            except Exception as e:
                pprint(parsetree)
                print(eles)
                raise e
            ele = eles[0]
            new_parse = ele[1]
            # split.
            level = 0
            new_parsetrees = []
            buf = ''
            for (i, ch) in enumerate(new_parse):
                if ch == '(':
                    level += 1
                elif ch == ')':
                    level -= 1
                buf += ch
                if (i == len(new_parse) - 1) or (ch == ' ' and level == 0):
                    new_parsetrees.append(buf)
                    buf = ''
                    continue

            new_constituent = []
            for new_parsetree in new_parsetrees:
                constituent = find_constituents(new_parsetree, results)
                new_constituent.extend(constituent)
            if len(new_parsetrees) > 1:
                results.append(new_constituent)
            return new_constituent
        else:
            results.append([parsetree])
            return [parsetree]
    results = []
    find_constituents(parsetree, results)
    return results



def create_stanford_parser():
    from nltk.parse.stanford import StanfordParser
    return StanfordParser('/home/durin/software/stanford-parser-full-2015-12-09/stanford-parser.jar',
                        '/home/durin/software/stanford-parser-full-2015-12-09/stanford-parser-3.6.0-models.jar')


def extract_constituents_spans(data, CORENLP_IP="0.0.0.0", CORENLP_PORT=3456):
    ''' extract spans from paragraphs based on constituents in the parse tree.
    '''
    # use JavaNLP
    #import jsonrpc
    #server = jsonrpc.ServerProxy(jsonrpc.JsonRpc20(),
    #                             jsonrpc.TransportTcpIp(addr=(CORENLP_IP, CORENLP_PORT), timeout=1000.
    #                            ))
    parser = create_stanford_parser()
    data = deepcopy(data)
    batching = 5   # batching before send to java.
    for pi in range(0, len(data), batching):
        try:
            sents = []
            for paragraph in data[pi : pi + batching]:
                sents.extend(paragraph['context.sents'])
            # results = json.loads(server.parse(paragraph['context'])) # JavaNLP.
            trees = list(parser.parse_sents(sents))
            tree_i = 0
            for paragraph in data[pi : pi + batching]:
                spans = []
                for sent in paragraph['context.sents']:
                    tree = str(list(trees[tree_i])[0])
                    results = constituents_in_tree(tree)
                    spans.append(results)
                    tree_i += 1
                paragraph['spans'] = spans
            print(pi, '/', len(data))
        except Exception as e:
            pprint(paragraph)
            traceback.print_exc()
            import pdb; pdb.set_trace();
            raise e
    return data


def extract_qa_sent(data):
    data = deepcopy(data)
    parser = create_stanford_parser()

    for (pi, paragraph) in enumerate(data):
        for qa in paragraph['qas']:
            for answer in qa['answers']:
                answer['text.sent'] = sum(tokenize_paragraph(answer['text']), [])
            sentences = tokenize_paragraph(qa['question'])
            qa['question.sent'] = sum(sentences, [])
    return data


def create_vocab(data):
    stats = {}
    vocab = {
        '<none>': 0,
        '<unk>': 1
    }

    stats['max_span'] = 0
    stats['max_q'] = 0
    def update_vocab(sentence):
        for word in sentence:
            if word in vocab:
                continue
            vocab[word] = len(vocab)

    for (pi, paragraph) in enumerate(data):
        update_vocab(paragraph['context.tokens'])
        for qa in paragraph['qas']:
            update_vocab(qa['question.tokens'])
            if len(qa['question.tokens']) > stats['max_q']:
                stats['max_q'] = len(qa['question.tokens'])
            for answer in qa['answers']:
                if len(answer['text.tokens']) > stats['max_span']:
                    stats['max_span'] = len(answer['text.tokens'])
                update_vocab(answer['text.tokens'])

    stats['max_num_span'] = 0
    for (pi, paragraph) in enumerate(data):
        new_spans = []
        for spans in paragraph['spans']:
            new_span = []
            for span in spans:
                if len(span) > stats['max_span']:
                    continue
                new_span.append(span)
            new_spans.append(new_span)
        paragraph['spans'] = new_spans
        if len(sum(new_spans, [])) > stats['max_num_span']:
            stats['max_num_span'] = len(sum(new_spans, []))

    stats['vocab_size'] = len(vocab)
    return (vocab, stats)


def filter_vocab(data, vocab, stats):
    data = deepcopy(data)
    def filter_word(word):
        if word not in vocab:
            return '<unk>'
        return word
    for (pi, paragraph) in enumerate(data):
        sentences = paragraph['context.sents']
        for i, _ in enumerate(sentences):
            sentences[i] = map(filter_word, sentences[i])
        paragraph['context.tokens'] = map(filter_word, paragraph['context.tokens'])

        for qa in paragraph['qas']:
            for answer in qa['answers']:
                answer['text.tokens'] = map(filter_word, answer['text.tokens'])
            qa['question.tokens'] = map(filter_word, qa['question.tokens'])

        new_spans = []
        for spans in paragraph['spans']:
            new_span = []
            for span in spans:
                if len(span) > stats['max_span']:
                    continue
                new_span.append(span)
            new_spans.append(new_span)
        paragraph['spans'] = new_spans
    return data


if __name__ == '__main__':
    loader = Loader('data/dev-v1.1.json')
    data = merge_paragraphs(loader.data)
    print('data size', len(data))
    # data = data[:1000]
    print('validating')
    validate_answer_start(data)
    print('cleaning data')
    data = clean_paragraphs(data)
    print('tokenizing data')
    data = tokenize_paragraphs(data)
    print(data[0])
    data = extract_constituents_spans(data)
    write_json(data, 'output/dev-v1.1.json')
