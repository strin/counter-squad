# encoding: utf-8
from __future__ import print_function
from data import *

# test basic data processing pipeline.
def test_clean_text():
    assert(clean_text(u'Carolina Panthers 24–10')
        == 'carolina panthers 24-10'
    )


def test_clean_answer():
    assert(clean_answer(' Santa Clara, California...')
        == 'santa clara, california'
    )


def test_constituents_in_tree():
    results = constituents_in_tree('(ROOT (S (VP (NP (INTJ (UH Hello)) (NP (NN world)))) (. !)))')
    assert(results == [['Hello'], ['world'], ['Hello', 'world'], ['!'], ['Hello', 'world', '!']])
    results = constituents_in_tree('(ROOT (S (NP (PRP It)) (VP (VBZ is) (ADJP (RB so) (JJ beautiful))) (. .)))')
    print(results)
    assert(results == [['It'], ['is'], ['so'], ['beautiful'], ['so', 'beautiful'], ['is', 'so', 'beautiful'], ['.'], ['It', 'is', 'so', 'beautiful', '.']])
