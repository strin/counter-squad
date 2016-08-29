'''
Official evaluation script for the SQuAD dataset. Version 1.
'''

from collections import Counter
import json
import sys

class Evaluator(object):
    def __init__(self, data):
        self._answers = {}
        for paragraph in data:
            for qa in paragraph['qas']:
                answers = self._answers[qa['id']] = []
                for answer in qa['answers']:
                    answers.append(answer['text'])

    def ExactMatch(self, predictions):
        """
        Accepts a dict from question ID to predicted answer.
        """
        total = 0
        for question_id in self._answers.iterkeys():
            predicted_answer = predictions.get(question_id)
            if predicted_answer is not None and self.ExactMatchSingle(question_id, predicted_answer):
                total += 1

        return 100.0 * total / len(self._answers)

    def ExactMatchSingle(self, question_id, predicted_answer):
        predicted_answer = Evaluator.CleanAnswer(predicted_answer)
        for answer in self._answers[question_id]:
            if Evaluator.CleanAnswer(answer) == predicted_answer:
                return True
        return False

    WHITESPACE_AND_PUNCTUATION = set([' ', '.', ',', ':', ';', '!', '?', '$', '%', '(', ')', '[', ']', '-', '`', '\'', '"'])
    ARTICLES = set(['the', 'a', 'an'])

    @staticmethod
    def CleanAnswer(answer):
        answer = answer.lower()
        if isinstance(answer, unicode):
            answer = answer.replace(u'\u00a0', ' ')
        else:
            answer = answer.replace('\xc2\xa0', ' ')
        while len(answer) > 1 and answer[0] in Evaluator.WHITESPACE_AND_PUNCTUATION:
            answer = answer[1:]
        while len(answer) > 1 and answer[-1] in Evaluator.WHITESPACE_AND_PUNCTUATION:
            answer = answer[:-1]

        answer = answer.split()
        if len(answer) > 1 and answer[0] in Evaluator.ARTICLES:
            answer = answer[1:]
        answer = ' '.join(answer)

        return answer

    def F1(self, predictions):
        """
        Accepts a dict from question ID to predicted answer.
        """
        total = 0
        for question_id in self._answers.iterkeys():
            predicted_answer = predictions.get(question_id)
            if predicted_answer is not None:
                total += self.F1Single(question_id, predicted_answer)

        return 100.0 * total / len(self._answers)

    def F1Single(self, question_id, predicted_answer):
        def GetTokens(text):
            text = Evaluator.CleanAnswer(text)
            for delimeter in Evaluator.WHITESPACE_AND_PUNCTUATION:
                text = text.replace(delimeter, ' ')
            return text.split()

        f1 = 0
        predicted_answer_tokens = Counter(GetTokens(predicted_answer))
        num_predicted_answer_tokens = sum(predicted_answer_tokens.values())
        for answer in self._answers[question_id]:
            answer_tokens = Counter(GetTokens(answer))
            num_answer_tokens = sum(answer_tokens.values())
            num_same = sum((predicted_answer_tokens & answer_tokens).values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / num_predicted_answer_tokens
            recall = 1.0 * num_same / num_answer_tokens
            f1 = max(2 * precision * recall / (precision + recall), f1)

        return f1

