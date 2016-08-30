from evaluate import Evaluator

def error_examples(data, predictions):
    evaluator = Evaluator(data)
    scores = {}
    context_by_qid = {}
    qa_by_qid = {}

    for paragraph in data:
        for qa in paragraph['qas']:
            context_by_qid[qa['id']] = paragraph['context']
            qa_by_qid[qa['id']] = qa

    for (question_id, predicted_answer) in predictions.items():
        score = evaluator.F1Single(question_id, predicted_answer)
        scores[question_id] = score

    errors = sorted(scores.items(), key=lambda item: item[1])
    errors = [{
        'id': question_id,
        'context': context_by_qid[question_id],
        'qa': qa_by_qid[question_id],
        'prediction': predictions[question_id],
        'f1': score
    } for (question_id, score) in errors if score != 1.0]
    return errors

