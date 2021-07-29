# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import sys
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score


def read_answers(filename):
    answers = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            idx1, label = line.split()
            answers[(idx1)] = label
    return answers


def read_predictions(filenames):
    predictions = []
    for filename in filenames:
        prediction = {}
        with open(filename) as f:
            for line in f:
                line = line.strip()
                idx1, label = line.split()
                if 'txt' in line:
                    idx1 = idx1.split('/')[-1][:-4]
                prediction[(idx1)] = label
        predictions.append(prediction)
    return predictions


def calculate_scores(answers, predictions):
    scores = []
    for prediction in predictions:
        y_trues, y_preds = [], []
        for key in answers:
            if key not in prediction:
                logging.error(
                    "Missing prediction for ({},{}) pair.".format(key[0], key[1]))
                sys.exit()
            y_trues.append(answers[key])
            y_preds.append(prediction[key])
        score = {}
        score['Recall'] = recall_score(
            y_trues, y_preds, average='macro')
        score['Prediction'] = precision_score(
            y_trues, y_preds, average='macro')
        score['F1'] = f1_score(
            y_trues, y_preds, average='macro')
        score["Accuracy"] = accuracy_score(y_trues, y_preds)
        scores.append(score)
    return scores


def evaluate(answers_file, predictions_files):
    answers = read_answers(answers_file)
    predictions = read_predictions(predictions_files)
    scores = calculate_scores(answers, predictions)
    for i in range(len(scores)):
        print(predictions_files[i])
        print(scores[i])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Evaluate leaderboard predictions for smart contract dataset.')
    parser.add_argument('--answers', '-a',
                        help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', '-p', nargs='+',
                        help="filenames of the leaderboard predictions, in txt format.")
    args = parser.parse_args()
    evaluate(args.answers, args.predictions)
