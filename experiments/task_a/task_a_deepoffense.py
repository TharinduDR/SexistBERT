import argparse
import numpy as np
import os
import pandas as pd
import shutil
import sklearn
import torch
from sklearn.model_selection import train_test_split

from deepoffense.classification import ClassificationModel
from deepoffense.util.evaluation import macro_f1, weighted_f1
from deepoffense.util.label_converter import decode, encode
from experiments.task_a.deepoffense_config import TEMP_DIRECTORY, args, SEED, RESULT_FILE, SUBMISSION_FILE

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="bert-large-cased")
parser.add_argument('--model_type', required=False, help='model type', default="bert")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)
parser.add_argument('--folds', required=False, help='numbner of folds', default=1)
arguments = parser.parse_args()

folds = int(arguments.folds)

train = pd.read_csv("data/train_all_tasks.csv")
test = pd.read_csv("data/dev_task_a_entries.csv")
blind_test = pd.read_csv("data/test_task_a_entries.csv")

train = train.rename(columns={'label_sexist': 'labels'})

# load training data
train = train[['text', 'labels']]

train['labels'] = encode(train["labels"])

test_sentences = test['text'].tolist()
blind_test_sentences = blind_test['text'].tolist()

test_preds = np.zeros((len(test_sentences), folds))
blind_test_preds = np.zeros((len(blind_test_sentences), folds))

for fold in range(folds):
    MODEL_TYPE = arguments.model_type
    MODEL_NAME = arguments.model_name
    cuda_device = int(arguments.cuda_device)

    if os.path.exists(args['output_dir']) and os.path.isdir(args['output_dir']):
        shutil.rmtree(args['output_dir'])
    torch.cuda.set_device(cuda_device)
    model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=args,
                                use_cuda=torch.cuda.is_available(),
                                cuda_device=cuda_device)

    train = train.sample(frac=1, random_state=SEED * fold).reset_index(drop=True)
    train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * fold)
    model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1,
                      accuracy=sklearn.metrics.accuracy_score)
    predictions, raw_outputs = model.predict(test_sentences)
    test_preds[:, fold] = predictions
    blind_test_predictions, blind_test_raw_outputs = model.predict(blind_test_sentences)
    blind_test_preds[:, fold] = blind_test_predictions

final_predictions = []
for row in test_preds:
    row = row.tolist()
    final_predictions.append(int(max(set(row), key=row.count)))

test["label_pred"] = final_predictions
test['label_pred'] = decode(test['label_pred'])
test = test[['rewire_id', 'label_pred']]
test.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, index=False, encoding='utf-8')

blind_final_predictions = []
for row in blind_test_preds:
    row = row.tolist()
    blind_final_predictions.append(int(max(set(row), key=row.count)))

blind_test["label_pred"] = blind_final_predictions
blind_test['label_pred'] = decode(blind_test['label_pred'])
blind_test = blind_test[['rewire_id', 'label_pred']]
blind_test.to_csv(os.path.join(TEMP_DIRECTORY, SUBMISSION_FILE), header=True, index=False, encoding='utf-8')

# with open('data/gab_1M_unlabelled.csv') as f:
#     gab_lines = f.read().splitlines()
#
# gab_lines.pop(0)
# gap_predictions, gap_raw_outputs = model.predict(gab_lines)
# probabilities = softmax(gap_raw_outputs, axis=1)
# prediction_probabolities = list(zip(*probabilities))[0]
#
# with open('predictions_roberta_gab.txt', 'w') as f:
#     # write each integer to the file on a new line
#     for number in prediction_probabolities:
#         f.write(str(number) + '\n')
#
# with open('data/reddit_1M_unlabelled.csv') as f:
#     reddit_lines = f.read().splitlines()
#
# reddit_lines.pop(0)
# reddit_predictions, reddit_raw_outputs = model.predict(reddit_lines)
# probabilities = softmax(reddit_raw_outputs, axis=1)
# prediction_probabolities = list(zip(*probabilities))[0]
#
# with open('predictions_roberta_reddit.txt', 'w') as f:
#     # write each integer to the file on a new line
#     for number in prediction_probabolities:
#         f.write(str(number) + '\n')
#
# def split(a, n):
#     k, m = divmod(len(a), n)
#     return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
#
# solid = Dataset.to_pandas(load_dataset('tharindu/SOLID', split='train'))
# solid_sentences = solid["text"].to_list()
#
#
# probability_predictions = []
# test_batches = split(solid_sentences, 10)
# for index, break_list in enumerate(test_batches):
#     print("Length of the break lists", len(break_list))
#     temp_solid_predictions, temp_solid_raw_outputs = model.predict(break_list)
#     temp_probability_predictions = []
#     for output in temp_solid_raw_outputs:
#         weights = softmax(output)
#         temp_probability_predictions.append(weights)
#     probability_predictions.extend(temp_probability_predictions)
#
# with open('predictions_roberta_solid.txt', 'w') as f:
#     # write each integer to the file on a new line
#     for number in probability_predictions:
#         f.write(str(number) + '\n')