import argparse
import os
import pandas as pd
import shutil
import sklearn
import statistics
import torch
from sklearn.model_selection import train_test_split

from deepoffense.classification import ClassificationModel
from deepoffense.util.evaluation import macro_f1, weighted_f1
from deepoffense.util.label_converter import decode, encode
from experiments.task_b.deepoffense_config import TEMP_DIRECTORY, args, SEED, RESULT_FILE

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="bert-large-cased")
parser.add_argument('--model_type', required=False, help='model type', default="bert")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)
arguments = parser.parse_args()

train = pd.read_csv("data/train_all_tasks.csv")
test = pd.read_csv("data/dev_task_b_entries.csv")

train = train.rename(columns={'label_category': 'labels'})

# load training data
train = train[['text', 'labels']]
train = train[train['labels'].notna()]

train['labels'] = encode(train["labels"])

test_sentences = test['text'].tolist()

MODEL_TYPE = arguments.model_type
MODEL_NAME = arguments.model_name
cuda_device = int(arguments.cuda_device)

if os.path.exists(args['output_dir']) and os.path.isdir(args['output_dir']):
    shutil.rmtree(args['output_dir'])
torch.cuda.set_device(cuda_device)
model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=args, num_labels=4,
                            use_cuda=torch.cuda.is_available(),
                            cuda_device=cuda_device)

train = train.sample(frac=1, random_state=SEED).reset_index(drop=True)
train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED)
model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
predictions, raw_outputs = model.predict(test_sentences)

test["label_pred"] = predictions
test['label_pred'] = decode(test['label_pred'])
test = test[['rewire_id', 'label_pred']]
test.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, index=False, encoding='utf-8')






