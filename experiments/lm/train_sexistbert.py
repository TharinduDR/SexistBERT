import shutil
import torch
import argparse

from deepoffense.classification.transformer_models.args.model_args import LanguageModelingArgs
from deepoffense.language_modeling.language_modeling_model import LanguageModelingModel


parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="roberta-large")
parser.add_argument('--model_type', required=False, help='model type', default="roberta")
parser.add_argument('--cuda_device', required=False, help='cuda device', default=0)
arguments = parser.parse_args()

with open('output_file.txt','wb') as wfd:
    for f in ['data/gab_1M_unlabelled.csv','data/reddit_1M_unlabelled.csv']:
        with open(f,'rb') as fd:
            next(fd)
            shutil.copyfileobj(fd, wfd)

with open('output_file.txt') as f:
    lines = f.read().splitlines()

train_lines = lines[:int(len(lines)*.8)]
test_lines = lines[int(len(lines)*.8):len(lines)]

with open('train.txt', 'w') as f:
    # write each integer to the file on a new line
    for line in train_lines:
        f.write(str(line) + '\n')

with open('test.txt', 'w') as f:
    # write each integer to the file on a new line
    for line in test_lines:
        f.write(str(line) + '\n')


model_args = LanguageModelingArgs()
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.num_train_epochs = 50
model_args.dataset_type = "simple"
model_args.train_batch_size = 16
model_args.eval_batch_size = 32
model_args.learning_rate = 6e-5
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 30000
model_args.save_eval_checkpoints = True
model_args.save_best_model = True
model_args.save_recent_only = True
model_args.wandb_project = "LM"
model_args.use_multiprocessing = False
model_args.use_multiprocessing_for_evaluation = False


MODEL_TYPE = arguments.model_type
MODEL_NAME = arguments.model_name
cuda_device = int(arguments.cuda_device)

train_file = "train.txt"
test_file = "test.txt"

model = LanguageModelingModel(MODEL_TYPE, MODEL_NAME, args=model_args,
                            use_cuda=torch.cuda.is_available(),
                            cuda_device=cuda_device)

# Train the model
model.train_model(train_file, eval_file=test_file)

# Evaluate the model
result = model.eval_model(test_file)

