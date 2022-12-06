import shutil

from deepoffense.classification.transformer_models.args.model_args import LanguageModelingArgs
from deepoffense.language_modeling.language_modeling_model import LanguageModelingModel

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
model_args.num_train_epochs = 25
model_args.dataset_type = "simple"
model_args.train_batch_size = 32
model_args.eval_batch_size = 32
model_args.learning_rate = 2e-5
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 30000
model_args.no_save = True


train_file = "train.txt"
test_file = "test.txt"

model = LanguageModelingModel(
    "roberta", "roberta-large", args=model_args
)

# Train the model
model.train_model(train_file, eval_file=test_file)

# Evaluate the model
result = model.eval_model(test_file)

