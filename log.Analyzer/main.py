from DataPreparation import DataPreparation
from transformers import BertTokenizer
import numpy as np
from Dataset import Dataset
from BertClassifier import BertClassifier
from Trainer import Trainer
from Evaluator import Evaluator
import torch 
import pandas as pd
import json

# Load hardcoded paths.
with open("config.json") as config_file:
            config = json.load(config_file)

# Create a new dataframe from data
data_prep = DataPreparation()
df = data_prep.process_data()

# Save the dataframe to a file for later use
#data_prep.write_to_csv(df, 'dataTraining.csv')

# OR

# # Import the dataset from a csv file
# df = pd.read_csv('dataTrainingFirst100.csv')

# Get the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
labels = {'Successful':0,
          'Failed':1,
          'Unstable':2
          }

# Split data while creating a seed.
np.random.seed(112)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), 
                                     [int(.8*len(df)), int(.9*len(df))])

# # Training
EPOCHS = 5
model = BertClassifier()
LR = 1e-6

# Bring the data to an appropriate form.
train_dataset = Dataset(df_train, labels, tokenizer)
val_dataset = Dataset(df_val, labels, tokenizer)

# Create a Trainer and train
trainer = Trainer(model, train_dataset, val_dataset, LR, EPOCHS)
trainer.train()

# Save the whole model
save_directory = config['save_directory']
model.save_model(save_directory)


#Evaluate the current model
# test_dataset = Dataset(df_test, labels, tokenizer)
# evaluator = Evaluator(model, test_dataset)
# evaluator.evaluate()

# OR 

# Evaluate an old model from a checkpoint 
# Load the model from a checkpoint
# checkpoint_path = config["checkpoint_path"]
# checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
# model_state_dict = checkpoint['model_state_dict']

# model = BertClassifier()
# model.load_state_dict(model_state_dict)

# # Create the test dataset
# test_dataset = Dataset(df_test, labels, tokenizer)

# # Instantiate the evaluator with the loaded model and test dataset
# evaluator = Evaluator(model, test_dataset)

# # Call the evaluate method
# evaluator.evaluate()
