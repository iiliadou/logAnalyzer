This folder contains the code to train a BERT model to classify Jenkins logs.
There are no logs provided for testing purposes (due to privacy of the company).
The bert-base-cased folder is needed for tokenization purposes.
The checkpoints folder contains the 2 best models. 
The scripts folder conatins some scripts that can be used on new logs, to mask the result etc.

To create the virtual environment:
	python -m venv log_analyzer_env
To activate the virtual environment:
	log_analyzer_env\Scripts\activate
To install the dependecies:
	pip install -r requirements.txt 
	To train/test a model:
	python main.py	
