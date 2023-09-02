import pandas as pd
import nltk
import json
from sklearn.utils import shuffle
from LogPreprocessor import LogPreprocessor
nltk.download('punkt')

# This class brings the data to the appropriate form (dataframe objects), it also calls the class LogPreprocessor that cleans the data.
class DataPreparation:

    def __init__(self):
        # Read paths from config file. 
        with open("config.json") as config_file:
            config = json.load(config_file)

        self.input_dir_succ = config["path_succesful_logs"]
        self.input_dir_fail = config["path_failed_logs"]
        self.input_dir_unst = config["path_unstable_logs"]

    def process_data(self):
        successful_logs = self.performTransformations(self.input_dir_succ)
        failed_logs = self.performTransformations(self.input_dir_fail)
        unstable_logs = self.performTransformations(self.input_dir_unst)


        # Combine the list of logs
        all_logs = successful_logs + failed_logs + unstable_logs

        # Create the DataFrame
        df = pd.DataFrame({"category": ["Successful"] * len(successful_logs) +
                                       ["Failed"] * len(failed_logs) +
                                       ["Unstable"] * len(unstable_logs),
                           "text": all_logs})
        
        # Shuffle the DataFrame
        shuffled_df = shuffle(df)

        return shuffled_df
    
    def performTransformations(self, input_dir):
        preprocessor_logs = LogPreprocessor(input_dir)
        logs = preprocessor_logs.preprocess_logs()
        tokens = self.tokenize_sentences(logs)
        # Log should be 512 length
        logs = self.add_tokens_to_logs(tokens)
        return logs

    def tokenize_sentences(self, sentences):
        # Tokenize the sentences into words
        tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
        return tokenized_sentences
    
    def add_tokens_to_logs(self, token_list): 
        logs = []
        log_line = []
        log = ""
        for tokenized_sentence in token_list:
            for token in tokenized_sentence:
                if len(log_line) < 510:
                    log_line.append(token)
                if len(log_line) >= 510:
                    log = " ".join(log_line)
                    logs.append(log)
                    log = ""
                    log_line.clear()
        return logs

    def write_to_csv(self, df, file_path):
        df.to_csv(file_path, index=False)
