
import os
import random
import pickle
from transformers import BertTokenizer
from LogPreprocessor import LogPreprocessor

class LogTokenizer:
    def __init__(self):
        os.environ['http_proxy'] = 'http://http-proxy-de02.dslocal.com:80'
        os.environ['https_proxy'] = 'http://http-proxy-de02.dslocal.com:80'
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_token_limit = 512

    def tokenize_logs(self, logs):
        while logs:
            log = logs.pop(0)
            if len(log) <= self.max_token_limit:
                yield self.tokenizer.encode(log, truncation=True, padding=True, max_length=self.max_token_limit)
            else:
                start = 0
                while start < len(log):
                    chunk = log[start: start + self.max_token_limit]
                    yield self.tokenizer.encode(chunk, truncation=True, padding=True, max_length=self.max_token_limit)
                    start += self.max_token_limit

    def process_logs(self, input_dir_succ, input_dir_fail, input_dir_unst, train_encodings_file, test_encodings_file):
        preprocessor_successful = LogPreprocessor(input_dir_succ)
        successful_logs = preprocessor_successful.preprocess_logs()
        preprocessor_failed = LogPreprocessor(input_dir_fail)
        failed_logs = preprocessor_failed.preprocess_logs()
        preprocessor_unstable = LogPreprocessor(input_dir_unst)
        unstable_logs = preprocessor_unstable.preprocess_logs()

        random.shuffle(failed_logs)
        random.shuffle(successful_logs)
        random.shuffle(unstable_logs)

        train_logs = failed_logs[:190] + successful_logs[:190] + unstable_logs[:190]
        train_labels = ['failed'] * 190 + ['successful'] * 190 + ['unstable'] * 190

        test_logs = failed_logs[190:240] + successful_logs[190:240] + unstable_logs[190:240]
        test_labels = ['failed'] * 50 + ['successful'] * 50 + ['unstable'] * 50

        del failed_logs
        del successful_logs
        del unstable_logs

        train_encodings = list(self.tokenize_logs(train_logs))
        test_encodings = list(self.tokenize_logs(test_logs))

        # Save train and test encodings to files
        with open(train_encodings_file, 'wb') as f:
            pickle.dump(train_encodings, f)

        with open(test_encodings_file, 'wb') as f:
            pickle.dump(test_encodings, f)
        
        print("len of train labels is: ")
        print(len(train_labels))
        return train_labels, test_labels
