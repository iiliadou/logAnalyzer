import re
import codecs
import os

# Class that performs the cleaning of the logs by using regexes.
class LogPreprocessor:
    def __init__(self, log_path):
        self.log_path = log_path
        self.preamble_str = "\x1B[8mha:" 
        self.postamble_str = "\x1B[0m"
        self.pattern_encoded = re.escape(self.preamble_str) + r".*?" + re.escape(self.postamble_str)
        self.regex_commit_pattern = "^(?:Checking out Revision|Commit message|Fetching(?: upstream)? changes)"
        self.regex_log_pattern = "(?:\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z\]\s+)?\[(?:ERROR|WARN|INFO|FATAL|WARNING)\]\s+\S.*"
        self.regex_filepath_pattern = "(?:[a-zA-Z]:)?(?:[\\/][\w\-. ]+)*(?:[\\/][\w\-. ]+(?:\.[a-zA-Z]+)?)?"
        self.replace_pattern = r"(C:\\[^ ]+)"
        self.replace_patt = r"(.\\[^ ]+)"
        self.sensitive_info_pattern = r"\b(?:password|user)\b"

    def clean_log(self, line):
        line = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", "at time", line)
        line = re.sub(r"[^a-zA-Z0-9\[\]\s\n]", " ", line)
        line = re.sub(r"\d+", "", line)
        return re.sub(r"\s{2,}", " ", line)

    def preprocess_logs(self):
        list_of_logs = []

        for filename in os.listdir(self.log_path):
            full_path = os.path.join(self.log_path, filename)
            if os.path.isfile(full_path):
                try:
                    with codecs.open(full_path, "r") as f:
                        content = f.readlines()
                        log_messages = ""
                        for line in content:
                            # Hide pipeline commands
                            if self.preamble_str in line and self.postamble_str in line:
                                line = re.sub(self.pattern_encoded, "", line)
                            # Hide sensitive information
                            if re.search(self.sensitive_info_pattern, line, re.IGNORECASE):
                                line = re.sub(self.sensitive_info_pattern, "", line)
                            if re.match(self.regex_log_pattern, line) or re.match(self.regex_commit_pattern, line):
                                if re.match(self.regex_filepath_pattern, line):
                                    line = re.sub(self.replace_patt, "filepath", line)
                                line = self.clean_log(line)
                                if "\n" not in line:
                                    line += "\n"
                                log_messages += line
                        list_of_logs.append(log_messages)
                except FileNotFoundError:
                    print(f"File not found: {full_path}")
                except Exception as e:
                    print(f"Error reading file: {full_path}\n{str(e)}")
                    
        return list_of_logs
    
    def write_list_to_file(file_path, content_list):
        with open(file_path, 'w') as file:
            for item in content_list:
                file.write(str(item) + '\n')
        print(f"The content of the list has been written to the file: {file_path}")
