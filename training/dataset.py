from datasets import load_dataset

def load_anagram_dataset(path):
    return load_dataset("csv", data_files=path)

