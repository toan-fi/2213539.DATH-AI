from datasets import load_dataset
from transformers import DistilBertTokenizer

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.tokenizer = DistilBertTokenizer.from_pretrained(config["model"]["name"])

    def load_data(self):
        dataset = load_dataset(self.config["dataset"]["name"])
        return dataset[self.config["dataset"]["train_split"]], dataset[self.config["dataset"]["test_split"]]

    def preprocess(self, examples):
        return self.tokenizer(examples['text'], truncation=True, padding=True, max_length=self.config["dataset"]["max_seq_length"])

    def get_tokenized_dataset(self):
        train_data, test_data = self.load_data()
        tokenized_train = train_data.map(self.preprocess, batched=True)
        tokenized_test = test_data.map(self.preprocess, batched=True)
        return tokenized_train, tokenized_test
