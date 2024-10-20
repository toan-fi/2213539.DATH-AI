from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from src.inference import Inference
import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

model = DistilBertForSequenceClassification.from_pretrained('./model')
tokenizer = DistilBertTokenizer.from_pretrained('./model')

inference = Inference(config, model, tokenizer)

if __name__ == "__main__":
    inference.run_inference()