import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


class Inference:
    def __init__(self, config, model, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.model = model

    def run_inference(self):
        with open(self.config["inference"]["input_file"], "r") as f:
            reviews = f.readlines()

        with open(self.config["inference"]["output_file"], "w") as f_out:
            for review in reviews:
                review = review.strip()
                if review:  # Only proceed if the line is not empty
                    inputs = self.tokenizer(review, return_tensors='pt', truncation=True, padding=True)
                    with torch.no_grad():
                        outputs = self.model(**inputs)

                    predictions = torch.argmax(outputs.logits, dim=-1)
                    sentiment = "positive" if predictions == 1 else "negative"

                    f_out.write(f"Review: {review}\nPredicted sentiment: {sentiment}\n\n")
                    print(f"Review: {review}\nPredicted sentiment: {sentiment}\n")
