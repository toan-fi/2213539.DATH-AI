from transformers import DistilBertForSequenceClassification

class ModelLoader:
    def __init__(self, config):
        self.config = config

    def load_model(self, fine_tuned=False):
        model_name = './model' if fine_tuned else self.config["model"]["name"]
        model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        return model
