import yaml
from transformers import DistilBertTokenizer
from src.data import DataLoader
from src.model import ModelLoader
from src.trainer import CustomTrainer

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

data_loader = DataLoader(config)
train_data, eval_data = data_loader.get_tokenized_dataset()

model_loader = ModelLoader(config)
model = model_loader.load_model(fine_tuned=False)

trainer = CustomTrainer(model=model, train_data=train_data, eval_data=eval_data, config=config)
trainer.train()

model.save_pretrained('./model')
tokenizer = DistilBertTokenizer.from_pretrained(config["model"]["name"])
tokenizer.save_pretrained('./model')
