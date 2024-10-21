from transformers import Trainer, TrainingArguments, TrainerCallback
from src.utils import setup_logging, save_checkpoint
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import csv, yaml
import os
import random
from torch.utils.data import Subset

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

def compute_metrics(pred):
    predictions, labels = pred
    preds = predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def get_eval_subset(dataset, fraction=0.01):
    dataset_size = len(dataset)
    subset_size = int(dataset_size * fraction)
    subset_indices = random.sample(range(dataset_size), subset_size)
    return Subset(dataset, subset_indices)

class MetricsLogger(TrainerCallback):
    def __init__(self, log_file='./logs/training_log.csv', plots_dir='./plots'):
        self.log_file = log_file
        self.plots_dir = plots_dir

        self.current_loss = None
        self.current_metrics = {"accuracy": None, "precision": None, "recall": None, "f1": None}

        if not os.path.exists(os.path.dirname(self.log_file)):
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)

        with open(self.log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["step", "loss", "accuracy", "precision", "recall", "f1"])

    def log_to_csv(self, state):
        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                state.global_step,
                self.current_loss,
                self.current_metrics["accuracy"],
                self.current_metrics["precision"],
                self.current_metrics["recall"],
                self.current_metrics["f1"]
            ])
        self.current_loss = None
        self.current_metrics = {"accuracy": None, "precision": None, "recall": None, "f1": None}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.current_loss = logs.get("loss")
            # if self.current_metrics["f1"] is not None:
            #     self.log_to_csv(state)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            self.current_metrics["accuracy"] = metrics.get("eval_accuracy", None)
            self.current_metrics["precision"] = metrics.get("eval_precision", None)
            self.current_metrics["recall"] = metrics.get("eval_recall", None)
            self.current_metrics["f1"] = metrics.get("eval_f1", None)

            # If loss is already available, log them all together
            if self.current_loss is not None:
                self.log_to_csv(state)

    def plot_metrics(self):
        df = pd.read_csv(self.log_file)

        # Plot training loss over steps
        plt.figure(figsize=(14, 8))
        plt.plot(df['step'], df['loss'], label="Loss", marker='o', color='r')
        plt.title("Training Loss Over Steps")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, "training loss over time.png"))
        plt.show()
        plt.close()

        # Plot evaluation metrics over steps
        plt.figure(figsize=(14, 8))
        plt.plot(df['step'], df['accuracy'], label="Accuracy", marker='o')
        plt.plot(df['step'], df['precision'], label="Precision", marker='o')
        plt.plot(df['step'], df['recall'], label="Recall", marker='o')
        plt.plot(df['step'], df['f1'], label="F1 Score", marker='o')
        plt.title("Evaluation Metrics Over Steps")
        plt.xlabel("Steps")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, "evaluation metrics over time.png"))
        plt.show()
        plt.close()


class CustomTrainer:
    def __init__(self, model, train_data, eval_data, config):
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self.config = config
        setup_logging(self.config["train"]["logging_dir"])
        self.metrics_logger = MetricsLogger(log_file='./logs/training_log.csv')

        self.eval_subset = get_eval_subset(eval_data)

    def get_training_args(self):
        return TrainingArguments(
            output_dir='./model',
            evaluation_strategy="steps",  # Evaluate after a specific number of steps
            eval_steps=100,               # Evaluate every 100 steps
            learning_rate=self.config["train"]["learning_rate"],
            per_device_train_batch_size=4,
            num_train_epochs=self.config["train"]["epochs"],
            weight_decay=self.config["train"]["weight_decay"],
            logging_dir='./logs',
            logging_steps=100,             # Log every 10 steps (loss)
            save_steps=1000,
            save_total_limit=3,
        )

    def train(self):
        trainer = Trainer(
            model=self.model,
            args=self.get_training_args(),
            train_dataset=self.train_data,
            eval_dataset=self.eval_subset,  # Use subset for evaluation during training
            compute_metrics=compute_metrics,  # Compute metrics after evaluation
            callbacks=[self.metrics_logger]   # Add the custom metrics logger callback
        )
        trainer.train()

        # Save the model
        save_checkpoint(self.model, self.config["model"]["checkpoint_dir"])

        # Plot loss and metrics after training is complete
        self.metrics_logger.plot_metrics()

        # Perform full evaluation on the entire dataset at the end
        self.full_evaluation(trainer)

    def full_evaluation(self, trainer):
        print("Running full evaluation on the entire validation set...")
        full_eval_results = trainer.evaluate(self.eval_data)

        # Print full evaluation metrics
        print("Final Evaluation on Full Validation Set:", full_eval_results)

        # Generate confusion matrix after full evaluation
        self.plot_confusion_matrix(trainer)

    def plot_confusion_matrix(self, trainer):
        # Get model predictions on the full evaluation set
        predictions = trainer.predict(self.eval_data)
        preds = predictions.predictions.argmax(-1)
        true_labels = self.eval_data["label"]

        # Compute confusion matrix
        conf_matrix = confusion_matrix(true_labels, preds)

        # Plot confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Negative", "Positive"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(self.metrics_logger.plots_dir, "confusion matrix.png"))
        plt.show()
        plt.close()
