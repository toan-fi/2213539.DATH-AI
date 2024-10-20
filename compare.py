from transformers import DistilBertForSequenceClassification, Trainer
from src.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import yaml
import os
import seaborn as sns
from matplotlib import pyplot as plt

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

data_loader = DataLoader(config)
tokenized_dataset = data_loader.get_tokenized_dataset()[1]

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Evaluate the Pre-trained Model
def evaluate_pretrained_model():
    print("\nEvaluating Pre-trained Model (DistilBERT)")
    pretrained_model = DistilBertForSequenceClassification.from_pretrained(config["model"]["name"], num_labels=2)
    pretrained_trainer = Trainer(
        model=pretrained_model,
        eval_dataset=tokenized_dataset,
        compute_metrics=compute_metrics
    )
    # Evaluate pre-trained model
    pretrained_results = pretrained_trainer.evaluate()
    print("Pre-trained Model Performance:", pretrained_results)
    return pretrained_results

# Evaluate the Fine-tuned Model
def evaluate_finetuned_model():
    print("\nEvaluating Fine-tuned Model")
    finetuned_model = DistilBertForSequenceClassification.from_pretrained('./model')
    finetuned_trainer = Trainer(
        model=finetuned_model,
        eval_dataset=tokenized_dataset,
        compute_metrics=compute_metrics
    )
    # Evaluate fine-tuned model
    finetuned_results = finetuned_trainer.evaluate()
    print("Fine-tuned Model Performance:", finetuned_results)
    return finetuned_results

def plot_comparison(pretrained_metrics, finetuned_metrics):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    pre_values = [pretrained_metrics['eval_accuracy'], pretrained_metrics['eval_precision'], pretrained_metrics['eval_recall'], pretrained_metrics['eval_f1']]
    fine_values = [finetuned_metrics['eval_accuracy'], finetuned_metrics['eval_precision'], finetuned_metrics['eval_recall'], finetuned_metrics['eval_f1']]

    bar_width = 0.35
    index = range(len(metrics))

    plt.figure(figsize=(10, 6))
    plt.bar(index, pre_values, bar_width, label='Pre-trained')
    plt.bar([i + bar_width for i in index], fine_values, bar_width, label='Fine-tuned')

    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Pre-trained vs Fine-tuned Model Performance')
    plt.xticks([i + bar_width / 2 for i in index], metrics)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('/home/llm-researcher/Desktop/HCMUT/HK241/DATH/plots', "Pre-trained VS Fine-tuned.png"))
    plt.show()
    plt.close()

def compare_models():
    print("\n==== Model Comparison: Pre-trained vs Fine-tuned ====\n")
    pretrained_metrics = evaluate_pretrained_model()
    finetuned_metrics = evaluate_finetuned_model()
    print("\n==== Comparison Results ====")
    print(
        f"Accuracy: Pre-trained = {pretrained_metrics['eval_accuracy']:.4f}, Fine-tuned = {finetuned_metrics['eval_accuracy']:.4f}")
    print(
        f"F1 Score: Pre-trained = {pretrained_metrics['eval_f1']:.4f}, Fine-tuned = {finetuned_metrics['eval_f1']:.4f}")
    print(
        f"Precision: Pre-trained = {pretrained_metrics['eval_precision']:.4f}, Fine-tuned = {finetuned_metrics['eval_precision']:.4f}")
    print(
        f"Recall: Pre-trained = {pretrained_metrics['eval_recall']:.4f}, Fine-tuned = {finetuned_metrics['eval_recall']:.4f}")

    plot_comparison(pretrained_metrics, finetuned_metrics)

if __name__ == "__main__":
    compare_models()
