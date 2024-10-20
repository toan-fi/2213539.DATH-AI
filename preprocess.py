import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
from src.data import DataLoader

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

data_loader = DataLoader(config)
train_dataset, test_dataset = data_loader.load_data()

train_df = pd.DataFrame(train_dataset)
print("Sample data:\n", train_df.head())

# Visualize
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=train_df)
plt.title('Label Distribution in IMDb Dataset')
plt.xlabel('Sentiment (0 = Negative, 1 = Positive)')
plt.ylabel('Count')
plt.savefig(os.path.join('/home/llm-researcher/Desktop/HCMUT/HK241/DATH-AI/plots', "dataset label distribution.png"))
plt.show()
plt.close()


train_df['review_length'] = train_df['text'].apply(len)

plt.figure(figsize=(8, 6))
sns.histplot(train_df['review_length'], bins=30, kde=True)
plt.title('Distribution of Review Lengths')
plt.xlabel('Number of Characters in Review')
plt.ylabel('Count')
plt.savefig(os.path.join('/home/llm-researcher/Desktop/HCMUT/HK241/DATH-AI/plots', "dataset review length distribution.png"))
plt.show()
plt.close()


# Tokenize the dataset
print("\nRaw Review Sample:")
print(train_dataset[0]["text"])

print("\nTokenized Review Sample:")
print(data_loader.preprocess(train_dataset[0]))

train_tokenized = train_dataset.map(data_loader.preprocess, batched=True)
train_tokenized.save_to_disk("./data/train_tokenized")
