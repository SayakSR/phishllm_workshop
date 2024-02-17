import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support

phishing_df = pd.read_csv('phishing_full.csv')
benign_df = pd.read_csv('benign_full.csv')
phishing_df['label'] = 0
benign_df['label'] = 1

df = pd.concat([phishing_df, benign_df], ignore_index=True)

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_texts = train_df['text'].tolist()
train_labels = train_df['label'].tolist()
test_texts = test_df['text'].tolist()
test_labels = test_df['label'].tolist()

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Training Arguments
args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    gradient_accumulation_steps=4
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# Evaluate model
predictions = trainer.predict(test_dataset)
accuracy = (predictions.predictions.argmax(-1) == test_labels).mean()  
precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions.predictions.argmax(-1), average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}") 
print(f"Recall: {recall}")
print(f"F1 Score: {f1}") 

model.save_pretrained('./html_structure_model')

print('Model saved')
