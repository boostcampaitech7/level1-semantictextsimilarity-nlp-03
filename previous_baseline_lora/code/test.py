import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

BASE_MODEL = "tunib/electra-ko-base"

def calculate_accuracy(true_labels, predicted_labels):
    correct_predictions = sum(true_labels == predicted_labels)
    total_predictions = len(true_labels)
    accuracy = correct_predictions / total_predictions
    return accuracy

model = transformers.AutoModel.from_pretrained(BASE_MODEL)
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL)

# Load STS dataset
dataset = load_dataset("klue", "sts")

# Tokenize and encode dataset
def preprocess_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# Split dataset into train and validation sets
train_dataset = encoded_dataset["train"]
validation_dataset = encoded_dataset["validation"]

# Fine-tune the model
model.train_model(train_dataset)

# Evaluate the model on test set
predictions = model.predict(validation_dataset["input_ids"])
predicted_labels = predictions.argmax(axis=1)
true_labels = validation_dataset["label"]

accuracy = calculate_accuracy(true_labels, predicted_labels)
print("Accuracy:", accuracy)

#모델 이어서 학습
# model.load_state_dict(torch.load('model.pt'))
# tensorboard --logdir=./lightning_logs/version_0