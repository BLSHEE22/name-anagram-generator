from transformers import Trainer, TrainingArguments
from model.model import load_model
from model.tokenizer import load_tokenizer
from training.dataset import load_anagram_dataset
from config import *

tokenizer = load_tokenizer()
model = load_model()

dataset = load_anagram_dataset(DB_PATH)

def tokenize(example):
    inputs = tokenizer(example["input"], padding="max_length", truncation=True)
    labels = tokenizer(example["output"], padding="max_length", truncation=True)
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized = dataset.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=15,
    per_device_train_batch_size=4,
    overwrite_output_dir=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"]
)

trainer.train()
model.save_pretrained("./anagram_model")
tokenizer.save_pretrained("./anagram_model")
