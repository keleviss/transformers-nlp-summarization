import torch
import evaluate
import numpy as np
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from datasets import load_dataset, DatasetDict
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

dataset = load_dataset("multi_news")

dataset['train'] = dataset['train'].shuffle(seed=42).select(range(10000))
dataset['validation'] = dataset['validation'].shuffle(seed=42).select(range(500))
dataset['test'] = dataset['test'].shuffle(seed=42).select(range(1000))

def show_samples(dataset, num_samples=3, seed=42):
    sample = dataset['train'].shuffle(seed=seed).select(range(num_samples))
    for example in sample:
        print(f"\n'>> Summary: {example['summary']}'")
        print(f"'>> Document: {example['document']}'")

show_samples(dataset)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# GOOGLE/MT5-SMALL
# from transformers import MT5Tokenizer, MT5ForConditionalGeneration
# model_checkpoint = "google/mt5-small"
# tokenizer = MT5Tokenizer.from_pretrained(model_checkpoint)
# model = MT5ForConditionalGeneration.from_pretrained(model_checkpoint)

#FACEBOOK/BART-BASE
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_checkpoint = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

model = model.to(device)

max_input_length = 512
max_target_length = 30

# Tokenization function
def preprocess_function(examples):
    model_inputs = tokenizer(examples["document"], max_length=max_input_length, truncation=True,)
    labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = DatasetDict()

# Tokenize each split of the dataset
for split_name in dataset.keys():
    split = dataset[split_name]
    tokenized_split = split.map(preprocess_function, batched=True)
    tokenized_dataset[split_name] = tokenized_split

# Verify the tokenized dataset
print("Tokenized documents in the train split:", tokenized_dataset['train']['input_ids'][:5])
print("Tokenized summaries in the train split:", tokenized_dataset['train']['labels'][:5])

print(tokenized_dataset)

rouge_score = evaluate.load("rouge")

batch_size = 8
num_train_epochs = 1

# Show the training loss with every epoch
logging_steps = len(tokenized_dataset["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-multi-news",
    evaluation_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_steps=logging_steps,
    push_to_hub=True,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    rouge_dict = dict((rn, round(result[rn] * 100, 2)) for rn in rouge_names)  
    return rouge_dict

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

tokenized_dataset = tokenized_dataset.remove_columns(
    dataset["train"].column_names
)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.push_to_hub(commit_message="Training complete", tags="summarization")
