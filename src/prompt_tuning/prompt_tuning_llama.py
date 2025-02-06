import re
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit
from datasets import Dataset
import pandas as pd
import torch
# Load the LLaMA 3.2 model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"  # Replace with the exact model name
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
# Function to clean text (remove punctuation and unexpected characters)
def clean_text(text):
    cleaned_text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove non-alphanumeric characters
    return cleaned_text.strip()
# Tokenize the dataset with input and labels
def tokenize_function(examples):
    inputs = examples["eng_input"]
    labels = examples["eng_label"]
    # Convert inputs and labels to string, handle missing values, and clean them
    inputs = [clean_text(str(input_)) if input_ is not None else "" for input_ in inputs]
    labels = [clean_text(str(label)) if label is not None else "" for label in labels]
    # Tokenize inputs and labels
    tokenized_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)  # LLaMA supports larger contexts
    tokenized_labels = tokenizer(labels, truncation=True, padding="max_length", max_length=512)
    # Add the labels to the tokenized inputs
    tokenized_inputs["labels"] = tokenized_labels["input_ids"]
    return tokenized_inputs
# Load dataset from CSV file
df = pd.read_csv("ynet_175k_matches.csv")
dataset = Dataset.from_pandas(df)
# Map the tokenization function to the dataset
dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
# Configure prompt tuning
tuning_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text="Complete the following sentence and make sure it translates well into Hebrew",  # Meaningful initialization text
    num_virtual_tokens=32,  # Start with fewer virtual tokens
    tokenizer_name_or_path=model_name
)
# Define the PEFT model
peft_model = get_peft_model(model, tuning_config)
# Set up training arguments
training_args = TrainingArguments(
    output_dir="./llama_word_gen_model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=0.001,  # Lower learning rate
    num_train_epochs=10,
    save_steps=50,
    logging_steps=10,
    fp16=torch.cuda.is_available(),
    optim="adamw_torch"
)
# Create trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
# Train the model
trainer.train()
# Save the trained model and tokenizer
output_model_dir = "./llama_model_176k"
peft_model.save_pretrained(output_model_dir)
tokenizer.save_pretrained(output_model_dir)
print(f"Model and tokenizer saved to {output_model_dir}")

