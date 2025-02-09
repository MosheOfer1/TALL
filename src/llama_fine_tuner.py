import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from dataclasses import dataclass
from typing import Optional, Dict, Sequence, Union
import torch


@dataclass
class DataCollatorForDynamicBatching:
    tokenizer: AutoTokenizer
    max_length: int

    def __call__(self, features: Sequence[Dict[str, Union[Sequence, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Get all input_ids from the batch
        batch_input_ids = [feature["input_ids"] for feature in features]

        # Find the max length in this specific batch
        batch_max_length = min(
            max(len(ids) for ids in batch_input_ids),
            self.max_length
        )

        # Pad all sequences to batch_max_length
        padded_input_ids = [
            ids + [self.tokenizer.pad_token_id] * (batch_max_length - len(ids))
            for ids in batch_input_ids
        ]

        # Create attention mask
        attention_mask = [
            [1] * len(ids) + [0] * (batch_max_length - len(ids))
            for ids in batch_input_ids
        ]

        # Convert to tensors
        batch = {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(padded_input_ids).clone()  # For casual language modeling
        }

        return batch


from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
import os
import argparse
from typing import Dict, Sequence


class HebrewDatasetPreparation:
    def __init__(self, file_path: str, tokenizer, max_length: int = 512):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_dataset(self) -> Dataset:
        # Read the text file where each line is a sentence
        with open(self.file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

        # Create dataset
        dataset = Dataset.from_dict({"text": texts})
        return dataset

    def preprocess_function(self, examples: Dict[str, Sequence[str]]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_length,  # Keep max_length only for truncation
            padding=False  # No padding during preprocessing
        )


def setup_model_and_tokenizer(model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Get PEFT model
    model = get_peft_model(model, lora_config)

    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune LLaMA model on Hebrew dataset')

    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='llama-hebrew-finetuning',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Weights & Biases run name')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases logging')

    # Required arguments
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name or path of the base model')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the Hebrew dataset text file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the fine-tuned model')

    # Optional training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Training batch size per device')
    parser.add_argument('--grad_accumulation', type=int, default=4,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')

    # LoRA specific arguments
    parser.add_argument('--lora_r', type=int, default=16,
                        help='LoRA attention dimension')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                        help='LoRA dropout value')

    # Training schedule arguments
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Save checkpoint every X steps')
    parser.add_argument('--eval_steps', type=int, default=500,
                        help='Evaluate every X steps')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Number of warmup steps')

    return parser.parse_args()


def main():
    # Get command line arguments
    args = parse_args()

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        args.model_name,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout
    )

    # Prepare dataset
    dataset_prep = HebrewDatasetPreparation(args.dataset_path, tokenizer, args.max_length)
    dataset = dataset_prep.load_dataset()

    # Split dataset (90% train, 10% validation)
    dataset = dataset.train_test_split(test_size=0.1)

    # Process datasets
    tokenized_train = dataset["train"].map(
        dataset_prep.preprocess_function,
        remove_columns=dataset["train"].column_names,
        batched=True
    )
    tokenized_val = dataset["test"].map(
        dataset_prep.preprocess_function,
        remove_columns=dataset["test"].column_names,
        batched=True
    )

    # Handle wandb configuration
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.wandb_run_name or f"llama-hebrew-{os.path.basename(args.dataset_path)}",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=100,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        gradient_accumulation_steps=args.grad_accumulation,
        warmup_steps=args.warmup_steps,
        eval_strategy="steps",  # Updated from evaluation_strategy to eval_strategy
        logging_dir=f"{args.output_dir}/logs",
        save_total_limit=3,
        report_to="wandb" if not args.no_wandb else "none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorForDynamicBatching(tokenizer, args.max_length)
    )

    # Train the model
    trainer.train()

    # Save the final model
    trainer.save_model(f"{args.output_dir}/final")


if __name__ == "__main__":
    main()