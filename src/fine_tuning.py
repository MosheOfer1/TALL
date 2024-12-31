import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from typing import Dict, List
import logging
import os
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BloomzHebrewFineTuner:
    def __init__(
            self,
            model_name: str = "bigscience/bloomz-560m",
            output_dir: str = "bloomz-hebrew-finetuned",
            max_length: int = 128,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Using device: {self.device}")

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32
        )

        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def load_texts_from_file(self, file_path: str) -> List[str]:
        """Load texts from a file, one sentence per line."""
        logger.info(f"Loading texts from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(texts)} sentences")
        return texts

    def prepare_dataset(self, texts: List[str]) -> Dataset:
        """Prepare the dataset for training."""
        dataset_dict = {"text": texts}
        dataset = Dataset.from_dict(dataset_dict)

        def tokenize_function(examples: Dict) -> Dict:
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        return tokenized_dataset

    def train(
            self,
            train_texts: List[str],
            eval_texts: List[str],
            batch_size: int = 8,
            num_epochs: int = 3,
            learning_rate: float = 2e-5,
            warmup_steps: int = 500,
            logging_steps: int = 100,
    ):
        """Fine-tune the model on the provided texts."""
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_texts)
        eval_dataset = self.prepare_dataset(eval_texts)

        # Initialize training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            eval_strategy="steps",
            save_strategy="steps",
            save_steps=1000,
            load_best_model_at_end=True,
            gradient_accumulation_steps=4,
            fp16=self.device == "cuda",
            logging_dir=os.path.join(self.output_dir, "logs"),
        )

        # Initialize data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Start training
        logger.info("Starting training...")
        trainer.train()

        # Save the final model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Fine-tune BLOOMZ model on Hebrew text')
    parser.add_argument('--train-file', type=str, required=True,
                        help='Path to training data file (one sentence per line)')
    parser.add_argument('--eval-file', type=str, required=True,
                        help='Path to evaluation data file (one sentence per line)')
    parser.add_argument('--model-name', type=str, default="bigscience/bloomz-560m",
                        help='Name or path of the base model')
    parser.add_argument('--output-dir', type=str, default="bloomz-hebrew-finetuned",
                        help='Directory to save the fine-tuned model')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--num-epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--max-length', type=int, default=128,
                        help='Maximum sequence length')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Initialize fine-tuner
    fine_tuner = BloomzHebrewFineTuner(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_length=args.max_length
    )

    # Load texts from files
    train_texts = fine_tuner.load_texts_from_file(args.train_file)
    eval_texts = fine_tuner.load_texts_from_file(args.eval_file)

    # Start training
    fine_tuner.train(
        train_texts=train_texts,
        eval_texts=eval_texts,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )