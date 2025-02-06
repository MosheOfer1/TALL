import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from typing import Dict, List, Optional
import logging
import os
import argparse
import glob

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration class for different model architectures"""

    @staticmethod
    def get_model_config(model_name: str) -> Dict:
        """Get model-specific configuration"""
        # Base configuration
        config = {
            "trust_remote_code": True,  # Required for some models like Qwen
            "padding_side": "right",
            "use_fast_tokenizer": True
        }

        return config


class GenericModelFineTuner:
    def __init__(
            self,
            model_name: str,
            output_dir: str = "finetuned-model",
            max_length: int = 128,
            model_config: Optional[Dict] = None
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Get model-specific configuration
        self.model_config = model_config or ModelConfig.get_model_config(model_name)

        logger.info(f"Using device: {self.device}")
        logger.info(f"Model configuration: {self.model_config}")

        # Initialize tokenizer and model with specific configurations
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.model_config["trust_remote_code"],
            use_fast=self.model_config["use_fast_tokenizer"]
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=self.model_config["trust_remote_code"],
            device_map="auto" if self.device == "cuda" else None,
            **{k: v for k, v in self.model_config.items()
               if k not in ["trust_remote_code", "use_fast_tokenizer", "padding_side"]}
        )

        # Handle tokenizer settings
        self._setup_tokenizer()

    def _setup_tokenizer(self):
        """Setup tokenizer with proper padding token and settings"""
        # Set padding side
        self.tokenizer.padding_side = self.model_config["padding_side"]

        # Handle padding token
        if self.tokenizer.pad_token is None:
            if "llama" in self.model_name.lower():
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.model.config.eos_token_id
            elif "qwen" in self.model_name.lower():
                self.tokenizer.pad_token = '<|endoftext|>'
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            else:  # Default behavior (BLOOMZ and others)
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
        """Prepare the dataset for training with dynamic padding."""
        dataset_dict = {"text": texts}
        dataset = Dataset.from_dict(dataset_dict)

        def tokenize_function(examples: Dict) -> Dict:
            return self.tokenizer(
                examples["text"],
                padding=True,
                truncation=True,
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
            resume_from_checkpoint: bool = False,
            gradient_accumulation_steps: int = 1
    ):
        """Fine-tune the model on the provided texts."""
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_texts)
        eval_dataset = self.prepare_dataset(eval_texts)

        # Initialize training arguments with model-specific optimizations
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            eval_steps=logging_steps,
            save_steps=logging_steps * 50,
            gradient_accumulation_steps=gradient_accumulation_steps,
            eval_accumulation_steps=gradient_accumulation_steps,
            fp16=self.device == 'cuda',
            logging_dir=os.path.join(self.output_dir, "logs"),
            save_total_limit=3,
            load_best_model_at_end=True,
            eval_strategy="steps",
            save_strategy="steps",
            max_grad_norm=1.0,  # Add gradient clipping
            weight_decay=0.01,
            # Add model-specific training arguments
            optim="adamw_torch" if "llama" in self.model_name.lower() else "adamw_hf",
            lr_scheduler_type="cosine",
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

        # Find latest checkpoint if resuming
        checkpoint_dir = None
        if resume_from_checkpoint:
            checkpoint_dirs = glob.glob(os.path.join(self.output_dir, "checkpoint-*"))
            if checkpoint_dirs:
                checkpoint_dir = max(checkpoint_dirs, key=lambda x: int(x.split("-")[-1]))
                logger.info(f"Resuming from checkpoint: {checkpoint_dir}")

        # Start training
        trainer.train(resume_from_checkpoint=checkpoint_dir)

        # Save the final model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Fine-tune language models')
    parser.add_argument('--train-file', type=str, required=True,
                        help='Path to training data file (one sentence per line)')
    parser.add_argument('--eval-file', type=str, required=True,
                        help='Path to evaluation data file (one sentence per line)')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Name or path of the base model')
    parser.add_argument('--output-dir', type=str, default="finetuned-model",
                        help='Directory to save the fine-tuned model')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Number of steps for gradient accumulation')
    parser.add_argument('--num-epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--max-length', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--resume-from-checkpoint', action='store_true',
                        help='Resume training from the latest checkpoint if available')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Initialize fine-tuner
    fine_tuner = GenericModelFineTuner(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_length=args.max_length
    )

    # Load texts
    train_texts = fine_tuner.load_texts_from_file(args.train_file)
    eval_texts = fine_tuner.load_texts_from_file(args.eval_file)

    # Start training
    fine_tuner.train(
        train_texts=train_texts,
        eval_texts=eval_texts,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        resume_from_checkpoint=args.resume_from_checkpoint,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )