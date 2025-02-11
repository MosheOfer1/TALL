import torch
from torch import nn
import pandas as pd
import argparse
import string
from datasets import Dataset
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)


def clean_data(df):
    """Clean the dataset according to specified rules"""
    # Remove rows with missing values
    df = df.dropna(subset=["sentence", "target"]).reset_index(drop=True)

    # Remove punctuation from sentences
    df["sentence"] = df["sentence"].apply(
        lambda x: x.translate(str.maketrans("", "", string.punctuation))
    )

    # Convert targets to lowercase
    df["target"] = df["target"].apply(lambda x: x.lower())

    return df


class TranslationPromptTuner:
    def __init__(self, csv_file, model_name, soft_prompt_length=10):
        self.model_name = model_name
        self.soft_prompt_length = soft_prompt_length
        self.max_length = 512 - soft_prompt_length  # Reserve space for soft prompts

        # Load and clean data
        self.df = clean_data(pd.read_csv(csv_file))
        self.dataset = Dataset.from_pandas(self.df)

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Tokenize dataset
        self.tokenized_dataset = self.dataset.map(
            self.tokenize_function,
            batched=True
        )

    def tokenize_function(self, examples):
        # Tokenize sentences and targets separately
        tokenized_sentences = self.tokenizer(
            examples["sentence"],
            truncation=True,
            max_length=self.max_length // 2,
            padding=False
        )
        tokenized_targets = self.tokenizer(
            examples["target"],
            truncation=True,
            max_length=self.max_length // 2,
            padding=False,
            add_special_tokens=False
        )

        combined_input_ids = []
        labels = []
        for s, t in zip(tokenized_sentences["input_ids"], tokenized_targets["input_ids"]):
            # Combine and truncate to max_length
            combined = s + t
            combined = combined[:self.max_length]
            # Create labels (-100 for prompt, actual tokens for target)
            label = [-100] * len(s) + t
            label = label[:self.max_length]
            combined_input_ids.append(combined)
            labels.append(label)

        return {"input_ids": combined_input_ids, "labels": labels}

    def train(self):
        # Initialize wandb
        wandb.init(project="qwen-soft-prompt-tuning", config={
            "model": self.model_name,
            "soft_prompt_length": self.soft_prompt_length,
            "dataset_size": len(self.df)
        })

        # Initialize model
        base_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        model = SoftPromptModel(base_model, self.soft_prompt_length)

        # Training setup
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=3e-2,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_dir="./logs",
            logging_steps=20,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="wandb",
            lr_scheduler_type="cosine",
            gradient_accumulation_steps=2,
            fp16=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.tokenized_dataset,
            eval_dataset=self.tokenized_dataset,  # Use validation split in practice
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        # Start training
        trainer.train()

        # Save soft prompts
        torch.save(model.soft_prompt.state_dict(), "soft_prompts.pth")
        wandb.save("soft_prompts.pth")

        # Finish wandb
        wandb.finish()


class SoftPromptModel(nn.Module):
    def __init__(self, model, soft_prompt_length):
        super().__init__()
        self.model = model
        self.soft_prompt_length = soft_prompt_length
        self.soft_prompt = nn.Parameter(
            torch.randn(soft_prompt_length, model.config.hidden_size))
        # Freeze base model
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        batch_size = inputs_embeds.size(0)

        # Add soft prompts
        soft_prompts = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = torch.cat([soft_prompts, inputs_embeds], dim=1)

        # Adjust attention mask and labels
        if attention_mask is not None:
            attention_mask = torch.cat([
                torch.ones((batch_size, self.soft_prompt_length),
                           device=attention_mask.device),
                attention_mask
            ], dim=1)

        if labels is not None:
            labels = torch.cat([
                torch.full((batch_size, self.soft_prompt_length), -100,
                           device=labels.device),
                labels
            ], dim=1)

        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Soft Prompt Tuning for Qwen2.5-0.5B")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to training CSV file")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="Model name or path")
    parser.add_argument("--soft_prompt_length", type=int, default=10,
                        help="Length of soft prompt tokens")

    args = parser.parse_args()

    tuner = TranslationPromptTuner(
        csv_file=args.csv_path,
        model_name=args.model_name,
        soft_prompt_length=args.soft_prompt_length
    )
    tuner.train()