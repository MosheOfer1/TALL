import os
import random
import argparse
import string

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
import pandas as pd
import wandb
from tqdm import tqdm


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PromptDataset(Dataset):
    """
    Dataset that reads a CSV file with two columns: 'sentence' and 'target'.
    Before processing, it cleans the data by:
      - Removing rows with NaN in either column.
      - Removing punctuation from the sentence.
      - Converting the target to lowercase.

    Each example is tokenized separately for the sentence and target. The sentence
    tokens are then prepended to the target tokens and the sentence tokens are masked
    out (set to -100) in the labels so that the loss is computed only on the target tokens.
    """

    def __init__(self, csv_file, tokenizer, max_input_length, max_target_length):
        # Load the CSV data
        self.data = pd.read_csv(csv_file)
        # Remove rows with NaN values in "sentence" or "target"
        self.data = self.data.dropna(subset=["sentence", "target"]).reset_index(drop=True)
        # Clean the data:
        # Remove punctuation from the sentence
        self.data["sentence"] = self.data["sentence"].apply(
            lambda x: x.translate(str.maketrans("", "", string.punctuation))
        )
        # Convert target to lowercase
        self.data["target"] = self.data["target"].apply(lambda x: x.lower())

        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sentence = row["sentence"]
        target = row["target"]

        # Tokenize the sentence (context)
        sentence_enc = self.tokenizer(
            sentence,
            truncation=True,
            max_length=self.max_input_length,
            add_special_tokens=True,
            return_tensors="pt",
        )
        # Tokenize the target (what we want to generate)
        target_enc = self.tokenizer(
            target,
            truncation=True,
            max_length=self.max_target_length,
            add_special_tokens=True,
            return_tensors="pt",
        )

        # Remove batch dimension
        sentence_ids = sentence_enc.input_ids.squeeze(0)
        target_ids = target_enc.input_ids.squeeze(0)

        # Create a single sequence: [sentence tokens] + [target tokens]
        full_input_ids = torch.cat([sentence_ids, target_ids], dim=0)
        attention_mask = torch.ones(full_input_ids.shape, dtype=torch.long)

        # Create labels: mask the sentence tokens (with -100) so loss is computed only on the target tokens.
        labels = full_input_ids.clone()
        sentence_len = sentence_ids.size(0)
        labels[:sentence_len] = -100

        return {"input_ids": full_input_ids, "attention_mask": attention_mask, "labels": labels}


def collate_fn(batch, pad_token_id):
    """
    Collate function to pad sequences in the batch.
    For labels, we pad with -100 so that the loss is not computed on padding tokens.
    """
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    attention_masks = torch.nn.utils.rnn.pad_sequence(
        attention_masks, batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )

    return {"input_ids": input_ids, "attention_mask": attention_masks, "labels": labels}


class SoftPromptTuningModel(nn.Module):
    """
    Wraps a causal LM model (e.g. Qwen2.5-0.5B) and prepends trainable soft prompt embeddings
    to every input. All parameters of the main model are frozen.
    """

    def __init__(self, model, prompt_length):
        super().__init__()
        self.model = model
        self.prompt_length = prompt_length
        self.embedding_dim = model.get_input_embeddings().embedding_dim

        # Initialize the soft prompt embeddings (e.g. from a normal distribution)
        self.soft_prompt = nn.Parameter(torch.randn(prompt_length, self.embedding_dim))

        # Freeze the main model parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        batch_size = input_ids.size(0)
        # Get embeddings for the tokens in the input_ids
        input_embeddings = self.model.get_input_embeddings()(input_ids)

        # Expand the soft prompt so that it is the same for every sample in the batch
        soft_prompt_expanded = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate the soft prompt embeddings with the input embeddings
        inputs_embeds = torch.cat([soft_prompt_expanded, input_embeddings], dim=1)

        # Extend the attention mask to account for the soft prompt tokens (all ones)
        soft_prompt_mask = torch.ones(
            batch_size, self.prompt_length, dtype=attention_mask.dtype, device=attention_mask.device
        )
        extended_attention_mask = torch.cat([soft_prompt_mask, attention_mask], dim=1)

        # Adjust labels: prepend -100 for the soft prompt tokens so that they are ignored in the loss
        if labels is not None:
            extended_labels = torch.cat(
                [torch.full((batch_size, self.prompt_length), -100, dtype=labels.dtype, device=labels.device), labels],
                dim=1,
            )
        else:
            extended_labels = None

        outputs = self.model(
            inputs_embeds=inputs_embeds, attention_mask=extended_attention_mask, labels=extended_labels
        )
        return outputs

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        """
        Generate text while using the soft prompt.
        """
        batch_size = input_ids.size(0)

        # Get embeddings for input tokens
        input_embeddings = self.model.get_input_embeddings()(input_ids)

        # Expand soft prompt for batch
        soft_prompt_expanded = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate soft prompt with input embeddings
        inputs_embeds = torch.cat([soft_prompt_expanded, input_embeddings], dim=1)

        # Extend attention mask for soft prompt tokens
        if attention_mask is not None:
            soft_prompt_mask = torch.ones(
                batch_size, self.prompt_length,
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            attention_mask = torch.cat([soft_prompt_mask, attention_mask], dim=1)

        # Generate using the base model
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )

        # # Remove the soft prompt tokens from the output
        # outputs = outputs[:, self.prompt_length:]
        return outputs

    def save_soft_prompt(self, output_dir, epoch, val_loss):
        """
        Save the soft prompt embeddings and model state.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save the complete model state
        checkpoint = {
            'epoch': epoch,
            'soft_prompt': self.soft_prompt,
            'val_loss': val_loss,
        }

        checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}_val_{val_loss}.pt')
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    @classmethod
    def load_soft_prompt(cls, model, checkpoint_path):
        """
        Load the soft prompt embeddings from a saved checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        instance = cls(model, prompt_length=checkpoint['soft_prompt'].size(0))
        instance.soft_prompt.data = checkpoint['soft_prompt']
        return instance


def evaluate(model, val_loader, device):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

    avg_loss = total_loss / len(val_loader)
    model.train()
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description="Soft prompt tuning for Qwen2.5-0.5B")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV file with training examples")
    parser.add_argument("--model_name", type=str, default="qwen/Qwen2.5-0.5B", help="Pretrained model name")
    parser.add_argument("--prompt_length", type=int, default=30, help="Length of the soft prompt (number of tokens)")
    parser.add_argument("--max_input_length", type=int, default=128, help="Max token length for input sentence")
    parser.add_argument("--max_target_length", type=int, default=32, help="Max token length for target")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for the soft prompt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every N steps")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps for LR scheduler")
    parser.add_argument("--output_dir", type=str, default="soft_prompt", help="Directory to save the soft prompt")
    parser.add_argument("--val_ratio", type=float, default=0.005, help="Validation set ratio")
    parser.add_argument("--save_steps", type=int, default=800, help="Save checkpoint every N steps")

    args = parser.parse_args()

    wandb.init(project="soft_prompt_tuning", config=vars(args))
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    soft_prompt_model = SoftPromptTuningModel(model, args.prompt_length)
    soft_prompt_model.to(device)

    # Prepare the dataset and split into train/val
    full_dataset = PromptDataset(args.csv_file, tokenizer, args.max_input_length, args.max_target_length)
    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
    )

    total_steps = len(train_loader) * args.num_epochs
    optimizer = torch.optim.AdamW([soft_prompt_model.soft_prompt], lr=args.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = soft_prompt_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_loss += loss.item()

            if global_step % args.logging_steps == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "step": global_step,
                    "epoch": epoch + 1,
                })

            # Evaluate and save periodically
            if global_step % args.save_steps == 0:
                val_loss = evaluate(soft_prompt_model, val_loader, device)
                wandb.log({"val_loss": val_loss, "step": global_step})

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = soft_prompt_model.save_soft_prompt(
                        args.output_dir, epoch, val_loss
                    )
                    print(f"New best model saved to {checkpoint_path} with validation loss: {val_loss:.4f}")

            progress_bar.set_postfix({"loss": loss.item()})

        # Evaluate at the end of each epoch
        val_loss = evaluate(soft_prompt_model, val_loader, device)
        avg_epoch_loss = epoch_loss / len(train_loader)

        print(f"Epoch {epoch + 1} - Train Loss: {avg_epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "avg_epoch_loss": avg_epoch_loss,
            "epoch_val_loss": val_loss
        })

        # Save if it's the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = soft_prompt_model.save_soft_prompt(
                args.output_dir, epoch, val_loss
            )
            print(f"New best model saved to {checkpoint_path} with validation loss: {val_loss:.4f}")

    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    wandb.finish()


if __name__ == "__main__":
    main()
