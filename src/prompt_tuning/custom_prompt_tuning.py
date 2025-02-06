import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import argparse
from torch.utils.data import Dataset, DataLoader
import wandb
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    model_name: str
    data_path: str
    n_tokens: int
    learning_rate: float
    batch_size: int
    epochs: int
    max_length: int
    warmup_steps: int
    gradient_accumulation_steps: int
    eval_steps: int
    save_steps: int
    project_name: str
    output_dir: str


class SoftPrompt(nn.Module):
    def __init__(self, n_tokens: int, embedding_size: int, device: str):
        super().__init__()
        self.n_tokens = n_tokens
        self.embedding_size = embedding_size

        # Initialize with the model's embedding layer distribution
        self.soft_prompt = nn.Parameter(
            torch.normal(mean=0.0, std=0.02, size=(n_tokens, embedding_size))
        )
        self.device = device
        self.to(device)

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size = input_embeddings.shape[0]
        soft_prompt = self.soft_prompt.repeat(batch_size, 1, 1)
        return torch.cat([soft_prompt, input_embeddings], dim=1)


class PromptTuningCollator:
    def __init__(self, tokenizer, max_length: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        sentences = [item['sentence'] for item in batch]
        targets = [item['target'] for item in batch]

        # Tokenize inputs
        inputs = self.tokenizer(
            sentences,
            padding=False,  # We'll do dynamic padding
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )

        # Tokenize targets
        target_tokens = self.tokenizer(
            targets,
            padding=False,
            return_tensors=None,
        )

        # Dynamic padding for this batch
        max_len = max(len(ids) for ids in inputs['input_ids'])

        # Pad sequences
        input_ids = pad_sequence(
            [torch.tensor(ids) for ids in inputs['input_ids']],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        attention_mask = pad_sequence(
            [torch.tensor(mask) for mask in inputs['attention_mask']],
            batch_first=True,
            padding_value=0
        )

        target_ids = pad_sequence(
            [torch.tensor(ids) for ids in target_tokens['input_ids']],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_ids': target_ids,
            'sentences': sentences,  # Keep original sentences for logging
            'targets': targets  # Keep original targets for logging
        }


class PromptTuningDataset(Dataset):
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path)
        self.sentences = self.df['sentence'].tolist()
        self.targets = self.df['target'].tolist()

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return {
            'sentence': self.sentences[idx].replace(".",""),
            'target': self.targets[idx]
        }


class PromptTuner:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_wandb()
        self.setup_output_dir()
        self.setup_model_and_tokenizer()
        self.setup_soft_prompt()

    def setup_wandb(self):
        wandb.init(
            project=self.config.project_name,
            config=vars(self.config),
            name=f"prompt-tuning-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

    def setup_output_dir(self):
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(vars(self.config), f, indent=4)

    def setup_model_and_tokenizer(self):
        logger.info(f"Loading model: {self.config.model_name}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name).to(self.device)

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def setup_soft_prompt(self):
        self.soft_prompt = SoftPrompt(
            self.config.n_tokens,
            self.model.config.hidden_size,
            self.device
        )

        # Use AdamW optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.soft_prompt.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )

        # Cosine learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.warmup_steps,
            T_mult=2
        )

    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None):
        logger.info("Starting training...")
        global_step = 0
        total_loss = 0
        best_eval_loss = float('inf')

        for epoch in range(self.config.epochs):
            self.soft_prompt.train()
            epoch_loss = 0

            # Create progress bar
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")

            for step, batch in enumerate(pbar):
                loss = self.training_step(batch)
                total_loss += loss
                epoch_loss += loss

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    # Update progress bar
                    pbar.set_postfix({'loss': loss})

                    # Log to wandb
                    wandb.log({
                        'loss': loss,
                        'learning_rate': self.scheduler.get_last_lr()[0],
                        'global_step': global_step,
                    })

                    # Evaluate if needed
                    if global_step % self.config.eval_steps == 0 and eval_dataloader is not None:
                        eval_loss = self.evaluate(eval_dataloader)
                        wandb.log({'eval_loss': eval_loss})

                        # Save best model
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            self.save_soft_prompt('best_model')

                    # Save checkpoint if needed
                    if global_step % self.config.save_steps == 0:
                        self.save_soft_prompt(f'checkpoint-{global_step}')

            # End of epoch logging
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs} - Average loss: {avg_epoch_loss:.4f}")

            # Plot and save loss curve
            self.plot_loss_curve(epoch)

            # Save epoch checkpoint
            self.save_soft_prompt(f'epoch-{epoch + 1}')

    def training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        target_ids = batch['target_ids'].to(self.device)

        # Get only the first token of each target sequence
        target_ids = target_ids[:, 0].unsqueeze(-1)  # Shape: [batch_size, 1]

        # Get input embeddings
        input_embeddings = self.model.get_input_embeddings()(input_ids)

        # Add soft prompt
        modified_embeddings = self.soft_prompt(input_embeddings)

        # Extend attention mask to account for soft prompt tokens
        prompt_attention_mask = torch.ones(
            (attention_mask.shape[0], self.soft_prompt.n_tokens),
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        modified_attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)

        # Forward pass
        outputs = self.model(
            inputs_embeds=modified_embeddings,
            attention_mask=modified_attention_mask,
            labels=target_ids
        )

        loss = outputs.loss / self.config.gradient_accumulation_steps
        loss.backward()

        return loss.item() * self.config.gradient_accumulation_steps

    @torch.no_grad()
    def evaluate(self, eval_dataloader: DataLoader) -> float:
        logger.info("Running evaluation...")
        self.soft_prompt.eval()
        total_eval_loss = 0

        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)

            # Get only the first token of each target sequence
            target_ids = target_ids[:, 0].unsqueeze(-1)  # Shape: [batch_size, 1]

            input_embeddings = self.model.get_input_embeddings()(input_ids)
            modified_embeddings = self.soft_prompt(input_embeddings)

            # Extend attention mask to account for soft prompt tokens
            prompt_attention_mask = torch.ones(
                (attention_mask.shape[0], self.soft_prompt.n_tokens),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            modified_attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)

            outputs = self.model(
                inputs_embeds=modified_embeddings,
                attention_mask=modified_attention_mask,
                labels=target_ids
            )

            total_eval_loss += outputs.loss.item()

        avg_eval_loss = total_eval_loss / len(eval_dataloader)
        logger.info(f"Evaluation loss: {avg_eval_loss:.4f}")
        return avg_eval_loss

    def save_soft_prompt(self, checkpoint_name: str):
        save_path = self.output_dir / checkpoint_name
        save_path.mkdir(exist_ok=True)

        # Save soft prompt state
        torch.save(self.soft_prompt.state_dict(), save_path / 'soft_prompt.pt')

        # Save optimizer state
        torch.save(self.optimizer.state_dict(), save_path / 'optimizer.pt')

        # Save scheduler state
        torch.save(self.scheduler.state_dict(), save_path / 'scheduler.pt')

        logger.info(f"Saved checkpoint: {save_path}")

    def plot_loss_curve(self, epoch: int):
        plt.figure(figsize=(10, 6))
        runs = wandb.Api().runs(
            path=f"{wandb.run.entity}/{wandb.run.project}",
            filters={"display_name": wandb.run.name}
        )

        history = runs[0].history()
        plt.plot(history['global_step'], history['loss'], label='Training Loss')
        if 'eval_loss' in history:
            plt.plot(history['global_step'], history['eval_loss'], label='Evaluation Loss')

        plt.xlabel('Global Step')
        plt.ylabel('Loss')
        plt.title(f'Loss Curve - Epoch {epoch + 1}')
        plt.legend()

        # Save plot
        plt.savefig(self.output_dir / f'loss_curve_epoch_{epoch + 1}.png')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train soft prompts with advanced features')
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--n_tokens', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--project_name', type=str, default='prompt-tuning')
    parser.add_argument('--output_dir', type=str, default='outputs')

    args = parser.parse_args()
    config = TrainingConfig(**vars(args))

    # Set up datasets
    dataset = PromptTuningDataset(args.data_path)

    # Split dataset into train/eval
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )

    # Create dataloaders
    collator = PromptTuningCollator(AutoTokenizer.from_pretrained(args.model_name))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True
    )

    # Initialize and train
    tuner = PromptTuner(config)
    tuner.train(train_dataloader, eval_dataloader)


if __name__ == "__main__":
    main()