import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


class DimensionAlignmentAutoencoder(nn.Module):
    def __init__(self, input_dim, target_dim, hidden_factor=2):
        """
        Initialize the autoencoder for dimension alignment.

        Args:
            input_dim: Input dimension (he_en_model.config.d_model)
            target_dim: Target dimension (llm_model.config.hidden_size)
            hidden_factor: Factor to multiply target_dim for the hidden layer size
        """
        super().__init__()

        self.input_dim = input_dim
        self.target_dim = target_dim

        # Encoder layers that transform to target dimension
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, target_dim * hidden_factor),
            nn.LayerNorm(target_dim * hidden_factor),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(target_dim * hidden_factor, target_dim),
            nn.LayerNorm(target_dim),
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(target_dim, target_dim * hidden_factor),
            nn.LayerNorm(target_dim * hidden_factor),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(target_dim * hidden_factor, input_dim),
            nn.LayerNorm(input_dim),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def apply_attention_mask(self, x, attention_mask):
        """
        Apply attention mask to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            attention_mask: Attention mask of shape (batch_size, seq_len)

        Returns:
            Masked tensor of the same shape as input
        """
        # Expand mask to match input dimensions
        mask = attention_mask.unsqueeze(-1).expand_as(x)

        # Apply mask
        masked_x = x * mask
        return masked_x

    def encode(self, x, attention_mask=None):
        """
        Encode input to target dimension.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)
        """
        batch_size = None
        seq_len = None

        # Handle 3D input
        if len(x.shape) == 3:
            batch_size, seq_len, _ = x.shape
            # Reshape to 2D
            x = x.contiguous().view(-1, self.input_dim)

        # Apply encoder
        encoded = self.encoder(x)

        # Reshape back to 3D if needed
        if batch_size is not None:
            encoded = encoded.view(batch_size, seq_len, self.target_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            encoded = self.apply_attention_mask(encoded, attention_mask)

        return encoded

    def decode(self, z, attention_mask=None):
        """
        Decode encoded representation back to input dimension.

        Args:
            z: Encoded tensor
            attention_mask: Optional attention mask
        """
        batch_size = None
        seq_len = None

        # Handle 3D input
        if len(z.shape) == 3:
            batch_size, seq_len, _ = z.shape
            # Reshape to 2D
            z = z.contiguous().view(-1, self.target_dim)

        # Apply decoder
        decoded = self.decoder(z)

        # Reshape back to 3D if needed
        if batch_size is not None:
            decoded = decoded.view(batch_size, seq_len, self.input_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            decoded = self.apply_attention_mask(decoded, attention_mask)

        return decoded

    def forward(self, x, attention_mask=None, return_encoded=False):
        """
        Forward pass through the autoencoder.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)
            return_encoded: If True, return both encoded and decoded tensors
        """
        # First apply attention mask to input if provided
        if attention_mask is not None:
            x = self.apply_attention_mask(x, attention_mask)

        # Encode with attention mask
        encoded = self.encode(x, attention_mask)

        # Decode with attention mask
        decoded = self.decode(encoded, attention_mask)

        if return_encoded:
            return encoded, decoded
        return decoded


class EncoderHiddenStatesDataset(Dataset):
    def __init__(self, sentences, source_model, tokenizer, device='cuda', max_length=128, model_type='he_en'):
        """
        Dataset for getting encoder hidden states on the fly.
        """
        self.sentences = sentences
        self.source_model = source_model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.model_type = model_type
        self.source_model.eval()

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        # Tokenize
        inputs = self.tokenizer(
            sentence,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        attention_mask = inputs["attention_mask"]

        # Get hidden states based on model type
        with torch.no_grad():
            if self.model_type == 'he_en':
                outputs = self.source_model.model.encoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=attention_mask,
                )
                hidden_states = outputs.last_hidden_state.squeeze(0)
            else:  # llm
                outputs = self.source_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[-1].squeeze(0)

        # Return both hidden states and attention mask
        return hidden_states, attention_mask.squeeze(0)

    def collate_fn(self, batch):
        """Custom collate function to handle variable length sequences."""
        # Separate hidden states and attention masks
        hidden_states, attention_masks = zip(*batch)

        # Find max length in batch
        max_len = max(x.size(0) for x in hidden_states)

        # Pad sequences to max length
        padded_hidden_states = []
        padded_attention_masks = []

        for hidden_state, attention_mask in zip(hidden_states, attention_masks):
            if hidden_state.size(0) < max_len:
                # Pad hidden states
                padding = torch.zeros(
                    (max_len - hidden_state.size(0), hidden_state.size(1)),
                    dtype=hidden_state.dtype,
                    device=hidden_state.device
                )
                hidden_state = torch.cat([hidden_state, padding], dim=0)

                # Pad attention mask
                mask_padding = torch.zeros(
                    max_len - attention_mask.size(0),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                attention_mask = torch.cat([attention_mask, mask_padding], dim=0)

            padded_hidden_states.append(hidden_state)
            padded_attention_masks.append(attention_mask)

        return {
            'hidden_states': torch.stack(padded_hidden_states),
            'attention_mask': torch.stack(padded_attention_masks)
        }


def masked_mse_loss(output, target, attention_mask):
    """
    Compute MSE loss only on non-padded elements.

    Args:
        output: Model output tensor (batch_size, seq_len, hidden_dim)
        target: Target tensor (batch_size, seq_len, hidden_dim)
        attention_mask: Binary mask indicating non-padded elements (batch_size, seq_len)
    """
    # Expand attention mask to match hidden dimension
    mask = attention_mask.unsqueeze(-1).expand_as(output)

    # Calculate squared differences
    squared_diff = (output - target) ** 2

    # Apply mask and calculate mean
    masked_squared_diff = squared_diff * mask

    # Sum of masked differences
    sum_squared_diff = masked_squared_diff.sum()

    # Count of non-padded elements (accounting for hidden dimension)
    num_elements = mask.sum()

    # Return mean loss
    return sum_squared_diff / num_elements if num_elements > 0 else sum_squared_diff


class AutoencoderPreTrainer:
    def __init__(self, autoencoder, source_model, tokenizer, device='cuda', model_type='he_en'):
        """
        Initialize the autoencoder trainer.

        Args:
            autoencoder: DimensionAlignmentAutoencoder instance
            source_model: Source model (he_en_model or llm_model)
            tokenizer: Tokenizer for the source model
            device: Device to use for training
            model_type: Type of model ('he_en' or 'llm')
        """
        self.autoencoder = autoencoder
        self.source_model = source_model
        self.tokenizer = tokenizer
        self.device = device
        self.model_type = model_type

        self.autoencoder.to(device)
        self.source_model.to(device)
        self.source_model.eval()

        # Initialize logging
        self.train_losses = []
        self.val_losses = []
        self.epoch_train_losses = []
        self.epoch_val_losses = []

    def create_data_loaders(self, sentences, batch_size, validation_split=0.1):
        """Create train and validation datasets and data loaders."""
        # Split sentences into train and validation
        val_size = int(len(sentences) * validation_split)
        train_sentences = sentences[:-val_size]
        val_sentences = sentences[-val_size:]

        # Create datasets
        train_dataset = EncoderHiddenStatesDataset(
            train_sentences,
            self.source_model,
            self.tokenizer,
            self.device,
            model_type=self.model_type
        )
        val_dataset = EncoderHiddenStatesDataset(
            val_sentences,
            self.source_model,
            self.tokenizer,
            self.device,
            model_type=self.model_type
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=train_dataset.collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=val_dataset.collate_fn
        )

        return train_loader, val_loader


    def save_logs(self, save_dir):
        """Save training logs to files"""
        import json
        import numpy as np

        # Save detailed (batch-level) losses
        np.save(os.path.join(save_dir, 'train_losses.npy'), np.array(self.train_losses))
        np.save(os.path.join(save_dir, 'val_losses.npy'), np.array(self.val_losses))

        # Save epoch-level losses
        logs = {
            'train_losses': self.epoch_train_losses,
            'val_losses': self.epoch_val_losses
        }

        with open(os.path.join(save_dir, 'training_log.json'), 'w') as f:
            json.dump(logs, f, indent=4)

    def plot_losses(self, save_dir):
        """Plot training and validation losses"""

        # Plot epoch-level losses
        plt.figure(figsize=(10, 6))
        plt.plot(self.epoch_train_losses, label='Training Loss')
        plt.plot(self.epoch_val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
        plt.close()

        # Plot detailed batch-level training loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses)
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Batch-level Training Loss')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'batch_loss_plot.png'))
        plt.close()

    def train(self,
              sentences,
              num_epochs=100,
              batch_size=32,
              learning_rate=1e-4,
              validation_split=0.1,
              save_dir='checkpoints',
              log_every=100):

        train_loader, val_loader = self.create_data_loaders(
            sentences, batch_size, validation_split
        )

        optimizer = torch.optim.AdamW(self.autoencoder.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        no_improve_count = 0

        # Create log file
        log_file = os.path.join(save_dir, 'training.log')

        print("Starting training...")
        for epoch in range(num_epochs):
            # Training
            self.autoencoder.train()
            epoch_train_loss = 0
            batch_count = 0
            running_loss = 0.0
            log_interval_loss = 0.0

            pbar = tqdm(enumerate(train_loader),
                        total=len(train_loader),
                        desc=f'Epoch {epoch + 1}/{num_epochs}')

            for batch_idx, batch_data in pbar:
                try:
                    optimizer.zero_grad()

                    # Get hidden states and attention mask from batch
                    hidden_states = batch_data['hidden_states']
                    attention_mask = batch_data['attention_mask']

                    # Forward pass with attention mask
                    output = self.autoencoder(hidden_states, attention_mask=attention_mask)

                    # Calculate masked loss
                    loss = masked_mse_loss(output, hidden_states, attention_mask)

                    # Store batch loss
                    self.train_losses.append(loss.item())
                    running_loss = 0.9 * running_loss + 0.1 * loss.item()
                    log_interval_loss += loss.item()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), 1.0)
                    optimizer.step()

                    epoch_train_loss += loss.item()
                    batch_count += 1

                    pbar.set_postfix({
                        'loss': f'{running_loss:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                    })

                    # Log every N batches
                    if (batch_idx + 1) % log_every == 0:
                        avg_interval_loss = log_interval_loss / log_every
                        print(f"\nBatch {batch_idx + 1}/{len(train_loader)}, "
                              f"Average Loss: {avg_interval_loss:.4f}, "
                              f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

                        with open(log_file, 'a') as f:
                            f.write(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, "
                                    f"Loss: {avg_interval_loss:.6f}, "
                                    f"LR: {scheduler.get_last_lr()[0]:.2e}\n")

                        log_interval_loss = 0.0

                except RuntimeError as e:
                    print(f"\nError in batch {batch_idx}:")
                    print(str(e))
                    continue

            avg_train_loss = epoch_train_loss / batch_count
            self.epoch_train_losses.append(avg_train_loss)

            # Validation with progress bar
            self.autoencoder.eval()
            val_loss = 0
            val_count = 0

            val_pbar = tqdm(val_loader,
                            desc='Validation',
                            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

            with torch.no_grad():
                running_val_loss = 0.0
                for batch_data in val_pbar:
                    try:
                        # Get hidden states and attention mask from batch
                        hidden_states = batch_data['hidden_states']
                        attention_mask = batch_data['attention_mask']

                        # Forward pass with attention mask
                        output = self.autoencoder(hidden_states, attention_mask=attention_mask)

                        # Calculate masked loss
                        batch_loss = masked_mse_loss(output, hidden_states, attention_mask)

                        val_loss += batch_loss.item()
                        self.val_losses.append(batch_loss.item())
                        val_count += 1

                        # Update running validation loss
                        running_val_loss = 0.9 * running_val_loss + 0.1 * batch_loss.item()
                        val_pbar.set_postfix({'val_loss': f'{running_val_loss:.4f}'})

                    except RuntimeError as e:
                        print(f"Error in validation batch: {str(e)}")
                        continue

                avg_val_loss = val_loss / val_count
                self.epoch_val_losses.append(avg_val_loss)

            # Log epoch results
            log_message = (
                f'\nEpoch {epoch + 1}/{num_epochs}:\n'
                f'Train Loss: {avg_train_loss:.6f}\n'
                f'Val Loss: {avg_val_loss:.6f}\n'
                f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}\n'
            )

            print(log_message)
            with open(log_file, 'a') as f:
                f.write(log_message + '\n')

            # Update learning rate
            scheduler.step(avg_val_loss)

            # Save best model and handle early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve_count = 0

                # Save model and logs
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.autoencoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'epoch_train_losses': self.epoch_train_losses,
                    'epoch_val_losses': self.epoch_val_losses,
                }, os.path.join(save_dir, 'best_autoencoder.pt'))

                print(f'Saved new best model with validation loss: {avg_val_loss:.6f}')
            else:
                no_improve_count += 1

            # Save current logs and plots
            self.save_logs(save_dir)
            self.plot_losses(save_dir)

            if no_improve_count >= 10:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

        print("Training completed!")
        return best_val_loss