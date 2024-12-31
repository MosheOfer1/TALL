import torch
import torch.nn.functional as F
import math
import os
from utils import print_progress_bar, print_model_info, setup_logger, log_error


class Trainer:
    def __init__(self, model, tokenizer1, tokenizer2, device, log_dir, save_dir, checkpoint):
        self.scheduler_state_dict = None
        self.optimizer_state_dict = None
        self.model = model.to(device)
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2
        self.pad_token_id = tokenizer2.pad_token_id
        self.eos_token_id = tokenizer2.eos_token_id

        self.device = device
        self.logger = setup_logger(log_dir)
        self.save_dir = save_dir
        self.best_eval_loss = float('inf')

        self.logger.info(f"Using device: {device}")
        self.logger.info("Model Architecture:")
        print_model_info(model, self.logger)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.start_epoch = 0
        if checkpoint:
            self.load_checkpoint(checkpoint)

    def evaluate_batch(self, logits, targets):
        # Create a mask to ignore pad tokens
        ignore_mask = (targets != self.pad_token_id)

        # Apply the mask to both logits and targets
        filtered_logits = logits.view(-1, logits.size(-1))[ignore_mask.view(-1)]
        filtered_targets = targets.view(-1)[ignore_mask.view(-1)]

        # Calculate loss using the filtered logits and targets
        loss = F.cross_entropy(filtered_logits, filtered_targets)

        # Calculate accuracy
        pred = logits.argmax(dim=-1)
        correct = ((pred == targets) & ignore_mask).float().sum()
        total = ignore_mask.float().sum()
        accuracy = correct / total if total > 0 else 0.0

        # Calculate perplexity
        perplexity = math.exp(loss.item())

        return loss.item(), accuracy.item(), perplexity

    def evaluate_full(self, dataloader, dataset_name):
        self.model.eval()
        total_loss, total_correct, total_tokens = 0, 0, 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids_1 = batch["input_ids_1"].to(self.device)
                attention_mask_1 = batch["attention_mask_1"].to(self.device)
                input_ids_2 = batch["input_ids_2"].to(self.device)
                attention_mask_2 = batch["attention_mask_2"].to(self.device)
                input_ids_3 = batch["input_ids_3"].to(self.device)
                attention_mask_3 = batch["attention_mask_3"].to(self.device)

                logits = self.model(input_ids_1, input_ids_2, input_ids_3,
                                    attention_mask1=attention_mask_1,
                                    attention_mask2=attention_mask_2,
                                    attention_mask3=attention_mask_3)

                targets = input_ids_3.contiguous()

                # Create a mask to ignore pad tokens
                ignore_mask = (targets != self.pad_token_id)

                # Apply the mask to both logits and targets
                filtered_logits = logits.view(-1, logits.size(-1))[ignore_mask.view(-1)]
                filtered_targets = targets.view(-1)[ignore_mask.view(-1)]

                # Calculate loss using the filtered logits and targets
                loss = F.cross_entropy(filtered_logits, filtered_targets, reduction='sum')
                total_loss += loss.item()

                pred = logits.argmax(dim=-1)
                total_correct += ((pred == targets) & ignore_mask).sum().item()
                total_tokens += ignore_mask.sum().item()

        avg_loss = total_loss / total_tokens
        accuracy = total_correct / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {
            "dataset": dataset_name,
            "loss": avg_loss,
            "accuracy": accuracy,
            "perplexity": perplexity
        }

    def log_prediction(self, input_ids_3, logits, step):
        target_sequence = input_ids_3[-1]
        predicted_ids = torch.argmax(logits[-1], dim=-1)

        input_text_1 = self.tokenizer2.decode(target_sequence, skip_special_tokens=True)

        self.logger.info(f"Step {step}, Prediction vs Actual:")
        self.logger.info(f"Input 1: {input_text_1}")

        # Token-wise comparison
        predicted_tokens = self.tokenizer2.convert_ids_to_tokens(predicted_ids)
        target_tokens = self.tokenizer2.convert_ids_to_tokens(target_sequence)

        # Ensure both sequences have the same length for comparison
        max_length = max(len(predicted_tokens), len(target_tokens))
        predicted_tokens = predicted_tokens + [''] * (max_length - len(predicted_tokens))
        target_tokens = target_tokens + [''] * (max_length - len(target_tokens))

        # Create comparison table
        table = "| Index | Predicted Token | Actual Token | Match |\n"
        table += "|-------|-----------------|--------------|-------|\n"
        for i, (pred, target) in enumerate(zip(predicted_tokens, target_tokens)):
            match = "✓" if pred == target else "✗"
            table += f"| {i:5d} | {pred:15s} | {target:12s} | {match:5s} |\n"

        self.logger.info("Token-wise comparison:")
        self.logger.info("\n" + table)

    def train(self, train_dataloader, eval_dataloader, num_epochs, learning_rate, display_interval):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        if self.start_epoch > 0:
            optimizer_state_dict, scheduler_state_dict = self.optimizer_state_dict, self.scheduler_state_dict
            optimizer.load_state_dict(optimizer_state_dict)
            scheduler.load_state_dict(scheduler_state_dict)

        self.logger.info(f"Starting training from epoch {self.start_epoch}")
        global_step = 0

        for epoch in range(self.start_epoch, num_epochs):
            self.model.train()
            total_loss = 0
            for i, batch in enumerate(train_dataloader):
                if batch is None:
                    continue  # Skip this batch
                try:
                    input_ids_1 = batch["input_ids_1"].to(self.device)
                    attention_mask_1 = batch["attention_mask_1"].to(self.device)
                    input_ids_2 = batch["input_ids_2"].to(self.device)
                    attention_mask_2 = batch["attention_mask_2"].to(self.device)
                    input_ids_3 = batch["input_ids_3"].to(self.device)
                    attention_mask_3 = batch["attention_mask_3"].to(self.device)

                    optimizer.zero_grad()

                    # Get the pad and end token IDs
                    pad_token_id = self.pad_token_id
                    end_token_id = self.eos_token_id

                    # Add <pad> to the beginning
                    batch_size, seq_length = input_ids_3.shape
                    new_input_ids3 = torch.full((batch_size, seq_length + 1), pad_token_id, dtype=input_ids_3.dtype,
                                                device=input_ids_3.device)
                    new_input_ids3[:, 1:] = input_ids_3

                    # Update attention mask
                    new_attention_mask3 = torch.zeros((batch_size, seq_length + 1), dtype=attention_mask_3.dtype,
                                                      device=attention_mask_3.device)
                    new_attention_mask3[:, 1:] = attention_mask_3
                    # Log the shape and first sequence of new_input_ids3
                    self.logger.debug(f"new_input_ids3 shape: {new_input_ids3.shape}")
                    self.logger.debug(f"new_input_ids3 first sequence: {new_input_ids3[0].tolist()}")

                    # Update the model call
                    logits = self.model(input_ids_1,
                                        input_ids_2,
                                        new_input_ids3,  # Use the modified input_ids for the third input as well
                                        attention_mask1=attention_mask_1,
                                        attention_mask2=attention_mask_2,
                                        attention_mask3=new_attention_mask3)

                    # Create targets with the same size as new_input_ids3
                    targets = torch.full((batch_size, seq_length + 1), pad_token_id, dtype=input_ids_3.dtype,
                                         device=input_ids_3.device)
                    targets[:, :-1] = new_input_ids3[:, 1:]  # Shift right by one position

                    # Calculate loss only for the last token
                    # Find the last non-padded position for each sequence
                    last_non_pad = attention_mask_3.sum(dim=1) - 1  # Get last valid index

                    # Create a mask that's 1 only at the last token position
                    mask = torch.zeros_like(targets)
                    for idx in range(batch_size):
                        mask[idx, last_non_pad[idx]] = 1

                    # Reshape logits and targets
                    flat_logits = logits.view(-1, logits.size(-1))
                    flat_targets = targets.view(-1)
                    flat_mask = mask.view(-1)

                    # Get only the logits and targets for the last token
                    last_token_logits = flat_logits[flat_mask.bool()]
                    last_token_targets = flat_targets[flat_mask.bool()]

                    # Calculate loss only for last tokens
                    loss = F.cross_entropy(last_token_logits, last_token_targets)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    global_step += 1

                    if global_step % 10 == 0:
                        batch_metrics = self.evaluate_batch(logits, targets)
                        self.logger.info(f"Step {global_step}, Batch metrics: "
                                         f"Loss: {batch_metrics[0]:.4f}, "
                                         f"Accuracy: {batch_metrics[1]:.4f}, "
                                         f"Perplexity: {batch_metrics[2]:.4f}")

                    if global_step % display_interval == 0:
                        self.log_prediction(input_ids_3, logits, global_step)
                        # Log learning rates
                        for idx, param_group in enumerate(optimizer.param_groups):
                            self.logger.info(f"Learning rate (group {idx}): {param_group['lr']:.6f}")

                        # Log gradients
                        self.log_gradients(self.model)

                    print_progress_bar(i + 1, len(train_dataloader), epoch + 1, num_epochs,
                                       prefix='Training:', suffix=f'Loss: {loss.item():.4f}', length=30)

                    if len(eval_dataloader) != 0 and (i + 1) % (len(train_dataloader) // 2) == 0:
                        eval_metrics = self.evaluate_full(eval_dataloader, "Evaluation")
                        self.logger.info(f"Full dataset metrics at epoch {epoch + 1}, step {i + 1}:")
                        self.logger.info(f"  {eval_metrics['dataset']} dataset:")
                        self.logger.info(f"    Loss: {eval_metrics['loss']:.4f}")
                        self.logger.info(f"    Accuracy: {eval_metrics['accuracy']:.4f}")
                        self.logger.info(f"    Perplexity: {eval_metrics['perplexity']:.4f}")

                        if eval_metrics['loss'] < self.best_eval_loss:
                            self.best_eval_loss = eval_metrics['loss']
                            self.save_checkpoint(epoch, optimizer, scheduler, self.best_eval_loss, is_best=True)
                except Exception as e:
                    log_error(self.logger, e, batch)
                    continue  # Skip to the next batch

            avg_loss = total_loss / len(train_dataloader)
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {avg_loss:.4f}")
            self.logger.info(f"Current learning rate: {optimizer.param_groups[0]['lr']:.2e}")

            self.save_checkpoint(epoch, optimizer, scheduler, avg_loss)
            scheduler.step()

        self.logger.info("Training completed!")

    def log_gradients(self, model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.logger.info(f"Gradient norm: {total_norm:.4f}")

    def save_checkpoint(self, epoch, optimizer, scheduler, loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
        }
        if is_best:
            path = os.path.join(self.save_dir, 'best_model.pt')
            self.logger.info(f"New best model saved to {path}")
        else:
            path = os.path.join(self.save_dir, f'model_epoch_{epoch + 1}.pt')
            self.logger.info(f"Model saved to {path}")
        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path):
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_eval_loss = checkpoint['loss']
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        self.optimizer_state_dict, self.scheduler_state_dict = checkpoint['optimizer_state_dict'], checkpoint[
            'scheduler_state_dict']


def train_llm(model, train_dataloader, eval_dataloader, tokenizer1, tokenizer2, num_epochs, learning_rate, device,
              log_dir, save_dir, checkpoint, display_interval=100):
    trainer = Trainer(model, tokenizer1, tokenizer2, device, log_dir, save_dir, checkpoint)
    trainer.train(train_dataloader, eval_dataloader, num_epochs, learning_rate, display_interval)
