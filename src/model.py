import random
import string

import torch
import torch.nn as nn
from transformers import MarianConfig
from transformers.models.marian.modeling_marian import MarianDecoder, MarianEncoder
import torch.nn.functional as F

from auto_encoder import DimensionAlignmentAutoencoder



class CustomLLM(nn.Module):
    def __init__(self,
                 he_en_model,
                 en_he_model,
                 llm_model,
                 align_he_en: DimensionAlignmentAutoencoder=None,
                 align_en_he: DimensionAlignmentAutoencoder=None,
                 tokenizer3=None
                 ):
        super().__init__()

        # Hebrew-English components
        self.he_en_model_encoder = he_en_model.model.encoder

        # First custom transformer layers
        self.embed_tokens = llm_model.base_model.embed_tokens if hasattr(llm_model.base_model, 'embed_tokens') else llm_model.base_model.word_embeddings

        # Create a new decoder config with matching dimensions
        decoder_config = MarianConfig(
            d_model=llm_model.config.hidden_size,
            encoder_attention_heads=he_en_model.config.encoder_attention_heads,
            encoder_ffn_dim=he_en_model.config.encoder_ffn_dim,
            encoder_layers=he_en_model.config.encoder_layers,
            decoder_attention_heads=he_en_model.config.decoder_attention_heads,
            decoder_ffn_dim=he_en_model.config.decoder_ffn_dim,
            decoder_layers=he_en_model.config.decoder_layers,
            max_length=he_en_model.config.max_length,
            vocab_size=he_en_model.config.vocab_size,
            scale_embedding=he_en_model.config.scale_embedding,
            pad_token_id=he_en_model.config.pad_token_id,
            eos_token_id=he_en_model.config.eos_token_id,
            decoder_start_token_id=he_en_model.config.decoder_start_token_id,
        )

        # Dimension alignment layer
        if align_he_en is None:
            self.align_he_en = DimensionAlignmentAutoencoder(
                input_dim=he_en_model.config.d_model,
                target_dim=llm_model.config.hidden_size
            ).encoder
        else:
            self.align_he_en = align_he_en.encoder
        self.align_he_en_input_dim ,self.align_he_en_target_dim = he_en_model.config.d_model, llm_model.config.hidden_size

        self.custom_decoder1 = MarianDecoder(decoder_config)
        self.custom_decoder1.set_input_embeddings(None)

        # LLM layers (main body of the model)
        self.main_model = llm_model.base_model
        self.main_model.set_input_embeddings(None)

        if align_en_he is None:
            self.align_en_he = DimensionAlignmentAutoencoder(
                input_dim=llm_model.config.hidden_size,
                target_dim=en_he_model.config.d_model
            ).encoder
        else:
            self.align_en_he = align_en_he.encoder
        self.align_en_he_input_dim ,self.align_en_he_target_dim = llm_model.config.hidden_size, en_he_model.config.d_model

        # Second custom transformer layers
        self.custom_encoder2 = MarianEncoder(en_he_model.config)
        self.custom_encoder2.set_input_embeddings(None)

        # English-Hebrew components
        self.en_he_decoder = en_he_model.model.decoder
        self.lm_head = en_he_model.lm_head
        self.final_logits_bias = en_he_model.final_logits_bias

        # Freeze layers
        self._freeze_layers()

        # Initialize punct_tokens as None
        self.punct_tokens = None
        # If tokenizer3 is provided during initialization, calculate punct_tokens
        if tokenizer3 is not None:
            self.calculate_punct_tokens(tokenizer3)

    def calculate_punct_tokens(self, tokenizer3):
        """Calculate punctuation token IDs once and store them."""
        self.punct_tokens = set()
        for punct in string.punctuation:
            tokens = tokenizer3.encode(punct, add_special_tokens=False)
            self.punct_tokens.update(tokens)
            for token_id in range(tokenizer3.vocab_size):
                token = tokenizer3.decode([token_id])
                if any(p in token for p in string.punctuation):
                    self.punct_tokens.add(token_id)

    def _freeze_layers(self):
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze the alignment and transformation layers
        for param in self.align_he_en.parameters():
            param.requires_grad = True
        for param in self.align_en_he.parameters():
            param.requires_grad = True

        # Unfreeze the custom decoder and encoder
        for param in self.custom_decoder1.parameters():
            param.requires_grad = True
        for param in self.custom_encoder2.parameters():
            param.requires_grad = True


    def forward(self, input_ids1, input_ids2, input_ids3, attention_mask1=None, attention_mask2=None, attention_mask3=None, llm=None, tokenizer2=None):
        # Ensure input tensors are of the correct data type
        input_ids1 = input_ids1.long()
        input_ids2 = input_ids2.long()
        input_ids3 = input_ids3.long()

        he_en_encoder_output = self.he_en_model_encoder(
            input_ids=input_ids1,
            attention_mask=attention_mask1,
        ).last_hidden_state

        # Align Hebrew-English encoder output dimensions with LLM
        he_en_encoder_output = self.encode(
            self.align_he_en,
            self.align_he_en_input_dim,
            self.align_he_en_target_dim,
            he_en_encoder_output,
            attention_mask1
        )

        # Get embeddings for the second input
        inputs_embeds2 = self.embed_tokens(input_ids2)

        # Process through custom decoder
        x = self.custom_decoder1(
            inputs_embeds=inputs_embeds2,
            attention_mask=attention_mask2,
            encoder_hidden_states=he_en_encoder_output,
            encoder_attention_mask=attention_mask1,
        )[0]

        # If llm is provided, process and print intermediate output
        if llm is not None and tokenizer2 is not None:
            intermediate_logits = llm.lm_head(x)
            intermediate_tokens = torch.argmax(intermediate_logits, dim=-1)
            intermediate_text = tokenizer2.decode(intermediate_tokens[0], skip_special_tokens=True)
            print("Intermediate output (before main model):", intermediate_text)

        x = self.main_model(inputs_embeds=x, attention_mask=attention_mask2).last_hidden_state

        # If llm is provided, process and print intermediate output again
        if llm is not None and tokenizer2 is not None:
            intermediate_logits = llm.lm_head(x)
            intermediate_tokens = torch.argmax(intermediate_logits, dim=-1)
            intermediate_text = tokenizer2.decode(intermediate_tokens[0], skip_special_tokens=True)
            print("Intermediate output (after main model):", intermediate_text)

        x = self.encode(
            self.align_en_he,
            self.align_en_he_input_dim,
            self.align_en_he_target_dim,
            x,
            attention_mask2
        )

        x = self.custom_encoder2(
            inputs_embeds=x,
            attention_mask=attention_mask2
        ).last_hidden_state

        # Teacher forcing
        x = self.en_he_decoder(
            encoder_hidden_states=x,
            encoder_attention_mask=attention_mask2,
            input_ids=input_ids3,
            attention_mask=attention_mask3,
        )[0]

        logits = self.lm_head(x) + self.final_logits_bias

        return logits

    def prepare_inputs(self, sentence, he_en_model, tokenizer1, tokenizer2, tokenizer3, device):
        # Tokenizer 1: Hebrew sentence
        inputs_1 = tokenizer1(sentence, return_tensors="pt")
        input_ids_1 = inputs_1["input_ids"].to(device)
        attention_mask_1 = inputs_1["attention_mask"].to(device)

        # Translate the sentence
        with torch.no_grad():
            translated_ids = he_en_model.generate(input_ids=input_ids_1, attention_mask=attention_mask_1)
        translated_sentence = tokenizer1.decode(translated_ids[0], skip_special_tokens=True)

        # Tokenizer 2: English translation
        inputs_2 = tokenizer2(translated_sentence, return_tensors="pt")
        input_ids_2 = inputs_2["input_ids"].to(device)
        attention_mask_2 = inputs_2["attention_mask"].to(device)

        # Tokenizer 3: Full Hebrew sentence
        inputs_3 = tokenizer3(text_target=sentence, return_tensors="pt")
        input_ids_3 = inputs_3["input_ids"][:, :-1].to(device)
        attention_mask_3 = inputs_3["attention_mask"][:, :-1].to(device)

        # Add <pad> to the beginning
        batch_size, seq_length = input_ids_3.shape
        new_input_ids3 = torch.full((batch_size, seq_length + 1), tokenizer3.pad_token_id, dtype=input_ids_3.dtype,
                                    device=input_ids_3.device)
        new_input_ids3[:, 1:] = input_ids_3

        # Update attention mask
        new_attention_mask3 = torch.zeros((batch_size, seq_length + 1), dtype=attention_mask_3.dtype,
                                          device=attention_mask_3.device)
        new_attention_mask3[:, 1:] = attention_mask_3

        return {
            "input_ids_1": input_ids_1,
            "attention_mask_1": attention_mask_1,
            "input_ids_2": input_ids_2,
            "attention_mask_2": attention_mask_2,
            "input_ids_3": new_input_ids3,
            "attention_mask_3": new_attention_mask3
        }

    def generate(self, sentence, he_en_model, tokenizer1, tokenizer2, tokenizer3, device, max_length=50,
                 temperature=1.0, top_k=50, top_p=0.95, llm=None, punct_prob=0.2, streamer=None):
        """
        Generate text with controlled punctuation probability and streaming output.

        Args:
            [previous args...]
            streamer (TextStreamer, optional): Streamer for real-time text output
        """
        self.eval()

        # Calculate punct_tokens only if not already calculated
        if self.punct_tokens is None:
            self.calculate_punct_tokens(tokenizer3)

        # Prepare initial input tensors
        inputs = self.prepare_inputs(sentence, he_en_model, tokenizer1, tokenizer2, tokenizer3, device)
        input_ids1 = inputs["input_ids_1"]
        input_ids2 = inputs["input_ids_2"]
        attention_mask1 = inputs["attention_mask_1"]
        attention_mask2 = inputs["attention_mask_2"]

        # Initialize the output sequence
        generated_ids = inputs["input_ids_3"].clone()
        attention_mask3 = inputs["attention_mask_3"].clone()

        if streamer is not None:
            # Stream initial tokens
            streamer.put(generated_ids[0])

        for _ in range(max_length):
            # Forward pass
            with torch.no_grad():
                outputs = self(
                    input_ids1=input_ids1,
                    input_ids2=input_ids2,
                    input_ids3=generated_ids,
                    attention_mask1=attention_mask1,
                    attention_mask2=attention_mask2,
                    attention_mask3=attention_mask3,
                    llm=llm,
                    tokenizer2=tokenizer2
                )

            next_token_logits = outputs[:, -1, :]

            # Randomly decide whether to allow punctuation for this token
            allow_punct = random.random() < punct_prob

            if not allow_punct:
                for punct_id in self.punct_tokens:
                    next_token_logits[:, punct_id] = float('-inf')

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                top_k = min(top_k, next_token_logits.size(-1))
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < top_k_logits[:, [-1]]] = float('-inf')

            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                mask = torch.ones_like(next_token_logits, dtype=torch.bool)
                mask.scatter_(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[mask] = float('-inf')

            # Sample the next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append the new token to the sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            attention_mask3 = torch.cat([attention_mask3, torch.ones_like(next_token)], dim=1)

            if streamer is not None:
                # Stream the current token
                streamer.put(generated_ids[0])

            # Update inputs for next iteration
            current_sentence = tokenizer3.decode(generated_ids[0], skip_special_tokens=True)
            inputs = self.prepare_inputs(current_sentence, he_en_model, tokenizer1, tokenizer2, tokenizer3, device)
            input_ids1 = inputs["input_ids_1"]
            input_ids2 = inputs["input_ids_2"]
            attention_mask1 = inputs["attention_mask_1"]
            attention_mask2 = inputs["attention_mask_2"]

            # Check if we've generated an EOS token
            if next_token.item() == tokenizer3.eos_token_id:
                break

        if streamer is not None:
            # End the streaming
            streamer.end()

        return generated_ids

    @classmethod
    def load_pretrained(cls, checkpoint_path, he_en_model, en_he_model, llm_model, device, tokenizer3):
        # Load only the model weights
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

        if 'model_state_dict' in checkpoint:
            # This is a checkpoint saved by our Trainer
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # This might be a checkpoint saved by pytorch-lightning or other libraries
            state_dict = checkpoint['state_dict']
        else:
            # Assume it's just the state dict
            state_dict = checkpoint

        model = cls(he_en_model, en_he_model, llm_model, tokenizer3=tokenizer3)

        # Load the state dict
        model.load_state_dict(state_dict)

        return model.to(device)

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

    def encode(self, encoder, input_dim, target_dim, x, attention_mask=None):
        """
        Encode input to target dimension.

        Args:
            target_dim:
            input_dim:
            encoder:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)
        """
        batch_size = None
        seq_len = None

        # Handle 3D input
        if len(x.shape) == 3:
            batch_size, seq_len, _ = x.shape
            # Reshape to 2D
            x = x.contiguous().view(-1, input_dim)

        # Apply encoder
        encoded = encoder(x)

        # Reshape back to 3D if needed
        if batch_size is not None:
            encoded = encoded.view(batch_size, seq_len, target_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            encoded = self.apply_attention_mask(encoded, attention_mask)

        return encoded


class TextStreamer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.current_text = ""

    def put(self, token_ids):
        """Process new tokens and print the updated text."""
        # Decode new tokens
        new_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)

        # Only print the difference between current and new text
        if len(new_text) > len(self.current_text):
            diff = new_text[len(self.current_text):]
            print(diff, end="", flush=True)
            self.current_text = new_text

    def end(self):
        """Called at the end of generation."""
        print()  # New line at the end