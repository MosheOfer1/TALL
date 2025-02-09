import re
import logging
from datetime import datetime
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers.data.data_collator import DataCollatorMixin
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit
import torch
from typing import List, Dict, Optional
from torch.utils.data import Dataset as TorchDataset
from create_dataset_for_prompt_tuning import preprocess_text, translate, check_using_input_and_label, \
    heb_to_eng_tokenizer, heb_to_eng_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'translation_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Clean text by removing punctuation and unexpected characters."""
    cleaned_text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    return cleaned_text.strip()


@dataclass
class DynamicPaddingCollator(DataCollatorMixin):
    """Custom data collator that performs dynamic padding per batch."""
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        # Find max length in the batch for inputs and labels
        max_input_length = max(len(feature["input_ids"]) for feature in features)
        max_label_length = max(len(feature["labels"]) for feature in features)

        # Log the dynamic padding lengths
        logger.debug(f"Batch padding - Max input length: {max_input_length}, Max label length: {max_label_length}")

        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

        # Pad all sequences in the batch to the max length found
        for feature in features:
            # Pad input_ids
            input_padding_length = max_input_length - len(feature["input_ids"])
            padded_input_ids = feature["input_ids"] + [self.tokenizer.pad_token_id] * input_padding_length
            batch["input_ids"].append(padded_input_ids)

            # Pad attention_mask
            attention_mask = feature["attention_mask"] + [0] * input_padding_length
            batch["attention_mask"].append(attention_mask)

            # Pad labels
            label_padding_length = max_label_length - len(feature["labels"])
            padded_labels = feature["labels"] + [-100] * label_padding_length  # Use -100 for label padding
            batch["labels"].append(padded_labels)

        # Convert lists to tensors
        batch = {k: torch.tensor(v) for k, v in batch.items()}

        return batch


class OnTheFlyTranslationDataset(TorchDataset):
    def __init__(self, sentences: List[str], tokenizer, cache_size: int = 1000):
        """Initialize the dataset with sentences and tokenizer."""
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.cache_size = cache_size
        self.cache = {}
        self.processed_count = 0
        self.failed_count = 0
        self.cache_hits = 0

        logger.info(f"Initialized dataset with {len(sentences)} sentences and cache size {cache_size}")

    def process_sentence(self, sentence: str, idx: int) -> Optional[Dict]:
        """Process a single sentence into input and label."""
        try:
            words = preprocess_text(sentence).split()
            if len(words) < 2:
                logger.debug(f"Sentence {idx} too short: {sentence}")
                return None

            heb_input, heb_label = " ".join(words[:-1]), words[-1]
            logger.debug(f"Processing sentence {idx}:")
            logger.debug(f"  Hebrew input: {heb_input}")
            logger.debug(f"  Hebrew label: {heb_label}")

            # Translate Hebrew to English
            eng_input = translate(heb_input, heb_to_eng_tokenizer, heb_to_eng_model)[0]
            eng_label = translate(heb_label, heb_to_eng_tokenizer, heb_to_eng_model)[0].lower()

            if not eng_input or not eng_label:
                logger.warning(f"Translation failed for sentence {idx}")
                return None

            logger.debug(f"  English input: {eng_input}")
            logger.debug(f"  English label: {eng_label}")

            # Verify translation
            match, message = check_using_input_and_label(eng_input, eng_label, heb_label)
            if not match:
                logger.warning(f"Translation verification failed for sentence {idx}: {message}")
                return None

            logger.debug(f"  Translation verified successfully: {message}")

            cleaned_input = clean_text(eng_input)
            cleaned_label = clean_text(eng_label)

            logger.debug(f"  Cleaned input: {cleaned_input}")
            logger.debug(f"  Cleaned label: {cleaned_label}")

            self.processed_count += 1
            if self.processed_count % 100 == 0:
                logger.info(f"Successfully processed {self.processed_count} sentences")
                logger.info(f"Failed count: {self.failed_count}")
                logger.info(f"Cache hits: {self.cache_hits}")

            return {
                "eng_input": cleaned_input,
                "eng_label": cleaned_label,
                "original": {
                    "heb_input": heb_input,
                    "heb_label": heb_label,
                    "eng_input": eng_input,
                    "eng_label": eng_label
                }
            }

        except Exception as e:
            logger.error(f"Error processing sentence {idx}: {str(e)}")
            self.failed_count += 1
            return None

    def tokenize_example(self, example: Dict) -> Dict:
        """Tokenize the processed example without padding."""
        try:
            # Tokenize input sequence
            tokenized_inputs = self.tokenizer(
                example["eng_input"],
                truncation=True,
                add_special_tokens=True,
            )

            # Tokenize label sequence
            tokenized_labels = self.tokenizer(
                example["eng_label"],
                truncation=True,
                add_special_tokens=True,
            )

            # Return the tokenized sequences without padding
            return {
                "input_ids": tokenized_inputs["input_ids"],
                "attention_mask": tokenized_inputs["attention_mask"],
                "labels": tokenized_labels["input_ids"]
            }

        except Exception as e:
            logger.error(f"Tokenization error: {str(e)}")
            raise

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single example, processing it if not in cache."""
        if idx in self.cache:
            self.cache_hits += 1
            logger.debug(f"Cache hit for index {idx}")
            return self.cache[idx]

        logger.debug(f"Processing new item at index {idx}")

        # Process new example
        sentence = self.sentences[idx]
        processed = self.process_sentence(sentence, idx)

        # If processing failed, try the next sentence
        attempts = 0
        original_idx = idx
        while processed is None and attempts < 5:
            attempts += 1
            idx = (idx + 1) % len(self.sentences)
            logger.warning(f"Attempt {attempts}: Failed to process sentence {original_idx}, trying {idx}")
            sentence = self.sentences[idx]
            processed = self.process_sentence(sentence, idx)

        if processed is None:
            logger.error(f"All attempts failed for index {original_idx}, using dummy example")
            processed = {
                "eng_input": "dummy input",
                "eng_label": "dummy label"
            }

        # Tokenize the processed example
        tokenized = self.tokenize_example(processed)

        # Update cache
        if len(self.cache) >= self.cache_size:
            removed_key = next(iter(self.cache))
            self.cache.pop(removed_key)
            logger.debug(f"Cache full, removed item {removed_key}")

        self.cache[idx] = tokenized
        logger.debug(f"Added item {idx} to cache")

        return tokenized


def setup_prompt_tuning(model_name: str, num_virtual_tokens: int = 4) -> tuple:
    """Set up the model, tokenizer, and prompt tuning configuration."""
    logger.info(f"Setting up prompt tuning with {model_name}")

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    logger.info("Configuring prompt tuning...")
    tuning_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.RANDOM,
        num_virtual_tokens=num_virtual_tokens,
        tokenizer_name_or_path=model_name
    )

    logger.info("Creating PEFT model...")
    peft_model = get_peft_model(model, tuning_config)

    return tokenizer, peft_model



def main():
    logger.info("Starting translation model training")

    # Configuration
    model_name = "bigscience/bloomz-560m"
    input_file = "../data_sets/ynet_train_256k.txt"
    output_dir = "./word_gen_model"

    # Read sentences
    logger.info(f"Reading sentences from {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        sentences = f.read().strip().splitlines()
    logger.info(f"Loaded {len(sentences)} sentences")

    # Setup model and tokenizer
    tokenizer, peft_model = setup_prompt_tuning(model_name)

    # Create dataset with on-the-fly processing
    logger.info("Creating dataset...")
    dataset = OnTheFlyTranslationDataset(sentences, tokenizer)

    # Create custom data collator with dynamic padding
    data_collator = DynamicPaddingCollator(tokenizer=tokenizer)

    # Training arguments
    logger.info("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        # gradient_accumulation_steps=8,
        learning_rate=0.005,
        num_train_epochs=5,
        save_steps=500,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
    )

    # Create and start trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train()

    # Save the model
    output_model_dir = "./ynet_256k_model"
    logger.info(f"Saving model to {output_model_dir}")
    peft_model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)

    logger.info("Training complete!")
    logger.info(f"Final statistics:")
    logger.info(f"  Processed examples: {dataset.processed_count}")
    logger.info(f"  Failed examples: {dataset.failed_count}")
    logger.info(f"  Cache hits: {dataset.cache_hits}")


if __name__ == "__main__":
    main()