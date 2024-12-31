import argparse
import os
import traceback

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, MarianMTModel, MarianTokenizer, AutoModelForCausalLM

from model import CustomLLM
from dataset import create_dataloaders
from training import train_llm
from utils import evaluate_text, calculate_metrics
from auto_encoder import DimensionAlignmentAutoencoder, AutoencoderPreTrainer


def setup_autoencoders(args, he_en_model, llm_model, en_he_model, tokenizer1, tokenizer2, device, train_texts):
    """
    Setup both autoencoders: either load pre-trained or create new untrained ones
    """
    # Create the autoencoders
    autoencoder_he_en = DimensionAlignmentAutoencoder(
        input_dim=he_en_model.config.d_model,
        target_dim=llm_model.config.hidden_size
    ).to(device)

    autoencoder_en_he = DimensionAlignmentAutoencoder(
        input_dim=llm_model.config.hidden_size,
        target_dim=en_he_model.config.d_model
    ).to(device)

    # Create directories for both autoencoders only if training
    if args.train_autoencoder:
        os.makedirs(os.path.join(args.save_dir, 'autoencoder_he_en'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'autoencoder_en_he'), exist_ok=True)

    # Handle he_en autoencoder
    if args.load_pretrained_autoencoder and args.autoencoder_he_en_path and os.path.exists(args.autoencoder_he_en_path):
        print(f"Loading pre-trained he-en autoencoder from {args.autoencoder_he_en_path}")
        checkpoint = torch.load(args.autoencoder_he_en_path, map_location=device, weights_only=True)
        autoencoder_he_en.load_state_dict(checkpoint['model_state_dict'])
    elif args.train_autoencoder:
        print("Training new he-en autoencoder...")
        trainer_he_en = AutoencoderPreTrainer(
            autoencoder=autoencoder_he_en,
            source_model=he_en_model,
            tokenizer=tokenizer1,
            device=device,
            model_type='he_en'
        )
        best_loss_he_en = trainer_he_en.train(
            sentences=train_texts,
            num_epochs=args.autoencoder_epochs,
            batch_size=args.autoencoder_batch_size,
            learning_rate=args.autoencoder_lr,
            save_dir=os.path.join(args.save_dir, 'autoencoder_he_en')
        )
        print(f"He-En autoencoder training completed with best loss: {best_loss_he_en}")
    else:
        print("Using untrained he-en autoencoder")

    # Handle en_he autoencoder
    if args.load_pretrained_autoencoder and args.autoencoder_en_he_path and os.path.exists(args.autoencoder_en_he_path):
        print(f"Loading pre-trained en-he autoencoder from {args.autoencoder_en_he_path}")
        checkpoint = torch.load(args.autoencoder_en_he_path, map_location=device, weights_only=True)
        autoencoder_en_he.load_state_dict(checkpoint['model_state_dict'])
    elif args.train_autoencoder:
        print("Training new en-he autoencoder...")

        # Get training sentences only if training
        if args.use_english_dataset:
            print(f"Loading English dataset (using {args.english_dataset_size} sentences)...")
            try:
                dataset = load_dataset("agentlans/high-quality-english-sentences")
                english_texts = dataset['train']['text'][:args.english_dataset_size]
                print(f"Loaded {len(english_texts)} sentences from the English dataset")
            except Exception as e:
                print(f"Error loading English dataset: {str(e)}")
                print("Falling back to translated texts...")
                english_texts = translate_texts(train_texts, he_en_model, tokenizer1, device)
        else:
            english_texts = translate_texts(train_texts, he_en_model, tokenizer1, device)

        trainer_en_he = AutoencoderPreTrainer(
            autoencoder=autoencoder_en_he,
            source_model=llm_model,
            tokenizer=tokenizer2,
            device=device,
            model_type='llm'
        )
        best_loss_en_he = trainer_en_he.train(
            sentences=english_texts,
            num_epochs=args.autoencoder_epochs,
            batch_size=args.autoencoder_batch_size,
            learning_rate=args.autoencoder_lr,
            save_dir=os.path.join(args.save_dir, 'autoencoder_en_he')
        )
        print(f"En-He autoencoder training completed with best loss: {best_loss_en_he}")
    else:
        print("Using untrained en-he autoencoder")

    # Load the best model weights after training (only if they exist)
    if args.train_autoencoder:
        for autoencoder, path in [
            (autoencoder_he_en, os.path.join(args.save_dir, 'autoencoder_he_en', 'best_autoencoder.pt')),
            (autoencoder_en_he, os.path.join(args.save_dir, 'autoencoder_en_he', 'best_autoencoder.pt'))]:
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location=device, weights_only=True)
                autoencoder.load_state_dict(checkpoint['model_state_dict'])
                autoencoder = autoencoder.to(device)

    return autoencoder_he_en, autoencoder_en_he


def translate_texts(texts, he_en_model, tokenizer1, device):
    """Helper function to translate Hebrew texts to English"""
    print("Translating training texts to English...")
    english_texts = []
    for sentence in tqdm(texts):
        inputs = tokenizer1(sentence, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            translated_ids = he_en_model.generate(**inputs)
            translated_text = tokenizer1.decode(translated_ids[0], skip_special_tokens=True)
            english_texts.append(translated_text)
    return english_texts


def add_autoencoder_args(parser):
    group = parser.add_argument_group('Autoencoder')

    # Control flags
    group.add_argument('--train_autoencoder', action='store_true', default=False,
                       help='Whether to train new autoencoders')
    group.add_argument('--load_pretrained_autoencoder', action='store_true',
                       help='Whether to load pre-trained autoencoders')

    # Paths for both autoencoders
    group.add_argument('--autoencoder_he_en_path', type=str, default=None,
                       help='Path to pre-trained he-en autoencoder checkpoint')
    group.add_argument('--autoencoder_en_he_path', type=str, default=None,
                       help='Path to pre-trained en-he autoencoder checkpoint')

    # Dataset configuration
    group.add_argument('--use_english_dataset', action='store_true',
                       help='Whether to use the English dataset for en-he autoencoder training')
    group.add_argument('--english_dataset_size', type=int, default=300000,
                       help='Number of sentences to use from the English dataset')

    # Training parameters
    group.add_argument('--autoencoder_epochs', type=int, default=100,
                       help='Number of epochs for autoencoder training')
    group.add_argument('--autoencoder_batch_size', type=int, default=32,
                       help='Batch size for autoencoder training')
    group.add_argument('--autoencoder_lr', type=float, default=1e-3,
                       help='Learning rate for autoencoder training')

def main():
    parser = argparse.ArgumentParser(description="Train a custom LLM model")
    parser.add_argument("--he-en-model", type=str, default="Helsinki-NLP/opus-mt-tc-big-he-en",
                        help="Name or path of the Hebrew-English model")
    parser.add_argument("--en-he-model", type=str, default="Helsinki-NLP/opus-mt-en-he",
                        help="Name or path of the English-Hebrew model")
    parser.add_argument("--llm-model", type=str, default="facebook/opt-350m",
                        help="Name or path of the LLM model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu)")
    parser.add_argument("--data-file", type=str,
                        help="Path to the data file containing Hebrew sentences")
    parser.add_argument("--num-epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning rate for training")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory to save log files")
    parser.add_argument("--save-dir", type=str, default="model_checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--train-split", type=float, default=0.8,
                        help="Proportion of data to use for training")
    parser.add_argument("--display-interval", type=int, default=100,
                        help="Display interval to log the last sentence in the batch")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint file to resume training from")
    parser.add_argument("--pretrained-model", type=str, default=None,
                        help="Path to a pretrained CustomLLM model to load and fine-tune")
    parser.add_argument("--lr-plot-path", type=str, default="lr_finder_plot.png",
                        help="Path to save the learning rate finder plot")
    parser.add_argument("--generate", action="store_true", help="Run in generation mode")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for text generation sampling")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k value for text generation sampling")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Top-p (nucleus sampling) value for text generation")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation on the test set")
    parser.add_argument("--evaluate-text", type=str,
                        help="Evaluate a specific text input")
    parser.add_argument("--eval-batch-size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed token-by-token comparison")
    add_autoencoder_args(parser)  # Add autoencoder-specific arguments

    args = parser.parse_args()

    # Validate arguments
    if not args.generate and args.data_file is None:
        parser.error("--data-file is required when not in generation mode")

    # Load models
    he_en_model = MarianMTModel.from_pretrained(args.he_en_model).to(args.device)
    en_he_model = MarianMTModel.from_pretrained(args.en_he_model).to(args.device)
    llm_model = AutoModelForCausalLM.from_pretrained(args.llm_model).to(args.device)

    # Load tokenizers
    tokenizer1 = MarianTokenizer.from_pretrained(args.he_en_model)
    tokenizer2 = AutoTokenizer.from_pretrained(args.llm_model)
    tokenizer3 = MarianTokenizer.from_pretrained(args.en_he_model)

    if args.pretrained_model:
        # Load the pretrained CustomLLM
        print(f"Loading pretrained model from {args.pretrained_model}")
        custom_llm = CustomLLM.load_pretrained(
            args.pretrained_model,
            he_en_model,
            en_he_model,
            llm_model,
            args.device,
            tokenizer3=tokenizer3
        )
    else:
        # Load sentences from the data file
        with open(args.data_file, 'r', encoding='utf-8') as f:
            sentences = f.readlines()
        sentences = [line.strip() for line in sentences if line.strip()]

        # Setup both autoencoders
        autoencoder_he_en, autoencoder_en_he = setup_autoencoders(
            args=args,
            he_en_model=he_en_model,
            llm_model=llm_model,
            en_he_model=en_he_model,
            tokenizer1=tokenizer1,
            tokenizer2=tokenizer2,
            device=args.device,
            train_texts=sentences
        )
        # Create a new CustomLLM with both autoencoders
        custom_llm = CustomLLM(
            he_en_model,
            en_he_model,
            llm_model,
            align_he_en=autoencoder_he_en,
            align_en_he=autoencoder_en_he,
            tokenizer3=tokenizer3
        )

    # Move the model to the specified device
    custom_llm = custom_llm.to(args.device)

    if args.evaluate:
        # Load the full dataset
        with open(args.data_file, 'r', encoding='utf-8') as f:
            sentences = f.readlines()
        sentences = [line.strip() for line in sentences if line.strip()]

        # Create dataloaders with evaluation batch size
        eval_dataloader, _ = create_dataloaders(
            sentences,
            he_en_model,
            tokenizer1,
            tokenizer2,
            tokenizer3,
            batch_size=args.eval_batch_size,
            train_split=1,
            device=args.device
        )

        # Run evaluation
        metrics = calculate_metrics(
            custom_llm,
            eval_dataloader,
            tokenizer3,
            args.device,
            verbose=args.verbose
        )

        print("\nEvaluation Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Perplexity: {metrics['perplexity']:.4f}")

    elif args.evaluate_text:
        # Evaluate single text input
        results = evaluate_text(
            custom_llm,
            args.evaluate_text,
            he_en_model,
            tokenizer1,
            tokenizer2,
            tokenizer3,
            args.device
        )

        print("\nText Evaluation Results:")
        print(f"Input text: {results['input_text']}")
        print(f"Generated text: {results['generated_text']}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Perplexity: {results['perplexity']:.4f}")

    elif args.generate:
        print("Entering generation mode. Type 'quit' to exit.")
        while True:
            sentence = input("Enter a Hebrew sentence: ")
            if sentence.lower() == 'quit':
                break
            try:
                generated_ids = custom_llm.generate(
                    sentence,
                    he_en_model,
                    tokenizer1,
                    tokenizer2,
                    tokenizer3,
                    args.device,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    llm=None,
                )
                generated_text = tokenizer3.decode(generated_ids[0], skip_special_tokens=True)
                print(f"Generated text:\n{generated_text}")
            except Exception as e:
                print(f"An error occurred during generation: {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
    else:
        # Create dataloaders
        train_dataloader, eval_dataloader = create_dataloaders(sentences, he_en_model, tokenizer1, tokenizer2,
                                                               tokenizer3,
                                                               batch_size=args.batch_size, train_split=args.train_split,
                                                               device=args.device)

        # Set CUDA_LAUNCH_BLOCKING for better error reporting
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        # Train the model
        train_llm(custom_llm, train_dataloader, eval_dataloader, tokenizer1, tokenizer3,
                  num_epochs=args.num_epochs,
                  learning_rate=args.learning_rate,
                  device=args.device,
                  log_dir=args.log_dir,
                  save_dir=args.save_dir,
                  checkpoint=args.checkpoint,
                  display_interval=args.display_interval)


if __name__ == "__main__":
    main()
