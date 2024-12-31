# TALL: Trainable Architecture Enhanced LLMs in Low-Resource Languages

**TALL** focuses on enhancing LLM performance for low-resource languages, especially Hebrew, using custom transformers and training pipelines.

---

## Features

- **Custom Transformers**: Integrates translation models and LLMs for end-to-end sentence processing.
- **Training Pipelines**: Fine-tune and evaluate LLMs with Hebrew-English datasets.
- **Evaluation Metrics**: Includes accuracy and perplexity calculation for multiple approaches.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MosheOfer1/TALL.git
   cd TALL
   ```
2. **Set Up a Python Environment**:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```
3. **Install Dependencies**:
  ```bash
  python -m venv venv
  pip install -r requirements.txt
  ```

## Usage
1. **Evaluating NLP Models**
   To evaluate your models on a test dataset:
   ```bash
   python evaluator.py --test-file data/test_sentences.txt \
    --log-dir logs \
    --output-file results/model_comparison.csv \
    --device cuda \
    --he-to-en-model Helsinki-NLP/opus-mt-tc-big-he-en \
    --en-to-he-model Helsinki-NLP/opus-mt-en-he \
    --llm-model bigscience/bloomz-560m \
    --use-direct --use-naive --use-soft-prompt \
    --soft-prompt-model models/soft_prompt_model
   ```
2. **Training Custom LLM Models**:
   To train a new model with autoencoders:
    ```bash
    python train_model.py --data-file data/train_sentences.txt \
    --device cuda \
    --he-en-model Helsinki-NLP/opus-mt-tc-big-he-en \
    --en-he-model Helsinki-NLP/opus-mt-en-he \
    --llm-model facebook/opt-350m \
    --num-epochs 10 \
    --batch-size 16 \
    --learning-rate 5e-5 \
    --log-dir logs \
    --save-dir checkpoints \
    --train-autoencoder \
    --autoencoder-epochs 50 \
    --autoencoder-lr 1e-3
    ```
3. **Generating Predictions**:
  To interactively generate predictions:
    ```bash
    python train_model.py --generate \
    --he-en-model Helsinki-NLP/opus-mt-tc-big-he-en \
    --en-he-model Helsinki-NLP/opus-mt-en-he \
    --llm-model facebook/opt-350m
    ```

## Supported Approaches

### 1. **Direct Hebrew Approach**
Processes Hebrew sentences directly using a Hebrew-trained language model. No translation is performed in this approach.

### 2. **Naive Approach**
- Translates the Hebrew sentence (excluding the last word) to English using a Hebrew-to-English translation model.
- Generates a completion in English using an LLM.
- Translates the completed English sentence back to Hebrew using an English-to-Hebrew translation model.

### 3. **Soft Prompt Approach**
- Uses pre-trained soft prompts to guide an LLM for generating sentence completions.
- The Hebrew sentence is translated to English, processed by the LLM with soft prompts, and then translated back to Hebrew.

### 4. **Fine-Tuned Approach**
- Leverages a fine-tuned version of a language model trained specifically on Hebrew sentences to predict the next word in the sequence.

### 5. **Custom Model Approach**
- Combines multiple models (translation models and LLMs) and integrates custom components, including autoencoders, for end-to-end sentence processing.
- Processes Hebrew sentences through a pipeline that aligns the dimensions between models for improved accuracy and performance.

---

## Results

The table below compares three approaches (`direct_hebrew`, `naive`, and `custom_model`) for predicting the final word of a Hebrew sentence:

**Original Sentence**: היסטוריה קופה אמריקה היא טורניר הכדורגל הבינלאומי הוותיק ביותר בעולם  
**Truncated Sentence**: היסטוריה קופה אמריקה היא טורניר הכדורגל הבינלאומי הוותיק ביותר  

| Approach       | Predicted Word | Actual Word | Is Correct |
|----------------|----------------|-------------|------------|
| direct_hebrew  | של            | בעולם       | False      |
| naive          | ב             | בעולם       | False      |
| TALL   | בעולם         | בעולם       | True       |

This table highlights how the `TALL` successfully predicts the correct word, while the other approaches do not.

---

