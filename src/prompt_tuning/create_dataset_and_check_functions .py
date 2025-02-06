import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import string

# Load the Helsinki-NLP models for Hebrew to English and English to Hebrew
heb_to_eng_model_name = 'Helsinki-NLP/opus-mt-tc-big-he-en'
eng_to_heb_model_name = 'Helsinki-NLP/opus-mt-en-he'

# Initialize the tokenizers and models
heb_to_eng_tokenizer = MarianTokenizer.from_pretrained(heb_to_eng_model_name)
heb_to_eng_model = MarianMTModel.from_pretrained(heb_to_eng_model_name)

eng_to_heb_tokenizer = MarianTokenizer.from_pretrained(eng_to_heb_model_name)
eng_to_heb_model = MarianMTModel.from_pretrained(eng_to_heb_model_name)


def translate(text, tokenizer, model, num_return_sequences=1, max_length=100):
    """
    Translates the text using the specified tokenizer and model.

    Parameters:
    - text: The input text to be translated.
    - tokenizer: The Marian tokenizer for the translation model.
    - model: The Marian model for translation.
    - num_return_sequences: Number of translation sequences to return.
    - max_length: Maximum length of the translated text.

    Returns:
    - A list of translated texts.
    """
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = model.generate(**inputs, num_return_sequences=num_return_sequences, num_beams=3,
                                           max_length=max_length)
        translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
        return translated_texts
    except Exception as e:
        print(f"Translation error: {e}")
        return [""]

def check_using_input_and_label(eng_input, eng_label, heb_label):
    """
    Appends English input and label then translates it back to Hebrew and checks if the last word in the sentence
    it matches the Hebrew label.

    Returns a boolean indicating whether the translation matches and a message.
    """
    eng_sentence = preprocess_text(eng_input) + " " + preprocess_text(eng_label)
    back_to_hebrew = translate(eng_sentence, eng_to_heb_tokenizer, eng_to_heb_model)[0]
    back_to_hebrew_words = back_to_hebrew.split()

    if preprocess_text(back_to_hebrew_words[-1]) == preprocess_text(heb_label):
        return True, "Match found using translating English sentence."

    return False, f"Mismatch found using translating English sentence: translated sentence: {back_to_hebrew}"


def check_using_input_and_label_first_tran_label(eng_input, eng_label, heb_label, max_iterations=5):
    """
    First the label is translated and then checked if it is more than one word in Hebrew then remove last word from the English label
    until one word is obtained in Hebrew.
    Translates the English input and label back to Hebrew and checks if it matches the Hebrew label.

    max_iterations (int): The maximum number of iterations allowed to prevent infinite loop.

    Returns:
        tuple: A boolean indicating whether the translation matches and a message.
    """
    label_back_to_hebrew = translate(eng_label, eng_to_heb_tokenizer, eng_to_heb_model)[0]

    iteration = 0  # Initialize a counter for iterations
    print(f"label_back_to_hebrew: {label_back_to_hebrew}")
    # Keep reducing the label until only one word remains in Hebrew translation or max_iterations reached
    while len(label_back_to_hebrew.split()) > 1 and iteration < max_iterations:
        print(f"iteration: {iteration} label_back_to_hebrew - more then 1 word! - reducing eng label...")
        eng_label = eng_label.split()[:-1]  # Remove the last word from the English label
        eng_label = " ".join(eng_label)
        label_back_to_hebrew = translate(eng_label, eng_to_heb_tokenizer, eng_to_heb_model)[0]
        iteration += 1  # Increment the iteration counter

    print(f"label_back_to_hebrew: {label_back_to_hebrew} 1 word!!")
    # Check if the loop ended because of max_iterations limit
    if iteration >= max_iterations:
        return False, f"Reached max iterations limit ({max_iterations}) without finding a match."

    # Continue with the original logic
    return check_using_input_and_label(eng_input, eng_label, heb_label)

def check_using_eng_label_only(eng_label, heb_label):
    """
    NOT USEFUL!
    Translates only the English label back to Hebrew and compares it with the Hebrew label.

    Returns a boolean indicating whether the translation matches and a message.
    """
    back_to_hebrew_eng_label = translate(eng_label, eng_to_heb_tokenizer, eng_to_heb_model)[0]

    if preprocess_text(back_to_hebrew_eng_label) == preprocess_text(heb_label):
        return True, "Match found using translation of the English label."

    return False, f"Mismatch found using translation of the English label: translated label is: {back_to_hebrew_eng_label}"

def check_using_input_and_label_and_first_heb_word(eng_input, eng_label, heb_label):
    """
    NOT USEFUL!
    Translates the English input and label back to Hebrew and checks if the first word of the translation
    matches the Hebrew label.

    Returns a boolean indicating whether the translation matches and a message.
    """
    # Split eng_label into words if it is a string
    eng_label_words = eng_label.split()

    # Check if the first label is a single letter
    if len(eng_label_words[0]) == 1:
        # If eng_label[0] is a single letter and there's a second word, use the second word
        if len(eng_label_words) > 1:
            eng_label = eng_label_words[1]
        else:
            eng_label = eng_label_words[0]  # If no second word, fallback to first
    else:
        eng_label = eng_label_words[0]  # If first word is not a single letter, use it

    return check_using_input_and_label(eng_input, eng_label, heb_label)


def split_into_sentences(text):
    """
    Splits text into sentences by new lines (one sentence per line).
    """
    return text.strip().splitlines()


def extract_first_five_words(sentence):
    """
    Extracts the first five words from a sentence.
    """
    words = sentence.split()
    return ' '.join(words[:5])


def preprocess_text(text):
    """
    Removes punctuation from the text for better comparison.
    """
    return text.translate(str.maketrans("", "", string.punctuation))

def process_hebrew_translation_dataset(input_file, output_csv_file, log_version, buffer_size=50):
    """
    Processes a Hebrew translation dataset and creates two CSV files:
    1. A file with all processed sentences.
    2. A file with sentences where `match_using_sentence == 1`.
    """
    match_csv_file = output_csv_file.replace(".csv", "_matches.csv")  # Derive matches CSV name
    sentences = read_sentences(input_file)
    buffer, match_buffer = [], []

    for idx, sentence in enumerate(sentences):
        result = process_sentence(sentence)

        if result:
            buffer.append(result)
            if result['match_using_sentence'] == 1:
                match_buffer.append(result)

        # Write to CSV in batches or at the last sentence
        if (idx + 1) % buffer_size == 0 or (idx + 1) == len(sentences):
            write_to_csv(buffer, output_csv_file, idx < buffer_size)
            write_to_csv(match_buffer, match_csv_file, idx < buffer_size)
            buffer.clear()
            match_buffer.clear()

def read_sentences(input_file):
    """Reads sentences from the input file."""
    with open(input_file, "r", encoding="utf-8") as infile:
        return split_into_sentences(infile.read())

def process_sentence(sentence):
    """
    Processes a single sentence to extract inputs, labels, and validate results.
    Returns None if the sentence is invalid.
    """
    words = preprocess_text(sentence).split()
    if len(words) < 2:  # Skip short sentences
        return None

    heb_input, heb_label = " ".join(words[:-1]), words[-1]
    eng_input, eng_label = translate_text(heb_input, heb_label)
    if not eng_input or not eng_label:
        return None

    match_sentence = validate_translation(eng_input, eng_label, heb_label)

    return {
        'eng_input': eng_input,
        'eng_label': eng_label,
        'match_using_sentence': 1 if match_sentence else 0,
        'heb_input': heb_input,
        'heb_label': heb_label
    }

def translate_text(heb_input, heb_label):
    """Translates Hebrew input and label to English."""
    eng_input = translate(heb_input, heb_to_eng_tokenizer, heb_to_eng_model, num_return_sequences=1)
    eng_label = translate(heb_label, heb_to_eng_tokenizer, heb_to_eng_model, num_return_sequences=1)
    return eng_input[0] if eng_input else None, eng_label[0] if eng_label else None

def validate_translation(eng_input, eng_label, heb_label):
    """Validates the translation using input and label."""
    return check_using_input_and_label(eng_input, eng_label, heb_label)[0]

def write_to_csv(buffer, csv_file, is_first_batch):
    """Writes buffered results to a CSV file."""
    mode = 'w' if is_first_batch else 'a'
    header = is_first_batch
    pd.DataFrame(buffer, columns=['eng_input', 'eng_label', 'match_using_sentence', 'heb_input', 'heb_label']).to_csv(
        csv_file, mode=mode, index=False, header=header, encoding='utf-8')


print("starting!!!!!!!!!!!!.")
process_hebrew_translation_dataset("../data_sets/ynet_train_256k.txt", "ynet_356k.csv", log_version="1", buffer_size=200)

print("Processing complete. Results written to CSV.")

