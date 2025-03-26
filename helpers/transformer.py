import gc
from copy import deepcopy
import evaluate
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import ClassLabel, Dataset, Features, Sequence, Value
from sklearn.model_selection import KFold
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, PreTrainedModel, PreTrainedTokenizer, Trainer, TrainingArguments


def reconstruct_sentence_from_token_ids(input_token_ids: list[int], associated_word_ids: list[int], tokenizer: PreTrainedTokenizer) -> list[str]:
    """
    Reconstructs a list of words from token IDs and associated word IDs, handling subwords appropriately.
    
    This function decodes token IDs to their corresponding tokens using a tokenizer. It then iterates through these tokens,
    aggregating subword tokens (prefixed with "##" in BERT-like tokenizers) into their full word forms. Special tokens
    (e.g., [CLS], [SEP] in BERT-like models) are ignored based on their associated word IDs being None.
    
    Args:
        input_token_ids (list[int]): A list of token IDs representing the encoded sentence.
        associated_word_ids (list[int]): A list of word IDs associated with each token. Subword tokens have the same word ID as their preceding tokens.
        tokenizer (PreTrainedTokenizer): The tokenizer used to decode token IDs back to tokens.
    
    Returns:
        list[str]: A list of reconstructed words from the token IDs.
    """
    
    # Decode the list of input token IDs back to their corresponding tokens
    tokens = tokenizer.convert_ids_to_tokens(input_token_ids)
    
    # Initialize an empty list to hold the reconstructed words
    reconstructed_words = []
    # Initialize an empty list to accumulate characters or subwords for the current word
    current_word_fragments = []

    # Iterate through each token and its associated word ID
    for token, word_id in zip(tokens, associated_word_ids):
        if word_id is None:
            # Skip special tokens which do not correspond to any word in the original sentence
            continue
        
        if token.startswith("##"):
            # If the token is a subword (part of a word), remove the "##" prefix and append it to the current word fragments
            current_word_fragments.append(token[2:])
        else:
            # If there's an ongoing word being built (from previous subwords), join its fragments and add to the reconstructed words list
            if current_word_fragments:
                reconstructed_words.append("".join(current_word_fragments))
                current_word_fragments = []  # Reset for the next word
            # Start accumulating fragments for the next word with the current token
            current_word_fragments.append(token)

    # After the loop, check if there's an unfinished word and add it to the reconstructed words list
    if current_word_fragments:
        reconstructed_words.append("".join(current_word_fragments))

    return reconstructed_words


def get_token_probabilities(input_text: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, aggregate_subwords: bool = True, device: str = 'cuda') -> dict[str, np.ndarray]:
    """
    Computes the predicted probabilities for each token in the input text using a given tokenizer and model.
    Optionally aggregates probabilities for subwords into their corresponding whole words.

    Args:
        input_text: The text to be tokenized and analyzed.
        tokenizer: The tokenizer to use for tokenizing the input text.
        model: The model to use for predicting token probabilities.
        aggregate_subwords: Whether to aggregate subword token probabilities into whole word probabilities. Defaults to True.
        device: The device to run the model on. Defaults to 'cuda'.

    Returns:
        A dictionary with two keys:
        - 'tokens'/'words': A list of tokens or aggregated words from the input text.
        - 'probs': A numpy array of the corresponding probability distributions for each token or aggregated word.
    """

    # Tokenize the input text and prepare inputs for the model
    model_inputs = tokenizer(input_text, return_tensors="pt", is_split_into_words=False)
    # Move model inputs to the specified device (e.g., GPU)
    model_inputs_on_device = {key: value.to(device) for key, value in deepcopy(model_inputs).items()}
    model = model.to(device)

    # Perform inference with the model
    model_outputs = model(**model_inputs_on_device)
    logits = model_outputs.logits
    # Convert logits to probabilities
    probabilities = F.softmax(logits, dim=-1).detach().cpu().numpy()[0]

    # Convert token IDs back to tokens
    tokens = tokenizer.convert_ids_to_tokens(model_inputs["input_ids"].squeeze().tolist())
    # Get word IDs for each token to handle subwords
    word_ids = model_inputs.word_ids(batch_index=0)  # Assuming a single input for simplicity

    if not aggregate_subwords:
        # If not aggregating subwords, return tokens and their probabilities directly
        return {
            'tokens': tokens,
            'probs': probabilities
        }

    # Aggregate probabilities for subwords
    aggregated_probabilities = {}
    for token, prob, word_id in zip(tokens, probabilities, word_ids):
        if word_id is not None:  # Ignore special tokens
            if word_id in aggregated_probabilities:
                aggregated_probabilities[word_id].append(prob)
            else:
                aggregated_probabilities[word_id] = [prob]

    # Calculate mean probability for aggregated subwords
    aggregated_probabilities = {word_id: np.mean(probs, axis=0) for word_id, probs in aggregated_probabilities.items()}
    
    # Extract unique word IDs, ignoring None values for special tokens
    unique_word_ids = sorted({word_id for word_id in word_ids if word_id is not None})

    # Reconstruct words from token IDs, handling subwords
    words = reconstruct_sentence_from_token_ids(
        input_token_ids=model_inputs["input_ids"].squeeze().tolist(),
        associated_word_ids=word_ids,
        tokenizer=tokenizer
    )
    # Pair reconstructed words with their aggregated probabilities
    word_probabilities = np.stack([aggregated_probabilities[word_id] for word_id in unique_word_ids])

    # Ensure the number of words matches the number of probability distributions
    assert len(words) == word_probabilities.shape[0], f"Word count and probability count mismatch: {len(words)} words vs {word_probabilities.shape[0]} probs"

    return {
        'words': words,
        'probs': word_probabilities
    }


def get_training_arguments(output_dir: str = './tmp/ner', learning_rate: float = 2e-5, per_device_train_batch_size: int = 32, per_device_eval_batch_size: int = 24, num_train_epochs: int = 4, weight_decay: float = 0.01, eval_accumulation_steps: int = 2, seed: int = 271828, bf16:bool = True, fp16=False, gradient_accumulation_steps:int = 1) -> TrainingArguments:
    """
    Generates the TrainingArguments object for training a model with the specified hyperparameters.
    
    Args:
        output_dir (str): The output directory for saving model checkpoints and logs. Defaults to './tmp/ner'.
        learning_rate (float): The learning rate for training. Defaults to 2e-5.
        per_device_train_batch_size (int): The batch size for training. Defaults to 32.
        per_device_eval_batch_size (int): The batch size for evaluation. Defaults to 24.
        num_train_epochs (int): The number of training epochs. Defaults to 4.
        weight_decay (float): The weight decay for regularization. Defaults to 0.01.
        eval_accumulation_steps (int): The number of evaluation steps to accumulate before performing an update. Defaults to 2.
        seed (int): The random seed for reproducibility. Defaults to 271828.
        bf16 (bool): Whether to use bfloat16 precision for training. Defaults to True.
        fp16 (bool): Whether to use mixed precision training. Defaults to False.
        gradient_accumulation_steps (int): The number of steps to accumulate gradients before performing an update. Defaults to 1.

    
    Returns:
        TrainingArguments: The TrainingArguments object with the specified hyperparameters.
    """
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        save_total_limit=1,
        logging_steps=1,
        eval_steps=1,
        save_steps=1,
        evaluation_strategy="epoch",
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        do_train=True,
        do_eval=True,
        logging_strategy="steps",
        bf16=bf16,
        fp16=fp16,
        eval_accumulation_steps=eval_accumulation_steps,
        push_to_hub=False,
        seed=seed,
    )
    
    return training_arguments


def train_named_entity_recognition_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, training_args: TrainingArguments, train_dataset: Dataset, valid_dataset: Dataset, test_dataset: Dataset, id_to_label: dict[int, str]) -> PreTrainedModel:
    """
    Trains a named entity recognition model using the provided datasets, model, and tokenizer.

    Args:
        model: The model to be trained.
        tokenizer: The tokenizer to be used for tokenizing the input texts.
        training_args: The training arguments specifying training parameters.
        train_dataset: The dataset to be used for training.
        valid_dataset: The dataset to be used for validation.
        test_dataset: The dataset to be used for testing.
        id_to_label: A dictionary mapping label IDs to their corresponding string labels.


    Returns:
        The trained model.
    """
    # Initialize the data collator for token classification tasks
    token_classification_collator = DataCollatorForTokenClassification(tokenizer)
    label_all_tokens = True
    test_dataset = test_dataset if test_dataset is not None else valid_dataset

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    
    def compute_metrics_for_evaluation(predictions_and_labels: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        """
        Computes metrics for model evaluation.

        Args:
            predictions_and_labels: A tuple containing the model predictions and the true labels.

        Returns:
            A dictionary containing precision, recall, f1, and accuracy metrics.
        """
        predictions, labels = predictions_and_labels
        # Convert logits to actual predictions
        predictions = np.argmax(predictions, axis=2)

        # Filter out the special tokens and convert IDs to labels
        true_predictions = [
            [id_to_label[pred] for (pred, label) in zip(prediction, label) if label != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id_to_label[label] for (pred, label) in zip(prediction, label) if label != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # Load the seqeval metric for named entity recognition
        metric = evaluate.load("seqeval")
        results = metric.compute(predictions=true_predictions, references=true_labels)

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def tokenize_and_align_labels(examples: dict[str, list[str]]) -> dict[str, list[str]]:
        """
        Tokenizes the input words and aligns the labels with the tokens.

        Args:
            examples: A dictionary containing the input words and the corresponding labels.

        Returns:
            A dictionary containing the tokenized input words and the aligned labels.
        """
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=512)

        aligned_labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            aligned_labels.append(label_ids)

        tokenized_inputs["labels"] = aligned_labels
        return tokenized_inputs

    # Apply the tokenization and label alignment to the training and validation datasets
    tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_valid_dataset = valid_dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        data_collator=token_classification_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_for_evaluation,
    )

    # Start training
    trainer.train()
    model.eval()

    # Evaluate the model on the test set
    test_results = trainer.evaluate(tokenized_test_dataset)
    
    return model, test_results


def get_batch_token_probabilities(texts: list[str], tokenizer: PreTrainedTokenizer, model: PreTrainedModel, aggregate_subwords: bool = True, batch_size: int = 32, device: str = 'cuda') -> list[dict[str, list[str] | np.ndarray]]:
    """
    Computes the predicted probabilities for each token in a batch of texts using a given tokenizer and model.
    Optionally aggregates probabilities for subwords into their corresponding whole words for each text in the batch.
    The output is a list of dictionaries, each corresponding to a text in the batch, with keys 'words' for the list of words or tokens,
    and 'probs' for the numpy array of probability distributions for each word or token.

    Args:
        texts: A list of texts to be tokenized and analyzed.
        tokenizer: The tokenizer to use for tokenizing the input texts.
        model: The model to use for predicting token probabilities.
        aggregate_subwords: Whether to aggregate subword token probabilities into whole word probabilities. Defaults to True.
        batch_size: The size of each batch for processing. Defaults to 32.
        device: The device to run the model on. Defaults to 'cuda'.

    Returns:
        A list of dictionaries, each with keys 'words' and 'probs', for each text in the batch.
    """
    batch_results = []  # Initialize the list to store results for each batch

    # Process texts in batches
    for start_idx in range(0, len(texts), batch_size):
        batch_texts = texts[start_idx:start_idx + batch_size]  # Extract texts for the current batch
        # Tokenize the batch of texts and prepare model inputs
        model_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, is_split_into_words=False)
        # Move model inputs to the specified device
        model_inputs_on_device = {k: v.to(device) for k, v in model_inputs.items()}
        model = model.to(device)  # Ensure the model is on the correct device

        # Perform inference with the model
        model_outputs = model(**model_inputs_on_device)
        logits = model_outputs.logits  # Extract logits from model outputs
        probabilities = F.softmax(logits, dim=-1).detach().cpu().numpy()  # Convert logits to probabilities

        # Process each text in the current batch
        for i, input_text in enumerate(batch_texts):
            tokens = tokenizer.convert_ids_to_tokens(model_inputs["input_ids"][i].squeeze().tolist())  # Convert token IDs back to tokens
            word_ids = model_inputs.word_ids(batch_index=i)  # Get word IDs for handling subwords

            if not aggregate_subwords:
                # If not aggregating subwords, store tokens and their probabilities directly
                batch_results.append({
                    'tokens': tokens,
                    'probs': probabilities[i]
                })
                continue

            # Aggregate probabilities for subwords
            word_probabilities = {}
            for token, prob, word_id in zip(tokens, probabilities[i], word_ids):
                if word_id is not None:  # Ignore special tokens
                    if word_id in word_probabilities:
                        word_probabilities[word_id].append(prob)
                    else:
                        word_probabilities[word_id] = [prob]

            # Calculate mean probability for aggregated subwords
            aggregated_probabilities = {word_id: np.mean(probs, axis=0) for word_id, probs in word_probabilities.items()}
            
            # Extract unique word IDs, ignoring None values for special tokens
            unique_word_ids = sorted(set(word_id for word_id in word_ids if word_id is not None))

            # Reconstruct words from token IDs, handling subwords
            words = reconstruct_sentence_from_token_ids(
                input_token_ids=model_inputs["input_ids"][i].squeeze().tolist(),
                associated_word_ids=word_ids,
                tokenizer=tokenizer
            )

            # Store the reconstructed words and their aggregated probabilities
            batch_results.append({
                'words': words,
                'probs': np.stack([aggregated_probabilities[word_id] for word_id in unique_word_ids])
            })

    return batch_results

def cross_val_predict_ner(
    hf_dataset: Dataset,
    model_checkpoint_path: str, 
    training_args, 
    id_to_label: dict[int, str],
    random_state: int = 271828, 
    n_splits_kfold: int = 5,
    device: str = 'cuda'
) -> list[dict]:
    """
    Performs cross-validation prediction for Named Entity Recognition (NER) using a specified model and dataset.

    Args:
        hf_dataset: The Hugging Face dataset containing the tokenized texts and IOB labels.
        model_checkpoint_path: The path to the pretrained model checkpoint.
        training_args: The training arguments specifying training parameters.
        id_to_label: A dictionary mapping label IDs to their corresponding string labels.
        random_state: The random state for KFold shuffling.
        n_splits_kfold: The number of splits for KFold.
        device: The device to run the model on ('cuda' or 'cpu').

    Returns:
        A list of dictionaries, each containing the UID, text, predicted words, and predicted probabilities for each row in the validation set.
    """
    # Ensure all label IDs are integers
    assert all(isinstance(k, int) for k in id_to_label.keys()), "Label IDs must be integers."

    # Initialize KFold
    kfold = KFold(n_splits=n_splits_kfold, shuffle=True, random_state=random_state)
    
    # Reverse mapping from label names to IDs
    label_to_id = {v: k for k, v in id_to_label.items()}
    num_labels = len(id_to_label)
    
    print(f'Label to ID mapping: {label_to_id}')
    print(f'ID to label mapping: {id_to_label}')
    print(f'Number of labels: {num_labels}')

    dataframe = hf_dataset.to_pandas()

    predicted_dataset = []

    for fold, (train_idx, val_idx) in enumerate(tqdm(kfold.split(dataframe), desc="Folds", total=n_splits_kfold)):
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()

        train_dataset = hf_dataset.select(train_idx)
        valid_dataset = hf_dataset.select(val_idx)

        # Load the pretrained model and tokenizer
        pretrained_language_model = AutoModelForTokenClassification.from_pretrained(model_checkpoint_path, num_labels=num_labels, id2label=id_to_label, label2id=label_to_id)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_path, model_max_length=512)

        # Train the NER model
        trained_ner_model, test_results = train_named_entity_recognition_model(
            model=pretrained_language_model,
            tokenizer=tokenizer,
            training_args=training_args,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=None,
            id_to_label=id_to_label,
        )

        # Clear memory
        pretrained_language_model = None
        gc.collect()
        torch.cuda.empty_cache()
        
        # Predict probabilities for the validation set
        predicted_probs = get_batch_token_probabilities(
            texts=[i['text'] for i in valid_dataset],
            tokenizer=tokenizer, 
            model=trained_ner_model, 
            aggregate_subwords=True, 
            batch_size=8, 
            device=device
        )

        # Append predictions to the dataset
        for item, probs in zip(valid_dataset, predicted_probs):
            predicted_dataset.append(
                {
                    'hash': item.get('hash', None),
                    'text': item.get('text', None),
                    "predicted_words": probs['words'],
                    "predicted_probs": probs['probs'].tolist()
                }
            )

    # Sort the predicted dataset based on UIDs
    uids = dataframe['hash'].values
    uid_to_index = {uid: index for index, uid in enumerate(uids)}
    sorted_data = sorted(predicted_dataset, key=lambda x: uid_to_index[x['hash']])

    return sorted_data