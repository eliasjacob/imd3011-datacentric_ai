import hashlib
import random
import re
from abc import ABC, abstractmethod
from typing import Any, Generator

import numpy as np
import pandas as pd
import skweak.utils
import spacy
from datasets import ClassLabel, Dataset, Features, Sequence, Value  # Import necessary classes from the datasets library
from skweak.base import SpanAnnotator
from spacy import displacy
from spacy.tokens import Doc, Span
from transformers import PreTrainedTokenizer


class BaseNERAnnotator(SpanAnnotator, ABC):
    """
    Abstract base class for NER annotators.

    Args:
        annotator_name (str): Name of the annotator.
        words_to_skip (list[str]): list of words to skip during annotation.
        label_key_name (str): Key name for the entity label in the output.
        merge_adjacent_entities (bool): Whether to merge adjacent entities with the same label.

    Attributes:
        words_to_skip (set): Set of lowercase words to skip during annotation.
        label_key_name (str): Key name for the entity label in the output.
        merge_adjacent_entities (bool): Flag to merge adjacent entities with the same label.
    """

    def __init__(
        self,
        annotator_name: str,
        label_key_name: str,
        words_to_skip: list[str] = None,
        merge_adjacent_entities: bool = True,
    ):
        # Initialize the parent class with the annotator name
        super().__init__(name=annotator_name)
        
        # Convert words to skip to a set of lowercase words for efficient lookup
        self.words_to_skip: set = set(word.lower() for word in (words_to_skip or []))
        
        # Store the key name for the entity label in the output
        self.label_key_name: str = label_key_name
        
        # Flag to determine whether to merge adjacent entities with the same label
        self.merge_adjacent_entities: bool = merge_adjacent_entities

    @abstractmethod
    def find_spans(
        self, doc: Doc
    ) -> Generator[tuple[int, int, str], None, None]:
        """
        Abstract method to find entity spans in a document.

        Args:
            doc (Doc): spaCy Doc object to annotate.

        Yields:
            tuple[int, int, str]: Start index, end index, and label of each entity span.
        """
        pass

    def _char_to_token_indices(
        self, doc: Doc, start_char: int, end_char: int
    ) -> tuple[int, int]:
        """
        Convert character indices to token indices.

        Args:
            doc (Doc): spaCy Doc object.
            start_char (int): Start character index.
            end_char (int): End character index.

        Returns:
            tuple[int, int]: Start and end token indices, or None if not found.
        """
        # Find the start token index corresponding to the start character index
        start_token = next(
            (
                token.i
                for token in doc
                if token.idx <= start_char < token.idx + len(token.text)
            ),
            None,
        )
        
        # Find the end token index corresponding to the end character index
        end_token = next(
            (
                token.i + 1
                for token in doc
                if token.idx <= end_char <= token.idx + len(token.text)
            ),
            None,
        )
        
        # Return the start and end token indices
        return start_token, end_token

    def _spans_overlap(
        self, span1: tuple[int, int, str, float], span2: tuple[int, int, str, float]
    ) -> bool:
        """
        Check if two spans overlap.

        Args:
            span1 (tuple[int, int, str, float]): First span (start, end, label, score).
            span2 (tuple[int, int, str, float]): Second span (start, end, label, score).

        Returns:
            bool: True if spans overlap, False otherwise.
        """
        # Check if the maximum start index is less than the minimum end index
        return max(span1[0], span2[0]) < min(span1[1], span2[1])

    def __call__(self, doc: Doc) -> Doc:
        """
        Apply the annotator to a document.

        Args:
            doc (Doc): spaCy Doc object to annotate.

        Returns:
            Doc: Annotated spaCy Doc object.
        """
        # Initialize an empty list of spans for the annotator in the document
        doc.spans[self.name] = []
        
        # Iterate over the spans found by the find_spans method
        for start, end, label in self.find_spans(doc):
            # Check if the span is allowed (not in the words_to_skip list)
            if self._is_allowed_span(doc, start, end):
                # Create a Span object and add it to the document's spans
                span = Span(doc, start, end, label=label)
                doc.spans[self.name].append(span)
        
        # Return the annotated document
        return doc

    
class SaltAnnotator(SpanAnnotator):
    def __init__(self, remove_salt: bool = True):
        super().__init__("lf_salt")
        self.salt_list = ['21-acetato', 'aceponato', 'acetato', 'acetilsalicilato', 'adipato', 'alendronato', 'alfaoxofenilpropionato', 'alginato', 'aminossalicilato', 'antimoniato', 'arginato', 'arsenito', 'ascorbato', 'aspartato', 'axetil', 'benzoato', 'besilato', 'betacipionato', 'bicarbonato', 'bissulfato', 'bitartarato', 'borato', 'brometo', 'bromidrato', 'butilbrometo', 'caproato', 'carbonato', 'carboxilato', 'ciclossilicato', 'cipionato', 'citrato', 'clatrato', 'clavulanato', 'clonixinato', 'cloranfenicol', 'cloreto', 'cloridrato', 'colistimetato', 'cromacato', 'cromato', 'cromoglicato', 'decanoato', 'di-hidrato', 'diaspartato', 'diatrizoato', 'dicloreto', 'dicloridrato', 'difosfato', 'diidrato', 'dimaleato', 'dimesilato', 'dinitrato', 'dinitrobenzoato', 'dipropionato', 'ditosilato', 'divalproato', 'dobesilato', 'docusato', 'embonato', 'enantato', 'esilato', 'estearato', 'estolato', 'etabonato', 'etanolato', 'etexilato', 'etilsuccinato', 'fempropionato', 'fendizoato', 'fenilpropionato', 'ferededato', 'ferrocianeto', 'fluoreto', 'folinato', 'fosfatidilcolina', 'fosfato', 'fosfito', 'fumarato', 'furoato', 'fusidato', 'gadobenato', 'gadopentetato', 'glicerofosfato', 'glicinato', 'glicirrizato', 'gliconato', 'gluceptato', 'gluconato', 'glutamato', 'hemi-hidrato', 'hemifumarato', 'hemisulfato', 'hemitartarato', 'hexafluoreto', 'hialuronato', 'hiclato', 'hidrobrometo', 'hidrocloreto', 'hidrogenotartarato', 'hidroxibenzoato', 'hidroxinaftoato', 'hipofosfito', 'ibandronato', 'iodeto', 'isetionato', 'isocaproato', 'lactato', 'lactobionato', 'laurato', 'laurilsulfato', 'levolisinato', 'levomalato', 'levulinato', 'lisetil', 'lisina', 'lisinato', 'malato', 'maleato', 'mepesuccinato', 'mesilato', 'metilbrometo', 'metilsulfato', 'metotrexato', 'micofenolato', 'molibdato', 'mono-hidrato', 'monofosfato', 'mononitrato', 'mucato', 'naftoato', 'nicotinato', 'nitrato', 'nitrito', 'nitroprusseto', 'oleato', 'orotato', 'oxalato', 'oxoglurato', 'palmitato', 'pamoato', 'pantotenato', 'pantotênico', 'permanganato', 'piconato', 'picossulfato', 'pidolato', 'pivalato', 'poliestirenossulfonato', 'polissulfato', 'propilenoglicolato', 'propionato', 'racealfa-hidroxigamametiltiobutanoato', 'racealfaoxobetametilbutanoato', 'racealfaoxobetametilpentanoato', 'racealfaoxogamametilpentanoato', 'resinato', 'sacarato', 'salicilato', 'selenato', 'selenito', 'silicato', 'subacetato', 'subgalato', 'succinato', 'sulfato', 'sulfeto', 'sulfito', 'sódico', 'tanato', 'tartarato', 'teoclato', 'tetra-hidrato', 'tiocianato', 'tosilato', 'triclofenato', 'trifenatato', 'undecanoato', 'undecilato', 'undecilenato', 'valerato', 'valproato', 'xinafoato', 'zirconato', 'zíncico']

        self.salt_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, self.salt_list)) + r')\b', re.IGNORECASE)
        self.remove_salt = remove_salt

    def find_spans(self, doc):
        for i in range(len(doc) - 2):
            if (self.salt_pattern.match(doc[i].text) and
                doc[i + 1].lower_ == 'de' and
                doc[i + 2].is_alpha):
                if self.remove_salt:
                    yield i+2, i+3, "MEDICAMENTO"
                else:
                    yield i, i+3, "MEDICAMENTO"

def render_entity_data_from_pipeline(
    text: str,
    pipeline_results: list[dict[str, Any]],
    colors: dict[str, str] = None,
    label_key_name: str = "entity_group",
) -> None:
    """
    Render entity data from pipeline results using spaCy's displacy.

    Args:
        text (str): Original text.
        pipeline_results (list[dict[str, Any]]): list of entity predictions from the pipeline.
        colors (dict[str, str]): Color mapping for entity labels.
        label_key_name (str): Key name for the entity label in the pipeline results.

    Returns:
        None
    """
    # Extract entity spans (start, end, label) from pipeline results
    entity_spans = [
        (result["start"], result["end"], result[label_key_name])
        for result in pipeline_results
    ]

    # If no colors are provided, generate random colors for each entity type
    if colors is None:
        # Get unique entity types
        entity_types = list(set([span[2] for span in entity_spans]))
        # Generate random colors for each entity type
        random_colors = get_random_colors(len(entity_types))
        # Map each entity type to a color
        colors = {
            entity_type: random_colors[i] for i, entity_type in enumerate(entity_types)
        }

    # Set displacy options with entity types and their corresponding colors
    displacy_options = {"ents": list(colors.keys()), "colors": colors}

    # Prepare data for displacy rendering
    displacy_data = [
        {
            "text": text,
            "ents": [
                {"start": span[0], "end": span[1], "label": span[2]}
                for span in entity_spans
            ],
        }
    ]

    # Render the entities using spaCy's displacy
    displacy.render(
        displacy_data, style="ent", manual=True, jupyter=True, options=displacy_options
    )

def get_random_colors(num_colors: int) -> list[str]:
    """
    Generate a list of random color hexadecimal codes.

    Args:
        num_colors (int): Number of colors to generate.

    Returns:
        list[str]: list of hexadecimal color codes.
    """
    # Generate a list of random hexadecimal color codes
    return [
        "#{:06x}".format(random.randint(0, 0xFFFFFF)) for _ in range(num_colors)
    ]

def merge_adjacent_entities(entities: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    """
    Merge adjacent entities with the same label.

    Args:
        entities (list[dict[str, Any]]): list of entity dictionaries.
        text (str): Original text.

    Returns:
        list[dict[str, Any]]: list of merged entity dictionaries.
    """
    # Return an empty list if there are no entities
    if not entities:
        return []

    # Initialize the list of merged entities and set the current entity to the first one
    merged_entities = []
    current_entity = entities[0]

    # Iterate over the remaining entities
    for next_entity in entities[1:]:
        # Check if the next entity is adjacent and has the same label as the current entity
        if next_entity["label"] == current_entity["label"] and (
            next_entity["start"] == current_entity["end"] + 1
            or next_entity["start"] == current_entity["end"]
        ):
            # Merge the current entity with the next entity
            current_entity["text"] = text[
                current_entity["start"] : next_entity["end"]
            ].strip()
            current_entity["end"] = next_entity["end"]
        else:
            # Add the current entity to the list of merged entities and update the current entity
            merged_entities.append(current_entity)
            current_entity = next_entity

    # Add the last current entity to the list of merged entities
    merged_entities.append(current_entity)

    return merged_entities

def extract_entities_in_gliner_format(spacy_doc: spacy.tokens.Doc, annotator_name: str, entities_to_remove: list[str]) -> list[dict[str, str]]:
    """
    Extract entities from a spaCy document in the format expected by the GLiNER model.

    Args:
        spacy_doc (spacy.tokens.Doc): The spaCy document containing the text and annotations.
        annotator_name (str): The name of the annotator whose annotations are to be used.
        entities_to_remove (list[str]): A list of entity labels to be removed from the text.

    Returns:
        list[dict[str, str]]: A list of dictionaries, each containing the cleaned entity text, its label, and its start and end positions.
    """
    # Initialize an empty list to store the cleaned entities
    cleaned_entities = []
    
    # Get the full text from the spaCy document
    full_text = spacy_doc.text

    # Create a regex pattern to match and remove specified entities
    regex_pattern = '|'.join(map(re.escape, entities_to_remove))
    regex_pattern = r'\b(?:' + regex_pattern + r')\b'

    # Iterate over the annotated spans in the spaCy document
    for span in spacy_doc.spans[annotator_name]:
        # Extract the text corresponding to the current span
        span_text = full_text[span.start_char:span.end_char]
        
        # Remove specified entities from the span text
        cleaned_text = re.sub(regex_pattern, '', span_text, flags=re.IGNORECASE).strip()

        # If the cleaned text is not empty, add it to the list of cleaned entities
        if cleaned_text:
            # Find all matches of the cleaned text within the original span text
            for match in re.finditer(re.escape(cleaned_text), span_text, flags=re.IGNORECASE):
                cleaned_entities.append({
                    "text": cleaned_text,
                    "label": span.label_,
                    "start": span.start_char + match.start(),
                    "end": span.start_char + match.end(),
                })

    return cleaned_entities

def convert_to_IOB(entity_spans: list[tuple[int, int, str, str]], input_text: str, tokenizer: PreTrainedTokenizer) -> list[tuple[str, str]]:
    """
    Converts entity spans to IOB format, with a fix for repeating sequences.

    Args:
        entity_spans (list[tuple[int, int, str, str]]): A list of entity spans. Each span is a tuple of (start, end, text, label).
        input_text (str): The input text.
        tokenizer (PreTrainedTokenizer): The tokenizer used to encode the input text.

    Returns:
        list[tuple[str, str]]: The list of tokens and their IOB tags.
    """

    tokenized_text = tokenizer(input_text, return_tensors="pt", is_split_into_words=False)
    tokens_int = tokenized_text["input_ids"].squeeze().tolist()
    # # Convert token IDs back to tokens
    # tokens_str = tokenizer.convert_ids_to_tokent(tokens_int)
    # Get word IDs for each token to handle subwords
    word_ids = tokenized_text.word_ids(batch_index=0)  # Assuming a single input for simplicity
    
    # Reconstruct words from token IDs, handling subwords
    tokens = reconstruct_sentence_from_token_ids(
        input_token_ids=tokens_int,
        associated_word_ids=word_ids,
        tokenizer=tokenizer
    )
    
    # Track the positions of the tokens to handle repeating sequences
    token_positions = []
    last_end_position = 0  # Tracks the end position of the last token to disambiguate repeating tokens
    for token in tokens:
        start_position = input_text.find(token, last_end_position)
        end_position = start_position + len(token)
        token_positions.append((start_position, end_position, token))
        last_end_position = end_position  # Update last_end_position to the end of the current token
    
    # Initialize the IOB-tagged output list with 'O' for each token
    iob_tags = ['O'] * len(tokens)
    
    # Process each entity span to assign IOB tags
    for span_start, span_end, matched_text, label in entity_spans:
        start_tagged = False
        for i, (start_position, end_position, token) in enumerate(token_positions):
            if start_position >= span_start and end_position <= span_end:
                if not start_tagged:
                    iob_tags[i] = f'B-{label}'
                    start_tagged = True
                else:
                    iob_tags[i] = f'I-{label}'
    
    # Combine tokens with their IOB tags
    iob_result = [(tokens[i], iob_tags[i]) for i in range(len(tokens))]
    
    return iob_result

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

def convert_from_iob_to_gliner_format(iob_tuples: list[tuple[str, str]], original_text: str) -> list[dict[str, str]]:
    """
    Convert IOB-tagged text to GLiNER format.

    Args:
        iob_tuples (list[tuple[str, str]]): A list of tuples where each tuple contains a word and its IOB tag.
        original_text (str): The original text from which the entities are extracted.

    Returns:
        list[dict[str, str]]: A list of dictionaries, each containing the text, label, start, and end positions of an entity.
    """
    # Initialize the result list and variables to track the current entity
    entities = []
    current_entity_words = []
    current_entity_label = None
    entity_start_index = None

    # Iterate over the IOB tuples
    for i, (word, tag) in enumerate(iob_tuples):
        if tag.startswith('B-'):
            # If there is an ongoing entity, finalize it
            if current_entity_words:
                entity_text = ' '.join(current_entity_words)
                entity_end_index = original_text.index(entity_text, entity_start_index) + len(entity_text)
                entities.append({
                    'text': entity_text,
                    'label': current_entity_label,
                    'start': entity_start_index,
                    'end': entity_end_index
                })
            # Start a new entity
            current_entity_words = [word]
            current_entity_label = tag[2:]  # Extract label after 'B-'
            entity_start_index = original_text.index(word, entity_start_index if entity_start_index is not None else 0)
        elif tag.startswith('I-') and current_entity_words and tag[2:] == current_entity_label:
            # Continue the current entity
            current_entity_words.append(word)
        elif tag == 'O':
            # Finalize the current entity if it exists
            if current_entity_words:
                entity_text = ' '.join(current_entity_words)
                entity_end_index = original_text.index(entity_text, entity_start_index) + len(entity_text)
                entities.append({
                    'text': entity_text,
                    'label': current_entity_label,
                    'start': entity_start_index,
                    'end': entity_end_index
                })
                current_entity_words = []
                current_entity_label = None
                entity_start_index = None

    # Handle case where the last word is part of an entity
    if current_entity_words:
        entity_text = ' '.join(current_entity_words)
        entity_end_index = original_text.index(entity_text, entity_start_index) + len(entity_text)
        entities.append({
            'text': entity_text,
            'label': current_entity_label,
            'start': entity_start_index,
            'end': entity_end_index
        })

    return merge_overlapping_named_entities(remove_duplicate_dicts(entities))

    # Example usage
    # iob_tuples = [('Some', 'O'), ('text', 'O'), ('with', 'O'), ('medicamento', 'B-MEDICAMENTO'), ('and', 'O'), ('other', 'O'), ('entities', 'O')]
    # original_text = "Some text with medicamento and other entities."
    # print(convert_from_iob_to_gliner_format(iob_tuples, original_text))

def remove_duplicate_dicts(dict_list: list[dict]) -> list[dict]:
    """
    Remove duplicate dictionaries from a list of dictionaries.

    Args:
        dict_list (list[dict]): A list of dictionaries.

    Returns:
        list[dict]: A list of dictionaries with duplicates removed.
    """
    # Convert each dictionary to a tuple of its items and use a set to remove duplicates
    seen = set()
    unique_dicts = []
    for d in dict_list:
        # Convert dictionary to a frozenset of its items
        items = frozenset(d.items())
        if items not in seen:
            seen.add(items)
            unique_dicts.append(d)
    return unique_dicts

def merge_overlapping_named_entities(entities: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Merge overlapping named entities in a list.

    Args:
        entities (list[dict[str, str]]): A list of dictionaries, each containing the text, label, start, and end positions of an entity.

    Returns:
        list[dict[str, str]]: A list of merged entities with no overlaps.
    """
    # Sort entities by their start index
    sorted_entities = sorted(entities, key=lambda entity: entity['start'])
    merged_entities = []

    for current_entity in sorted_entities:
        # If merged_entities is empty or current entity doesn't overlap with the last merged entity
        if not merged_entities or current_entity['start'] > merged_entities[-1]['end']:
            merged_entities.append(current_entity)
        else:
            # Overlapping found, update the last merged entity
            last_merged_entity = merged_entities[-1]
            # Update end index if the current entity extends further
            last_merged_entity['end'] = max(last_merged_entity['end'], current_entity['end'])
            # Update text if the current entity is longer
            if len(current_entity['text']) > len(last_merged_entity['text']):
                last_merged_entity['text'] = current_entity['text']

    return merged_entities


def remap_confidence_vectors(
    confidence_vectors: np.ndarray, 
    label_remapping: dict[int, int]
) -> np.ndarray:
    """
    Remaps the columns of a confidence vector according to a specified remapping.

    This function is useful when you need to aggregate or reassign the probabilities of certain classes
    to new class labels, based on a remapping dictionary. For example, if you have merged several classes
    into a single new class, you can use this function to sum the probabilities of the old classes to
    get the confidence of the new class.

    Args:
        confidence_vectors: A 2D numpy array where each row represents a sample and each column represents
                             the confidence of that sample belonging to a certain class.
        label_remapping: A dictionary where keys are the original class labels and values are the new class
                         labels. The function aggregates probabilities of old labels into the new labels
                         according to this mapping.

    Returns:
        A 2D numpy array of the same height as `confidence_vectors` but potentially different width,
        containing the remapped confidence vectors.

    Raises:
        ValueError: If `confidence_vectors` is not a 2D numpy array.
    """
    # Validate input dimensions
    if confidence_vectors.ndim != 2:
        raise ValueError("confidence_vectors must be a 2D numpy array.")

    # Determine the number of new labels after remapping
    n_new_labels = len(set(label_remapping.values()))
    # Initialize a new confidence matrix with zeros
    new_confidence_vectors = np.zeros((confidence_vectors.shape[0], n_new_labels), dtype=np.float32)

    # Aggregate probabilities for each old label into the new labels
    for old_label, new_label in label_remapping.items():
        new_confidence_vectors[:, new_label] += confidence_vectors[:, old_label]

    return new_confidence_vectors


def correct_label_issues_for_sentence(
    sentence_index: int, 
    all_issues: list[tuple[int, int]], 
    sentence_tokens_with_iob_labels: list[tuple[str, str]], 
    predicted_probabilities: list[list[float]], 
    id_to_label_map: dict[int, str]
) -> list[tuple[str, str]]:
    """
    Corrects the IOB labels for tokens in a sentence based on identified issues and prediction probabilities.

    Args:
        sentence_index: The index of the sentence being processed.
        all_issues: A list of tuples, where each tuple contains the sentence index and token index of an issue.
        sentence_tokens_with_iob_labels: A list of tuples, where each tuple contains a token and its corresponding IOB label.
        predicted_probabilities: A list of lists containing the prediction probabilities for each token in the sentence.
        id_to_label_map: A dictionary mapping label IDs to their corresponding IOB label strings.

    Returns:
        A list of tuples, where each tuple contains a token and its potentially corrected IOB label.
    """
    # Filter issues specific to the current sentence
    sentence_specific_issues = [issue for issue in all_issues if issue[0] == sentence_index]

    # If there are no issues in the sentence, return the original labels
    if not sentence_specific_issues:
        return sentence_tokens_with_iob_labels
    
    # Extract token indices for the issues in the current sentence
    issue_token_indices = [issue[1] for issue in sentence_specific_issues]

    corrected_labels = []

    # Iterate over each token and its label in the sentence
    for token_index, (token, original_label) in enumerate(sentence_tokens_with_iob_labels):
        # If the current token has an identified issue
        if token_index in issue_token_indices:
            # Get the prediction probabilities for the current token
            token_probs = predicted_probabilities[sentence_index][token_index]
            # Determine the predicted label based on the highest probability
            predicted_label = id_to_label_map[np.argmax(token_probs)]
            # Append the token and its corrected label to the output list
            corrected_labels.append((token, predicted_label))
        else:
            # If no issue, keep the original label
            corrected_labels.append((token, original_label))

    return corrected_labels

def add_iob_tags(tokens_with_labels):
    """
    Add IOB tags (B- and I-) to a list of tuples with tokens and labels.

    Args:
        tokens_with_labels (list[tuple[str, str]]): list of tuples with tokens and labels without IOB tags.

    Returns:
        list[tuple[str, str]]: list of tuples with tokens and labels with IOB tags.
    """
    
    tokens_with_labels = [(token, remove_iob_tag(label)) for token, label in tokens_with_labels]

    iob_tokens_with_labels = []
    prev_label = "O"

    for token, label in tokens_with_labels:
        if label == "O":
            iob_tokens_with_labels.append((token, label))
            prev_label = "O"
        else:
            if prev_label != label:
                iob_tokens_with_labels.append((token, "B-" + label))
            else:
                iob_tokens_with_labels.append((token, "I-" + label))
            prev_label = label

    return iob_tokens_with_labels

def remove_iob_tag(label):
    """
    Removes the 'B-' and 'I-' prefixes from an IOB label.

    Args:
        label: The IOB label to process.

    Returns:
        The label without the 'B-' or 'I-' prefix.
    """
    return label.replace('B-', '').replace('I-', '')


def calculate_md5(text: str) -> str:
    """
    Calculate the MD5 hash of a string.

    Args:
        text (str): The input string.

    Returns:
        str: The MD5 hash of the input string.
    """
    # Encode the input string as bytes
    encoded_text = text.encode('utf-8')
    # Calculate the MD5 hash of the encoded bytes
    md5_hash = hashlib.md5(encoded_text)
    # Return the hexadecimal representation of the hash
    return md5_hash.hexdigest()


def get_dataset_features(label_names: list[str]) -> Features:
    """
    Define the dataset features.

    Args:
        label_names (list[str]): list of label names.

    Returns:
        Features: The schema of the dataset.
    """
    return Features(
        {
            'tokens': Sequence(Value('string')),  # Sequence of tokens
            'ner_tags': Sequence(ClassLabel(names=label_names)),  # Sequence of named entity recognition tags
            'text': Value('string'),  # The original text
            'hash': Value('string')  # The MD5 hash of the text
        }
    )

def create_hmm_fixed_dicts(sentences_tokens_iob_fixed: list[list[tuple]], sentences: list[str]) -> dict[str, list]:
    """
    Create a dictionary with tokens, NER tags, text, and hash for the HMM-labeled training data.

    Args:
        sentences_tokens_iob_fixed (list[list[tuple]]): list of sentences with tokens and IOB tags.
        sentences (list[str]): list of original sentences.

    Returns:
        dict[str, list]: dictionary containing tokens, NER tags, text, and hash.
    """
    return {
        'tokens': [list(zip(*sentence))[0] for sentence in sentences_tokens_iob_fixed],
        'ner_tags': [list(zip(*sentence))[1] for sentence in sentences_tokens_iob_fixed],
        'text': sentences,
        'hash': [calculate_md5(sentence) for sentence in sentences]
    }

def create_hf_dataset(sentences_tokens_iob_fixed: list[list[tuple]], sentences: list[str], label_to_id: dict[str, int]) -> Dataset:
    """
    Create a Hugging Face Dataset from the HMM-labeled training data.

    Args:
        sentences_tokens_iob_fixed (list[list[tuple]]): list of sentences with tokens and IOB tags.
        sentences (list[str]): list of original sentences.
        label_to_id (dict[str, int]): Mapping from label names to label IDs.

    Returns:
        Dataset: The Hugging Face Dataset object.
    """
    
    # Create the reverse mapping from label IDs to label names
    # This is useful for converting label IDs back to label names
    id_to_label = {v: k for k, v in label_to_id.items()}

    # Extract the label names from the mapping
    # This list will be used to define the ClassLabel feature in the dataset
    label_names = list(label_to_id.keys())

    # Define the dataset features using the label names
    dataset_features = get_dataset_features(label_names)
    
    # Create the dictionary for the HMM-labeled training data
    train_hmm_fixed_dicts = create_hmm_fixed_dicts(sentences_tokens_iob_fixed, sentences)
    
    # Create and return the Hugging Face Dataset object
    return Dataset.from_dict(train_hmm_fixed_dicts, features=dataset_features)

class DataFrameAnnotator(BaseNERAnnotator):
	def __init__(
		self,
		annotator_name: str,
		data_frame: pd.DataFrame,
		column_weak_label: str,
		column_text: str,
		column_uid: str,
		label_key_name: str,
	):
		# Initialize the parent class with the annotator name, words to skip, label key name, and merge flag
		super().__init__(annotator_name=annotator_name, label_key_name=label_key_name)

		# Store the data frame
		self.data_frame = data_frame

		# Store the column names
		self.column_weak_label = column_weak_label
		self.column_text = column_text
		self.column_uid = column_uid

	def find_spans(self, doc: Doc) -> Generator[tuple[int, int, str], None, None]:
		"""
		Find entity spans in a document using the LangChain runnable sequence.

		Args:
		    doc (Doc): spaCy Doc object to annotate.

		Yields:
		    tuple[int, int, str]: Start index, end index, and label of each entity span.
		"""
		# Extract the text from the spaCy document
		text = doc.text
		row = self.data_frame[self.data_frame[self.column_text] == text]
		if len(row) == 0:
			return
		weak_labels = row[self.column_weak_label].iloc[0]

		predictions = weak_labels

		spans = []
		for pred in predictions:
			# Get the character indices for the start and end of the entity
			start_char, end_char = pred["start"], pred["end"]

			score = pred.get("score", 1.0)

			# Get the label of the entity
			label = pred[self.label_key_name]

			# Convert character indices to token indices
			start_token, end_token = self._char_to_token_indices(doc, start_char, end_char)

			# If valid token indices are found, add the span to the list
			if start_token is not None and end_token is not None:
				spans.append((start_token, end_token, label, score))

		# Sort spans by score (descending) and start position
		spans.sort(key=lambda x: (-x[3], x[0]))

		# Remove overlapping spans
		final_spans = []
		for span in spans:
			# Check if the current span overlaps with any existing span in final_spans
			if not any(self._spans_overlap(span, existing_span) for existing_span in final_spans):
				final_spans.append(span)

		# Yield the final spans, excluding the score
		for span in final_spans:
			yield span[:3]  # Yield only start, end, and label


# Added helper function for label validation reused below
def _validate_labels(labels: list[str], possible_labels: list[str]) -> list:
    if labels is not None:
        assert all(label in possible_labels for label in labels), f"All labels must be in {possible_labels}"
        return labels
    return [None]


def filter_issues_by_label_transition(
    sentence_index: int, 
    labels_for_all_sentences: list[list[str]], 
    words_for_all_sentences: list[list[str]],
    all_issues: list[tuple[int, int]], 
    predicted_probabilities: list[list[float]], 
    id_to_label_map: dict[int, str], 
    from_labels: list[str] = None,
    to_labels: list[str] = None,
    min_word_size: int = 1,
) -> list[tuple[int, int]]:
    """
    Filters issues within a specific sentence based on transitions from specified 'from_labels' to 'to_labels'.
    If 'from_labels' or 'to_labels' is None, it considers all issues without filtering by that specific label.

    Args:
        sentence_index (int): The index of the sentence to filter issues for.
        labels_for_all_sentences (list[list[str]]): A list of lists containing labels for all sentences.
        words_for_all_sentences (list[list[str]]): A list of lists containing words for all sentences
        all_issues (list[tuple[int, int]]): A list of tuples, each representing an issue with sentence index and token index.
        predicted_probabilities (list[list[float]]): A list of lists containing predicted probabilities for each label of tokens.
        id_to_label_map (dict[int, str]): A mapping from label IDs to label strings.
        from_labels (list[str]): The labels to filter from. If None, no filtering is applied based on from_labels.
        to_labels (list[str]): The labels to filter to. If None, no filtering is applied based on to_labels.
        min_word_size (int): The minimum number of characters in a word to consider it as an issue.

    Returns:
        list[tuple[int, int]]: A list of tuples, each containing the sentence index and token index of filtered issues.
    """
    # Validate label filters using helper function
    possible_labels = list(id_to_label_map.values()) + [None]
    from_labels = _validate_labels(from_labels, possible_labels)
    to_labels = _validate_labels(to_labels, possible_labels)

    # Filter issues specific to the current sentence
    sentence_specific_issues = [issue for issue in all_issues if issue[0] == sentence_index]
    # Extract token indices for the issues in the current sentence
    issue_token_indices = [issue[1] for issue in sentence_specific_issues]

    # Extract the actual labels for each token in the sentence
    actual_labels = labels_for_all_sentences[sentence_index]
    # Extract the predicted labels for each token in the sentence using the highest probability prediction
    predicted_labels = [id_to_label_map[np.argmax(probabilities)] for probabilities in predicted_probabilities[sentence_index]]

    filtered_issues = []
    # Iterate through each issue token index to filter based on label transitions
    for token_index in issue_token_indices:
        for from_label in from_labels: 
            if from_label is None or actual_labels[token_index] == from_label:
                for to_label in to_labels:
                    if to_label is None or predicted_labels[token_index] == to_label:
                        if len(words_for_all_sentences[sentence_index][token_index]) >= min_word_size:
                            filtered_issues.append((sentence_index, token_index))
                        break  # Stop checking to_labels once a match is found
                break  # Stop checking from_labels once a match is found

    return filtered_issues


def filter_all_issues_by_label_transition(
    labels_for_all_sentences: list[list[str]], 
    all_issues: list[tuple[int, int]], 
    words_for_all_sentences: list[list[str]],
    predicted_probabilities: list[list[float]], 
    id_to_label_map: dict[int, str], 
    from_labels: list[str] = None,
    to_labels: list[str] = None,
    min_word_size: int = 1,
) -> list[tuple[int, int]]:
    """
    Filters all issues across sentences based on specified label transitions from 'from_labels' to 'to_labels'.

    Args:
        labels_for_all_sentences (list[list[str]]): A list of lists containing labels for all sentences.
        all_issues (list[tuple[int, int]]): A list of tuples, each representing an issue with sentence index and token index.
        words_for_all_sentences (list[list[str]]): A list of lists containing words for all sentences.
        predicted_probabilities (list[list[float]]): A list of lists containing predicted probabilities for each label of tokens.
        id_to_label_map (dict[int, str]): A mapping from label IDs to label strings.
        from_labels (list[str]): The labels to filter from. If None, no filtering is applied based on from_labels.
        to_labels (list[str]): The labels to filter to. If None, no filtering is applied based on to_labels.
        min_word_size (int): The minimum number of characters in a word to consider it as an issue.

    Returns:
        list[tuple[int, int]]: A list of tuples, each containing the sentence index and token index of filtered issues.
    """
    # Validate label filters using helper function
    possible_labels = list(id_to_label_map.values()) + [None]
    from_labels = _validate_labels(from_labels, possible_labels)
    to_labels = _validate_labels(to_labels, possible_labels)

    # Identify unique sentences that have issues
    sentences_with_issues = list(set(issue[0] for issue in all_issues))
    
    print(f'There are {len(all_issues)} issues for {len(sentences_with_issues)} sentences')
    
    # Filter issues across all sentences
    filtered_issues = []
    for sentence_index in sentences_with_issues:
        filtered_issues.extend(filter_issues_by_label_transition(
            sentence_index=sentence_index,
            labels_for_all_sentences=labels_for_all_sentences,
            words_for_all_sentences=words_for_all_sentences,
            all_issues=all_issues,
            predicted_probabilities=predicted_probabilities,
            id_to_label_map=id_to_label_map,
            from_labels=from_labels,
            to_labels=to_labels,
            min_word_size=min_word_size
        ))
    
    # Identify unique sentences that have issues after filtering
    new_sentences_with_issues = list(set(issue[0] for issue in filtered_issues))
    
    print(f'After filtering, there are {len(filtered_issues)} issues for {len(new_sentences_with_issues)} sentences')
    return filtered_issues
