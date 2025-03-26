import re
from typing import Callable

import numpy as np
import pandas as pd
from snorkel.labeling import LabelingFunction, LFAnalysis, labeling_function

ABSTAIN = -1


def create_labeling_functions_from_regex(regex_patterns: list[tuple[re.Pattern, str, int, int]]) -> list[Callable]:
    """
    Create labeling functions from a list of regex patterns.

    Args:
        regex_patterns (list[tuple[re.Pattern, str, int, int]]): A list of tuples where each tuple contains:
            - pattern (re.Pattern): Compiled regex pattern to search for.
            - name (str): Name of the labeling function.
            - label (int): Label to return if the pattern matches.
            - label_else (int): Label to return if the pattern does not match.

    Returns:
        list[Callable]: A list of labeling functions.
    """
    labeling_functions = []
    for i, (pattern, name, match_label, else_label) in enumerate(regex_patterns):

        @labeling_function(name=f"lf_regex_{name}")
        def labeling_function_instance(x, pattern=pattern, match_label=match_label, else_label=else_label):
            # Return the match label if the pattern matches, otherwise return the else label
            return match_label if pattern.search(x.text) else else_label

        labeling_functions.append(labeling_function_instance)
    return labeling_functions


def int_to_alphabetic_string(n: int) -> str:
    """
    Convert an integer to a string using alphabetic characters (a, b, c, ..., z, aa, ab, ...).

    Args:
        n (int): The integer to convert.

    Returns:
        str: The corresponding alphabetic string.
    """
    result = []
    while n > 0:
        n -= 1
        # Calculate the current character and append it to the result list
        result.append(chr(n % 26 + ord("a")))
        # Move to the next character position
        n //= 26
    # Join the list into a string and return it
    return "".join(result[::-1])


def compute_conflict_matrices(label_matrix: np.ndarray, labeling_functions: list[LabelingFunction]) -> pd.DataFrame:
    """
    Compute and return conflict matrices for labeling functions, including normalized conflicts.

    Args:
        label_matrix (np.ndarray): The label matrix for the training set.
        labeling_functions (list[LabelingFunction]): A list of labeling functions.

    Returns:
        pd.DataFrame: A DataFrame containing both the conflict and normalized conflict matrices.
    """
    # Compute the normalized conflict matrix
    # normalize_by_overlaps=True normalizes the conflicts by the number of overlaps
    normalized_conflict_matrix = LFAnalysis(L=label_matrix, lfs=labeling_functions).lf_conflicts(normalize_by_overlaps=True)

    # Convert the normalized conflict matrix to a DataFrame
    # Transpose the DataFrame to have labeling function names as row labels
    normalized_conflict_df = pd.DataFrame([normalized_conflict_matrix], columns=[lf.name for lf in labeling_functions]).T

    # Rename the column to 'normalized_conflict' for clarity
    normalized_conflict_df.columns = ["normalized_conflict"]

    # Sort the DataFrame by the 'normalized_conflict' column in descending order
    normalized_conflict_df = normalized_conflict_df.sort_values(by="normalized_conflict", ascending=False)

    # Compute the conflict matrix without normalization
    conflict_matrix = LFAnalysis(L=label_matrix, lfs=labeling_functions).lf_conflicts()

    # Convert the conflict matrix to a DataFrame
    # Transpose the DataFrame to have labeling function names as row labels
    conflict_df = pd.DataFrame([conflict_matrix], columns=[lf.name for lf in labeling_functions]).T

    # Rename the column to 'conflict' for clarity
    conflict_df.columns = ["conflict"]

    # Sort the DataFrame by the 'conflict' column in descending order
    conflict_df = conflict_df.sort_values(by="conflict", ascending=False)

    # Concatenate the conflict and normalized conflict DataFrames along the columns
    final_conflict_matrix = pd.concat([conflict_df, normalized_conflict_df], axis=1)

    return final_conflict_matrix


def compute_pairwise_conflict_matrix(labeling_functions: list[LabelingFunction], label_matrix: np.ndarray) -> pd.DataFrame:
    """
    Compute the conflict matrix for a set of labeling functions and a set of labeled data.

    Args:
        labeling_functions (list[LabelingFunction]): A list of labeling functions.
        label_matrix (np.ndarray): A numpy array of shape (num_examples, num_lfs) containing the labels assigned by each labeling function to each example.

    Returns:
        pd.DataFrame: A DataFrame containing the conflict matrix.
    """
    # Initialize a matrix of zeros with shape (num_lfs, num_lfs)
    conflict_matrix = np.zeros((len(labeling_functions), len(labeling_functions)))

    # Compute the conflict between each pair of labeling functions
    for i in range(len(labeling_functions)):
        for j in range(len(labeling_functions)):
            # Calculate the conflict between labeling functions i and j
            # Conflict occurs when the labels are different and neither is ABSTAIN
            conflict_matrix[i, j] = ((label_matrix[:, i] != label_matrix[:, j]) & (label_matrix[:, i] != ABSTAIN) & (label_matrix[:, j] != ABSTAIN)).mean()

    # Convert the conflict matrix to a pandas DataFrame
    # Use labeling function names as row and column labels
    conflict_matrix_df = pd.DataFrame(conflict_matrix, columns=[lf.name for lf in labeling_functions], index=[lf.name for lf in labeling_functions])

    return conflict_matrix_df
