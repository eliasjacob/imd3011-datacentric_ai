from abc import ABC, abstractmethod
from typing import Any

import numpy as np

np.bool = np.bool_

from flyingsquid.label_model import LabelModel as FSLabelModel
from numba import njit, prange
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from tqdm import trange

from helpers.generative_model_src import SrcGenerativeModel

# The code in this file is adapted from the Wrench paper - https://github.com/JieyuZ2/wrench

@njit(parallel=True, nogil=True)
def update_posterior_probabilities(posterior_probs: np.ndarray, error_rates: np.ndarray, augmented_labels: np.ndarray) -> None:
    """Update posterior probabilities using error rates and augmented label matrix.

    Args:
        posterior_probs (np.ndarray): Posterior probabilities matrix.
        error_rates (np.ndarray): Error rates tensor.
        augmented_labels (np.ndarray): Augmented label matrix.
    """
    num_samples, num_classes = posterior_probs.shape
    for i in prange(num_samples):
        for j in range(num_classes):
            posterior_probs[i, j] *= np.prod(np.power(error_rates[:, j, :], augmented_labels[i, :, :]))


@njit(parallel=True, nogil=True)
def initialize_posterior_probabilities(posterior_probs: np.ndarray, label_matrix: np.ndarray, num_classes: int, ABSTAIN: int = -1) -> None:
    """Initialize posterior probabilities based on label matrix.

    Args:
        posterior_probs (np.ndarray): Posterior probabilities matrix to be initialized.
        label_matrix (np.ndarray): Label matrix.
        num_classes (int): Number of classes.
        ABSTAIN (int, optional): Value representing abstain. Defaults to -1.
    """
    num_samples, num_labelers = label_matrix.shape
    for i in prange(num_samples):
        class_counts = np.zeros(num_classes)
        for j in range(num_labelers):
            if label_matrix[i, j] != ABSTAIN:
                class_counts[label_matrix[i, j]] += 1
        if class_counts.sum() == 0:
            class_counts += 1  # Avoid division by zero
        posterior_probs[i] = class_counts


class BaseLabelModel(ABC):
    """Abstract base class for label models."""

    @abstractmethod
    def __init__(self):
        pass

    def fit(self, label_matrix: np.ndarray) -> None:
        """Fit the label model to the label matrix.

        Args:
            label_matrix (np.ndarray): Label matrix.
        """
        print(f"{self.__class__.__name__} does not have a fit method")

    @abstractmethod
    def predict_proba(self, label_matrix: np.ndarray) -> np.ndarray:
        """Predict class probabilities for the given label matrix.

        Args:
            label_matrix (np.ndarray): Label matrix.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        pass

    def score(self, label_matrix_valid: np.ndarray, true_labels: np.ndarray, metrics: list[str] = None, **kwargs) -> list[float] | list[np.ndarray]:
        """Evaluate the model using specified metrics.

        Args:
            label_matrix_valid (np.ndarray): Validation label matrix.
            true_labels (np.ndarray): True labels for validation data.
            metrics (list[str], optional): list of metrics to evaluate. Defaults to None.

        Returns:
            Union[list[float], list[np.ndarray]]: Evaluation scores for the specified metrics.
        """
        acceptable_scores = {
            'accuracy': accuracy_score,
            'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
            'mcc': matthews_corrcoef,
            'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro'),
            'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro'),
            'roc_auc': lambda y_true, y_pred: roc_auc_score(y_true, y_pred, average='macro'),
            'confusion_matrix': confusion_matrix
        }
        if metrics is None:
            metrics = ['accuracy', 'mcc']
        assert all([metric in acceptable_scores for metric in metrics]), f"metrics must be a subset of {list(acceptable_scores.keys())}"

        predicted_labels = self.predict(label_matrix_valid, **kwargs)
        scores = [acceptable_scores[metric](true_labels, predicted_labels) for metric in metrics]
        return {metric: score for metric, score in zip(metrics, scores)}

    def predict(self, label_matrix: np.ndarray, **kwargs) -> np.ndarray:
        """Predict class labels for the given label matrix.

        Args:
            label_matrix (np.ndarray): Label matrix.

        Returns:
            np.ndarray: Predicted class labels.
        """
        predicted_probabilities = self.predict_proba(label_matrix, **kwargs)
        return np.argmax(predicted_probabilities, axis=1)
    
    def __repr__(self) -> str:
        """String representation of the label model.

        Returns:
            str: String representation of the label model.
        """
        if hasattr(self, 'model'):
            return f"{self.__class__.__name__} - ({self.model})"
        else:
            return f"{self.__class__.__name__} - No model"


class DawidSkene(BaseLabelModel):
    """Dawid-Skene model for label aggregation."""

    def __init__(self, cardinality: int):
        """Initialize the Dawid-Skene model.

        Args:
            cardinality (int): Number of classes.
        """
        super().__init__()
        self.has_fit_method = True
        self.cardinality = cardinality

    def fit(self, label_matrix: np.ndarray, num_epochs: int = 10000, tolerance: float = 1e-5) -> None:
        """Fit the Dawid-Skene model to the label matrix.

        Args:
            label_matrix (np.ndarray): Label matrix.
            num_epochs (int, optional): Number of epochs for training. Defaults to 10000.
            tolerance (float, optional): Convergence tolerance. Defaults to 1e-5.
        """
        posterior_probs = self._initialize_posterior_probabilities(label_matrix)
        augmented_labels = self._initialize_augmented_labels(label_matrix)

        max_iter = num_epochs
        tol = tolerance
        old_class_marginals = None
        old_error_rates = None
        for iter in trange(max_iter):

            # M-step
            class_marginals, error_rates = self._m_step(augmented_labels, posterior_probs)

            # E-step
            posterior_probs = self._e_step(augmented_labels, class_marginals, error_rates)

            # Check for convergence
            if old_class_marginals is not None:
                class_marginals_diff = np.sum(np.abs(class_marginals - old_class_marginals))
                error_rates_diff = np.sum(np.abs(error_rates - old_error_rates))
                if class_marginals_diff < tol and error_rates_diff < tol:
                    break

            # Update current values
            old_class_marginals = class_marginals
            old_error_rates = error_rates

        self.error_rates = error_rates
        self.class_marginals = class_marginals

    def _initialize_posterior_probabilities(self, label_matrix: np.ndarray) -> np.ndarray:
        """Initialize posterior probabilities based on the label matrix.

        Args:
            label_matrix (np.ndarray): Label matrix.

        Returns:
            np.ndarray: Initialized posterior probabilities.
        """
        num_classes = self.cardinality
        num_samples, num_labelers = label_matrix.shape

        posterior_probs = np.zeros((num_samples, num_classes))
        initialize_posterior_probabilities(posterior_probs, label_matrix, num_classes)

        posterior_probs /= posterior_probs.sum(axis=1, keepdims=True)
        return posterior_probs

    def _initialize_augmented_labels(self, label_matrix: np.ndarray) -> np.ndarray:
        """Initialize augmented labels based on the label matrix.

        Args:
            label_matrix (np.ndarray): Label matrix.

        Returns:
            np.ndarray: Augmented label matrix.
        """
        label_matrix_offset = label_matrix + 1
        augmented_labels = (np.arange(self.cardinality + 1) == label_matrix_offset[..., None]).astype(int)
        return augmented_labels

    def _m_step(self, augmented_labels: np.ndarray, posterior_probs: np.ndarray) -> tuple:
        """Perform the M-step of the EM algorithm.

        Args:
            augmented_labels (np.ndarray): Augmented label matrix.
            posterior_probs (np.ndarray): Posterior probabilities.

        Returns:
            tuple: Class marginals and error rates.
        """
        num_samples, num_labelers, _ = augmented_labels.shape
        class_marginals = np.sum(posterior_probs, 0) / num_samples

        # Compute error rates
        error_rates = np.tensordot(posterior_probs, augmented_labels, axes=[[0], [0]]).transpose(1, 0, 2)

        # Normalize by summing over all observation classes
        sum_error_rates = np.sum(error_rates, axis=-1, keepdims=True)
        error_rates = np.divide(error_rates, sum_error_rates, where=sum_error_rates != 0)
        return class_marginals, error_rates

    def _e_step(self, augmented_labels: np.ndarray, class_marginals: np.ndarray, error_rates: np.ndarray) -> np.ndarray:
        """Perform the E-step of the EM algorithm.

        Args:
            augmented_labels (np.ndarray): Augmented label matrix.
            class_marginals (np.ndarray): Class marginals.
            error_rates (np.ndarray): Error rates.

        Returns:
            np.ndarray: Updated posterior probabilities.
        """
        num_samples, num_labelers, _ = augmented_labels.shape
        num_classes = self.cardinality

        posterior_probs = np.ones([num_samples, num_classes]) * class_marginals
        update_posterior_probabilities(posterior_probs, error_rates, augmented_labels)

        # Normalize posterior probabilities by dividing by the sum over all classes
        sum_posterior_probs = np.sum(posterior_probs, axis=-1, keepdims=True)
        posterior_probs = np.divide(posterior_probs, sum_posterior_probs, where=sum_posterior_probs != 0)
        return posterior_probs

    def _calc_likelihood(self, augmented_labels: np.ndarray, class_marginals: np.ndarray, error_rates: np.ndarray) -> float:
        """Calculate the log-likelihood of the model.

        Args:
            augmented_labels (np.ndarray): Augmented label matrix.
            class_marginals (np.ndarray): Class marginals.
            error_rates (np.ndarray): Error rates.

        Returns:
            float: Log-likelihood of the model.
        """
        num_samples, num_labelers, _ = augmented_labels.shape
        num_classes = self.cardinality
        log_likelihood = 0.0

        for i in range(num_samples):
            single_likelihood = 0.0
            for j in range(num_classes):
                class_prior = class_marginals[j]
                posterior_likelihood = np.prod(np.power(error_rates[:, j, :], augmented_labels[i, :, :]))
                posterior_prob = class_prior * posterior_likelihood
                single_likelihood += posterior_prob

            temp = log_likelihood + np.log(single_likelihood)

            if np.isnan(temp) or np.isinf(temp):
                raise ValueError('log likelihood is not a number')
            
            log_likelihood = temp

        return log_likelihood

    def predict_proba(self, label_matrix: np.ndarray) -> np.ndarray:
        """Predict class probabilities for the given label matrix.

        Args:
            label_matrix (np.ndarray): Label matrix.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        augmented_labels = self._initialize_augmented_labels(label_matrix)
        predicted_probs = self._e_step(augmented_labels, self.class_marginals, self.error_rates)
        return predicted_probs
    

class GenerativeModel(BaseLabelModel):
    def __init__(self, 
                 cardinality: int, 
                 lr: float = 1e-4,
                 l2: float = 1e-1,
                 n_epochs: np.int64 = 100,
                seed: int = 271828,
                 **kwargs: Any):
        super().__init__()
        self.has_fit_method = True
        self.cardinality = cardinality
        self.model = None
        self.lr = lr
        self.l2 = l2
        self.n_epochs = n_epochs
        self.seed = seed


    def fit(self,
            L: np.ndarray,
            class_balance: np.ndarray = None,
            threads: np.int64 = 10,
            verbose: bool = False,
            **kwargs: Any):

        if class_balance is None:  # All classes have equal weight
            class_balance = np.ones(self.cardinality ) / self.cardinality 

        L = self.process_label_matrix(L)

        ## TODO support multiclass class prior
        log_y_prior = np.log(class_balance)
        label_model = SrcGenerativeModel(cardinality=self.cardinality, class_prior=False, seed=self.seed)
        label_model.train(
            L=L,
            init_class_prior=log_y_prior,
            epochs=self.n_epochs,
            step_size=self.lr,
            reg_param=self.l2,
            verbose=verbose,
            cardinality=self.cardinality,
            threads=threads)

        self.model = label_model

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        L = self.process_label_matrix(L)
        return self.model.predict_proba(L)
    
    def process_label_matrix(self, L_):
        L = L_.copy()
        if self.cardinality > 2:
            L += 1
        else:
            abstain_mask = L == -1
            negative_mask = L == 0
            L[abstain_mask] = 0
            L[negative_mask] = -1
        return L
    
    def unprocess_label_matrix(self, L_):
        L = L_.copy()
        if self.cardinality > 2:
            L -= 1
        else:
            abstain_mask = L == -1
            negative_mask = L == 0
            L[abstain_mask] = -1
            L[negative_mask] = 0
        return L

class FlyingSquid(BaseLabelModel):
    """FlyingSquid model for label aggregation."""

    def __init__(self, cardinality: int, **kwargs):
        """Initialize the FlyingSquid model.

        Args:
            cardinality (int): Number of classes.
        """
        super().__init__()
        self.has_fit_method = True
        self.cardinality = cardinality
        self.model = None

    def fit(self, label_matrix: np.ndarray, class_balance: np.ndarray = None, ABSTAIN: int = -1, dependency_graph: list = None, verbose: bool = False, **kwargs) -> None:
        """Fit the FlyingSquid model to the label matrix.

        Args:
            label_matrix (np.ndarray): Label matrix.
            class_balance (np.ndarray, optional): Class balance. Defaults to None.
            ABSTAIN (int, optional): Value representing abstain. Defaults to -1.
            dependency_graph (list, optional): Dependency graph. Defaults to [].
            verbose (bool, optional): Verbose output. Defaults to False.
        """
        if class_balance is None:  # All classes have equal weight
            class_balance = np.ones(self.cardinality) / self.cardinality 
        
        if dependency_graph is None:
            dependency_graph = []
            
        num_samples, num_labelers = label_matrix.shape

        if self.cardinality > 2:
            model = []
            for i in range(self.cardinality):
                label_model = FSLabelModel(m=num_labelers, lambda_edges=dependency_graph)
                label_matrix_i = np.copy(label_matrix)
                target_mask = label_matrix_i == i
                abstain_mask = label_matrix_i == ABSTAIN
                other_mask = (~target_mask) & (~abstain_mask)
                label_matrix_i[target_mask] = 1
                label_matrix_i[abstain_mask] = 0
                label_matrix_i[other_mask] = -1
                label_model.fit(L_train=label_matrix_i, class_balance=np.array([1 - class_balance[i], class_balance[i]]), verbose=verbose, **kwargs)
                model.append(label_model)
        else:
            model = FSLabelModel(m=num_labelers, lambda_edges=dependency_graph)
            label_matrix_i = np.copy(label_matrix)
            abstain_mask = label_matrix_i == ABSTAIN
            negative_mask = label_matrix_i == 0
            label_matrix_i[abstain_mask] = 0
            label_matrix_i[negative_mask] = -1
            model.fit(L_train=label_matrix_i, class_balance=class_balance, verbose=verbose, **kwargs)
        
        self.model = model

    def predict_proba(self, label_matrix: np.ndarray, ABSTAIN: int = -1) -> np.ndarray:
        """Predict class probabilities for the given label matrix.

        Args:
            label_matrix (np.ndarray): Label matrix.
            ABSTAIN (int, optional): Value representing abstain. Defaults to -1.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        if self.cardinality > 2:
            probas = np.zeros((len(label_matrix), self.cardinality))
            for i in range(self.cardinality):
                label_matrix_i = np.copy(label_matrix)
                target_mask = label_matrix_i == i
                abstain_mask = label_matrix_i == ABSTAIN
                other_mask = (~target_mask) & (~abstain_mask)
                label_matrix_i[target_mask] = 1
                label_matrix_i[abstain_mask] = 0
                label_matrix_i[other_mask] = -1
                probas[:, i] = self.model[i].predict_proba(L_matrix=label_matrix_i)[:, 1]
            probas = np.nan_to_num(probas, nan=-np.inf)  # handle NaN
            probas = np.exp(probas) / np.sum(np.exp(probas), axis=1, keepdims=True)
        else:
            label_matrix_i = np.copy(label_matrix)
            abstain_mask = label_matrix_i == ABSTAIN
            negative_mask = label_matrix_i == 0
            label_matrix_i[abstain_mask] = 0
            label_matrix_i[negative_mask] = -1
            probas = self.model.predict_proba(L_matrix=label_matrix_i)
        return probas