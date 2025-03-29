from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Callable

import hashlib
import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class BaseDataset(ABC):
    @abstractmethod
    def __init__(
        self,
        features: list[str],
        split: str,
        label_matrix: np.array ,
        true_labels: np.array = None,
        transform_fn: Callable = None,
        embeddings: np.array = None,
        clean_unlabeled: bool = True,
        labeling_functions: list[str] = None,
        label_model: Any = None,
        label_model_fit_kwargs: dict = None,
        init: bool = True,
        sample_frac: float = None,
    ) -> None:
        """
        Abstract method to initialize the dataset object.

        Args:
            features (list[str]): list of feature names.
            split (str): Name of the split (e.g., "train", "dev", "test").
            label_matrix (np.array ): Label matrix of shape (num_examples, num_classes).
            true_labels (np.array , optional): True labels of shape (num_examples,). Defaults to None.
            transform_fn (Callable , optional): Function to transform examples. Defaults to None.
            embeddings (np.array , optional): Embeddings of shape (num_examples, embedding_dim). Defaults to None.
            clean_unlabeled (bool, optional): Whether to remove unlabeled examples. Defaults to True.
            labeling_functions (list[str] , optional): list of label functions. Defaults to None.
            label_model (Any, optional): The label model. Defaults to None.
            init (bool, optional): Whether to initialize the label model. Defaults to True.
            sample_frac (float , optional): Fraction of examples to sample. Defaults to None.
        """
        pass

    @abstractmethod
    def create_soft_labels(self) -> None:
        """
        Abstract method to create soft labels for the dataset.
        """
        pass

    @abstractmethod
    def remove_unlabeled_examples(self) -> None:
        """
        Abstract method to remove unlabeled examples from the dataset.
        """
        pass

    @abstractmethod
    def transform(self) -> None:
        """
        Abstract method to transform examples in the dataset.
        """
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        """
        Abstract method to get an example from the dataset.

        Args:
            index (int): Index of the example to get.

        Returns:
            dict: Dictionary containing feature vector and label vector.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Abstract method to get the length of the dataset.

        Returns:
            int: Number of examples in the dataset.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Abstract method to save the dataset to disk.

        Args:
            path (str): Path to save the dataset.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'BaseDataset':
        """
        Abstract method to load the dataset from disk.

        Args:
            path (str): Path to load the dataset from.

        Returns:
            BaseDataset: Loaded dataset object.
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """
        Abstract method to get a string representation of the dataset.

        Returns:
            str: String representation of the dataset.
        """
        pass


class WeakDataset(BaseDataset):
    def __init__(
        self,
        features: list[str],
        split: str,
        label_matrix: np.array ,
        true_labels: np.array = None,
        transform_fn: Callable = None,
        embeddings: np.array = None,
        clean_unlabeled: bool = True,
        labeling_functions: list[str] = None,
        label_model: Any = None,
        label_model_fit_kwargs: dict = None,
        init: bool = True,
        sample_frac: float = None,
    ) -> None:
        """
        A class representing a dataset.

        Args:
            features (list[str]): list of feature names.
            split (str): Name of the split (e.g., "train", "dev", "test").
            label_matrix (np.array ): Label matrix of shape (num_examples, num_classes).
            true_labels (np.array , optional): True labels of shape (num_examples,). Defaults to None.
            transform_fn (Callable , optional): Function to transform examples. Defaults to None.
            embeddings (np.array , optional): Embeddings of shape (num_examples, embedding_dim). Defaults to None.
            clean_unlabeled (bool, optional): Whether to remove unlabeled examples. Defaults to True.
            labeling_functions (list[str] , optional): list of label function. Defaults to None.
            label_model (Any, optional): The label model. Defaults to None.
            label_model_fit_kwargs (dict , optional): Keyword arguments for fitting the label model. Defaults to None.
            init (bool, optional): Whether to initialize the label model. Defaults to True.
            sample_frac (float , optional): Fraction of examples to sample. Defaults to None.
        """
        super().__init__(
            features=features,
            split=split,
            label_matrix=label_matrix,
            true_labels=true_labels,
            transform_fn=transform_fn,
            embeddings=embeddings,
            clean_unlabeled=clean_unlabeled,
            labeling_functions=labeling_functions,
            label_model=label_model,
            label_model_fit_kwargs=label_model_fit_kwargs,
            init=init,
            sample_frac=sample_frac,
        )

        # Validate input arguments
        assert split in ['train', 'valid', 'test'], "Split must be one of ['train', 'valid', 'test']"
        assert len(features) == len(label_matrix), "Features and label_matrix must have the same length"
        assert true_labels is None or len(features) == len(true_labels), "Features and true_labels must have the same length"
        assert transform_fn is None or callable(transform_fn), "transform_fn must be a callable function"
        assert transform_fn is not None or embeddings is not None, "Either transform_fn or embeddings must be provided"

        # Initialize instance variables
        self.features = features
        self.split = split
        self.label_matrix = label_matrix
        self.true_labels = true_labels
        self.transform_fn = transform_fn
        self.labeling_functions = labeling_functions
        self.lf_names = [lf.name for lf in self.labeling_functions] if self.labeling_functions is not None else None
        self.lf_patterns =[lf._f.__defaults__[0].pattern for lf in self.labeling_functions] if self.labeling_functions is not None else None
        self.label_model = label_model
        self.is_sample = sample_frac is not None
        self.sample_frac = 1.0 if sample_frac is None else sample_frac
        self.embeddings = embeddings
        self.label_model_fit_kwargs = {} if label_model_fit_kwargs is None else label_model_fit_kwargs

        # Remove unlabeled examples if requested
        if clean_unlabeled and self.split == 'train':
            self.remove_unlabeled_examples()

        # Sample the dataset if requested
        if self.is_sample:
            self._sample_dataset()

        self.sample_size = len(self.features)
        self.num_classes = len(set(self.true_labels)) if self.true_labels is not None else len([i for i in np.unique(self.label_matrix.flatten()) if i != -1])
        self.counter_votes = [dict(Counter(row)) for row in self.label_matrix]

        # Initialize label model and soft labels if requested
        if init:
            if self.label_model is not None:
                self.create_soft_labels()
            if self.embeddings is None:
                self.transform()

        self.embedding_dimensions = self.embeddings.shape[1] if self.embeddings is not None else None

    def create_soft_labels(self) -> None:
        """
        Create soft labels for the dataset.
        """
        assert self.label_model is not None, "Label model must be provided to create soft labels"
        self.label_model.fit(self.label_matrix, **self.label_model_fit_kwargs)
        self.soft_labels = self.label_model.predict_proba(self.label_matrix)

    def remove_unlabeled_examples(self) -> None:
        """
        Remove unlabeled examples from the training set.
        """
        # Check if all columns for every row are -1
        mask = (self.label_matrix != -1).any(axis=1)
        self.features = self.features[mask]
        if self.true_labels is not None:
            self.true_labels = self.true_labels[mask]
        self.label_matrix = self.label_matrix[mask]
        if self.embeddings is not None:
            self.embeddings = self.embeddings[mask]

    def transform(self) -> None:
        """
        Transform the features of the dataset using the provided transform function.
        """
        if self.transform_fn is not None:
            self.embeddings = self.transform_fn(self.features)
            self.embedding_dimensions = self.embeddings.shape[1]
        else:
            raise ValueError("No transform function provided!")

    def __getitem__(self, index: int) -> dict:
        """
        Get an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            dict: Dictionary containing the features, true labels, noisy labels, embedded features, labeling function names, and counter votes for the item.
        """
        return {
            'features': self.features[index],
            'true_labels': self.true_labels[index] if self.true_labels is not None else None,
            'label_matrix': self.label_matrix[index] if self.label_matrix is not None else None,
            'embeddings': self.embeddings[index] if self.embeddings is not None else None,
            'lfs_with_match': [(self.lf_names[i], self.lf_patterns[i], self.label_matrix[index][i]) for i in range(len(self.lf_names)) if self.label_matrix[index][i] != -1],
            'counter_votes': self.counter_votes[index],
            'soft_labels': self.soft_labels[index] if hasattr(self, 'soft_labels') else None,
        }

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Number of examples in the dataset.
        """
        return len(self.features)

    def save(self, path: str) -> None:
        """
        Save the dataset to a file.

        Args:
            path (str): Path to save the dataset to.
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'WeakDataset':
        """
        Load a dataset from a file.

        Args:
            path (str): Path to the file to load the dataset from.

        Returns:
            WeakDataset: Loaded dataset object.
        """
        return joblib.load(path)

    def __repr__(self) -> str:
        """
        Get a string representation of the dataset.

        Returns:
            str: String representation of the dataset.
        """
        return (f"WeakDataset(split='{self.split}', num_examples={len(self)}, "
                f"num_lfs={len(self.lf_names)}, num_classes={self.num_classes}, "
                f"embedding_dimensions={self.embedding_dimensions}, sample_frac={self.sample_frac}, "
                f"label_model={self.label_model})")

    def _sample_dataset(self) -> None:
        """
        Sample a fraction of the dataset.
        """
        sample_size = max(2, int(len(self.features) * self.sample_frac))
        np.random.seed(314)
        idxs = np.random.choice(len(self.features), sample_size, replace=False)
        self.features = self.features[idxs]
        self.label_matrix = self.label_matrix[idxs]
        if self.true_labels is not None:
            self.true_labels = self.true_labels[idxs]
        if self.embeddings is not None:
            self.embeddings = self.embeddings[idxs]


class ImageItem:
    """Represents an image with associated metadata and embedding.

    Attributes:
        image_data (Image): The image data.
        label_index (int): The integer label of the image.
        label_name (str): The string label of the image.
        embedding (np.ndarray): The embedding vector of the image.
    """

    def __init__(self, image_data: Image, label_index: int, label_name: str, embedding: np.ndarray):
        """Initializes an ImageItem with image data, label, and embedding.

        Args:
            image_data (Image): The image data.
            label_index (int): The integer label of the image.
            label_name (str): The string label of the image.
            embedding (np.ndarray): The embedding vector of the image.
        """
        self.image_data = image_data
        self.label_index = label_index
        self.label_name = label_name
        self.embedding = embedding
        self.hash = hashlib.md5(image_data.tobytes()).hexdigest()

    def __repr__(self) -> str:
        """Displays the image with its label.

        Returns:
            str: A string representation of the ImageItem.
        """
        plt.figure()
        plt.title(self.label_name)
        plt.imshow(np.array(self.image_data))
        plt.axis('off')
        plt.show()
        return f"ImageItem(label_name={self.label_name})"

class SimpleImageDataset:
    """A dataset of ImageItem objects.

    Attributes:
        image_items (list[ImageItem]): A list of ImageItem objects.
    """

    def __init__(self, image_items: list[ImageItem], split: str):
        """Initializes the ImageDataset with a list of ImageItem objects.

        Args:
            image_items (list[ImageItem]): A list of ImageItem objects.
            split (str): The name of the split (e.g., "train", "valid", "test").

        """
        assert split in ['train', 'valid', 'test'], "Split must be one of ['train', 'valid', 'test']"
        self.image_items = image_items
        self.split = split

    def __getitem__(self, index: int) -> ImageItem:
        """Gets the ImageItem at the specified index.

        Args:
            index (int): The index of the ImageItem to retrieve.

        Returns:
            ImageItem: The ImageItem at the specified index.
        """
        return self.image_items[index]

    def __len__(self) -> int:
        """Gets the number of ImageItem objects in the dataset.

        Returns:
            int: The number of ImageItem objects in the dataset.
        """
        return len(self.image_items)

    def __repr__(self) -> str:
        """Gets a string representation of the ImageDataset.

        Returns:
            str: A string representation of the ImageDataset.
        """
        return f"ImageDataset with {len(self.image_items)} images in the {self.split} split"
    
    def save(self, path: str) -> None:
        """Saves the ImageDataset to a file.

        Args:
            path (str): The path to save the ImageDataset to.
        """
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> 'SimpleImageDataset':
        """Loads an ImageDataset from a file.

        Args:
            path (str): The path to load the ImageDataset from.

        Returns:
            ImageDataset: The loaded ImageDataset.
        """
        return joblib.load(path)