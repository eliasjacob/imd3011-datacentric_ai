# Adapted from Wrench

from typing import Any, Callable, Iterable,Namedtuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from tqdm.auto import tqdm

from helpers.dataset import WeakDataset

#### Influence functions ####

def save_influence_score(
    influence_score: Any, 
    influence_type: str, 
    mode: str, 
    learning_rate: float, 
    weight_decay: float, 
    num_epochs: int, 
    batch_size: int, 
    base_filename: str
) -> None:
    """
    Save the influence score to a file with a name based on input parameters.

    Args:
        influence_score (Any): The influence score to be saved.
        influence_type (str): The type of influence score.
        mode (str): The mode in which the influence score was computed.
        learning_rate (float): The learning rate used in the computation.
        weight_decay (float): The weight decay used in the computation.
        num_epochs (int): The number of epochs used in the computation.
        batch_size (int): The batch size used in the computation.
        base_filename (str): The base name for the file.

    Returns:
        None
    """
    # Create a filename based on the input parameters
    filename = f'{base_filename}_{influence_type}_{mode}_{learning_rate}_{weight_decay}_{num_epochs}_{batch_size}.joblib'
    
    # Save the influence score to the specified filename using joblib
    joblib.dump(influence_score, filename)

def compute_item_scores(
        influence_scores: np.ndarray, 
        item_type: str = 'lfs'
    ) -> np.ndarray:
    """
    Compute the sum of influence scores for each item of a given type.

    Args:
        influence_scores (np.ndarray): A numpy array representing the influence scores.
        item_type (str, optional): A string representing the type of items to compute scores for ('lfs', 'examples', 'classes', or 'source-aware'). Defaults to 'lfs'.

    Returns:
        np.ndarray: A numpy array representing the sum of influence scores for each item of the specified type.
    """
    assert item_type in ['lfs', 'examples', 'classes', 'source-aware'], "item_type must be one of ['lfs', 'examples', 'classes', 'source-aware']"

    # Define the columns to aggregate for each item type
    columns_to_aggregate = {
        'lfs': (0, 2),
        'examples': (1, 2),
        'classes': (0, 1),
        'source-aware': 2
    }

    # Compute the sum of influence scores for each item of the specified type
    item_scores = influence_scores.sum(axis=columns_to_aggregate[item_type])
    return item_scores


def get_top_k_indices(
        influence_scores: np.ndarray, 
        k: int = 10, 
        item_type: str = 'lfs',
        dataset: WeakDataset = None
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the indices of the top k items with the highest and lowest influence scores.

    Args:
        influence_scores (np.ndarray): A numpy array representing the influence scores.
        k (int, optional): An integer representing the number of items to select. Defaults to 10.
        item_type (str, optional): A string representing the type of items to select ('lfs', 'examples', 'classes', or 'source-aware'). Defaults to 'lfs'.
        dataset (WeakDataset, optional): A WeakDataset object representing the training dataset. Defaults to None.

    Returns:
        tuple: A tuple containing the indices of the top k items with the lowest and highest influence scores.
    """
    assert item_type != 'lfs' or dataset is not None, "Dataset must be provided when item_type is 'lfs'"

    # Compute the sum of influence scores for each item of the specified type
    item_scores = compute_item_scores(influence_scores, item_type=item_type)

    if item_type == 'lfs':
        # Get the number of examples that each labeling function labeled
        num_labeled_examples = np.sum(dataset.label_matrix != -1, axis=0)

        # Compute the mean influence score for each labeling function
        mean_item_scores = item_scores / num_labeled_examples
        print(mean_item_scores.shape, item_scores.shape, num_labeled_examples.shape)
        worst_indices = np.argsort(mean_item_scores)[:k]
        best_indices = np.argsort(mean_item_scores)[::-1][:k]
    else:
        # Get the indices of the top k items with the lowest and highest influence scores
        worst_indices = np.argsort(item_scores)[:k]
        best_indices = np.argsort(item_scores)[::-1][:k]

    return worst_indices, best_indices


def get_top_k_items(
        dataset: WeakDataset, 
        influence_scores: np.ndarray, 
        k: int = 10, 
        item_type: str = 'lfs'
    ) -> pd.DataFrame:
    """
    Get a table of the top k items with the lowest and highest influence scores.

    Args:
        dataset (WeakDataset): A WeakDataset object representing the training dataset.
        influence_scores (np.ndarray): A numpy array representing the influence scores.
        k (int, optional): An integer representing the number of items to select. Defaults to 10.
        item_type (str, optional): A string representing the type of items to select ('lfs', 'examples', 'classes', or 'source-aware'). Defaults to 'lfs'.

    Returns:
        pandas.DataFrame: A pandas DataFrame representing the top k items with the lowest and highest influence scores.
    """
    
    if item_type != 'lfs':
        # Get the indices of the top k items with the lowest and highest influence scores
        worst_indices, best_indices = get_top_k_indices(influence_scores, k=k, item_type=item_type)
    else:
        # Get the number of examples that the labeling function labeled
        worst_indices, best_indices = get_top_k_indices(influence_scores, k=k, item_type=item_type, dataset=dataset)

    # Create a list of dictionaries representing the top k items with the lowest and highest influence scores
    rows = []
    if item_type == 'examples':
        for i in worst_indices:
            item = dataset.features[i]
            votes = dataset.counter_votes[i]
            label = dataset.soft_labels[i]
            score = compute_item_scores(influence_scores, item_type=item_type)[i]
            rows.append({'item': item, 'votes': votes, 'label_prob': label, 'label_cat': np.argmax(label), 'probability_class': label[np.argmax(label)] , 'influence_score': score, 'influence_type': 'BAD'})
        for i in best_indices:
            item = dataset.features[i]
            votes = dataset.counter_votes[i]
            label = dataset.soft_labels[i]
            score = compute_item_scores(influence_scores, item_type=item_type)[i]
            rows.append({'item': item, 'votes': votes, 'label_prob': label, 'label_cat': np.argmax(label), 'probability_class': label[np.argmax(label)] ,'influence_score': score, 'influence_type': 'GOOD'})

    elif item_type == 'lfs':
        if len(dataset.lf_names) < 5*2:
            k = len(dataset.lf_names)//2
        for i in worst_indices:
            item = dataset.lf_names[i]
            score = compute_item_scores(influence_scores, item_type=item_type)[i]
            rows.append({'item': item, 'influence_score': score, 'influence_type': 'BAD'})
        for i in best_indices:
            item = dataset.lf_names[i]
            score = compute_item_scores(influence_scores, item_type=item_type)[i]
            rows.append({'item': item, 'influence_score': score, 'influence_type': 'GOOD'})

    # Create a pandas DataFrame from the list of dictionaries
    df = pd.DataFrame(rows)
    df = df.sort_values(by='influence_score', ascending=False).reset_index(drop=True)

    # The column votes contain a dict. We want to split this dict into multiple columns
    if item_type == 'examples':
        df = pd.concat([df.drop(['votes'], axis=1), df['votes'].apply(pd.Series)], axis=1)
    df.fillna(0, inplace=True)
    
    return df


#### Misc ####

def calculate_outliers_iqr(
        input_iter: Iterable, 
        multiplier: float = 1.5
    ) -> dict[str, Any]:
    """
    Calculates the outliers of a given iterable using the interquartile range (IQR) method.

    Args:
        input_iter (Iterable): The iterable to calculate the outliers from.
        multiplier (float, optional): The number of IQR multiplier to consider as outliers. Defaults to 1.5.

    Returns:
        dict[str, Any]: A dictionary containing the lower bound, upper bound, fraction of outliers, fraction of lower bound outliers,
        fraction of upper bound outliers, non-outliers, and outliers.
    """
    # Convert the input iterable to a pandas Series
    input_series = pd.Series(input_iter)

    # Calculate the first quartile (Q1)
    q1 = input_series.quantile(0.25)

    # Calculate the third quartile (Q3)
    q3 = input_series.quantile(0.75)

    # Calculate the interquartile range (IQR)
    iqr = q3 - q1

    # Calculate the lower and upper bounds
    lower_bound = q1 - (multiplier * iqr)
    upper_bound = q3 + (multiplier * iqr)

    # Calculate the fraction of outliers
    fraction_outliers = (len(input_series[(input_series < lower_bound) | (input_series > upper_bound)]) / len(input_series))

    # Calculate the fraction of lower bound outliers
    fraction_lower_bound_outliers = (len(input_series[input_series < lower_bound]) / len(input_series))

    # Calculate the fraction of upper bound outliers
    fraction_upper_bound_outliers = (len(input_series[input_series > upper_bound]) / len(input_series))

    # Get the non-outliers and outliers
    non_outliers = input_series[(input_series >= lower_bound) & (input_series <= upper_bound)]
    outliers = input_series[(input_series < lower_bound) | (input_series > upper_bound)]

    # Return the results as a dictionary

    return {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'fraction_outliers': fraction_outliers,
        'fraction_lower_bound_outliers': fraction_lower_bound_outliers,
        'fraction_upper_bound_outliers': fraction_upper_bound_outliers,
        'non_outliers': non_outliers,
        'outliers': outliers
    }



### pyDVL helpers ###

class Losses(Namedtuple):
    training: NDArray[np.float64]
    validation: NDArray[np.float64]

class TorchLogisticRegression(nn.Module):
    """
    A simple binary logistic regression model.
    """

    def __init__(
        self,
        n_input: int,
    ):
        """
        Args:
            n_input: Number of features in the input.
        """
        super().__init__()
        self.fc1 = nn.Linear(n_input, 1, bias=True, dtype=float)

    def forward(self, x):
        """
        Args:
            x: Tensor [NxD], with N the batch length and D the number of features.
        Returns:
            A tensor [N] representing the probability of the positive class for each sample.
        """
        x = torch.as_tensor(x)
        return torch.sigmoid(self.fc1(x))


class TorchMLP(nn.Module):
    """
    A simple fully-connected neural network
    """

    def __init__(
        self,
        layers_size: list[int],
    ):
        """
        Args:
            layers_size: list of integers representing the number of
                neurons in each layer.
        """
        super().__init__()
        if len(layers_size) < 2:
            raise ValueError(
                "Passed layers_size has less than 2 values. "
                "The network needs at least input and output sizes."
            )
        layers = []
        for frm, to in zip(layers_size[:-1], layers_size[1:]):
            layers.append(nn.Linear(frm, to))
            layers.append(nn.Tanh())
        layers.pop()

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass through the network.

        Args:
            x: Tensor input of shape [NxD], with N batch size and D number of
            features.

        Returns:
            Tensor output of shape[NxK], with K the output size of the network.
        """
        return self.layers(x)


def fit_torch_model(
    model: nn.Module,
    training_data: DataLoader,
    val_data: DataLoader,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: Optimizer,
    scheduler: _LRScheduler = None,
    num_epochs: int = 1,
    progress: bool = True,
    device: torch.device = torch.device("cpu"),
) -> Losses:
    """
    Fits a pytorch model to the supplied data.
    Represents a simple machine learning loop, iterating over a number of
    epochs, sampling data with a certain batch size, calculating gradients and updating the parameters through a
    loss function.

    Args:
        model: A pytorch model.
        training_data: A pytorch DataLoader with the training data.
        val_data: A pytorch DataLoader with the validation data.
        optimizer: Select either ADAM or ADAM_W.
        scheduler: A pytorch scheduler. If None, no scheduler is used.
        num_epochs: Number of epochs to repeat training.
        progress: True, iff progress shall be printed.
        device: Device on which the model is and to which the batches should be moved.
    """
    train_loss = []
    val_loss = []

    for epoch in tqdm(range(num_epochs), disable=not progress, desc="Model fitting"):
        batch_loss = []
        for train_batch in training_data:
            batch_x, batch_y = train_batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred_y = model(batch_x)
            loss_value = loss(torch.squeeze(pred_y), torch.squeeze(batch_y))
            batch_loss.append(loss_value.cpu().item())
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()

            if scheduler:
                scheduler.step()
        with torch.no_grad():
            batch_val_loss = []
            for val_batch in val_data:
                batch_x, batch_y = val_batch
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred_y = model(batch_x)
                loss_value = loss(torch.squeeze(pred_y), torch.squeeze(batch_y))
                batch_val_loss.append(loss_value.cpu().item())

        mean_epoch_train_loss = np.mean(batch_loss)
        mean_epoch_val_loss = np.mean(batch_val_loss)
        train_loss.append(mean_epoch_train_loss)
        val_loss.append(mean_epoch_val_loss)
        print(
            f"Epoch: {epoch} ---> Training loss: {mean_epoch_train_loss}, Validation loss: {mean_epoch_val_loss}"
        )
    return Losses(train_loss, val_loss)


def new_resnet_model(output_size: int) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    # Fine-tune final few layers
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, output_size)

    return model

def plot_influences(
    x: NDArray[np.float64],
    influences: NDArray[np.float64],
    corrupted_indices: list[int] = None,
    *,
    ax: plt.Axes = None,
    xlabel: str = None,
    ylabel: str = None,
    legend_title: str = None,
    line: NDArray[np.float64] = None,
    suptitle: str = None,
    colorbar_limits: tuple = None,
) -> plt.Axes:
    """Plots the influence values of the training data with a color map.

    Args:
        x: Input to the model, of shape (N,2) with N being the total number
            of points.
        influences: an array  of shape (N,) with influence values for each
            data point.
        line: Optional, line of shape [Mx2], where each row is a point of the
            2-dimensional line. (??)
    """
    if ax is None:
        _, ax = plt.subplots()
    sc = ax.scatter(x[:, 0], x[:, 1], c=influences)
    if line is not None:
        ax.plot(line[:, 0], line[:, 1], color="black")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if suptitle is not None:
        plt.suptitle(suptitle)

    plt.colorbar(sc, label=legend_title)
    if colorbar_limits is not None:
        sc.clim(*colorbar_limits)
    if corrupted_indices is not None:
        ax.scatter(
            x[:, 0][corrupted_indices],
            x[:, 1][corrupted_indices],
            facecolors="none",
            edgecolors="r",
            s=80,
        )
    return ax

def plot_losses(losses: Losses):
    """Plots the train and validation loss

    Args:
        training_loss: list of training losses, one per epoch
        validation_loss: list of validation losses, one per epoch
    """
    _, ax = plt.subplots()
    ax.plot(losses.training, label="Train")
    ax.plot(losses.validation, label="Val")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Train epoch")
    ax.legend()
    plt.show()
