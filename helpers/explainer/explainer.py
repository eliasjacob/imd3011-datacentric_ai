from typing import Any, Optional, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from backpack import extend
from scipy.optimize import least_squares
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from helpers.explainer.influence_function import InfluenceFunction

ABSTAIN = -1

activation_func_dict = {
    'identity': lambda x: x,
    'exp': lambda x: np.exp(x),
    'taylor': lambda x: 1 + x + 0.5 * (x ** 2),
}


class AbstractModel(torch.nn.Module):
    def count_parameters(self) -> int:
        """
        Counts the number of trainable parameters in the model.

        Returns:
            int: Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def collect_grad(self) -> torch.Tensor:
        """
        Collects the gradients of all trainable parameters in the model.

        Returns:
            torch.Tensor: Concatenated gradients of all trainable parameters.
        """
        return torch.cat([p.grad.reshape(-1) for p in self.parameters() if p.requires_grad], dim=0)

    def collect_batch_grad(self, params: Optional[List[torch.nn.Parameter]] = None) -> torch.Tensor:
        """
        Collects the gradients of all trainable parameters in the model for each batch.

        Args:
            params (Optional[List[torch.nn.Parameter]]): List of parameters to collect gradients for (optional).

        Returns:
            torch.Tensor: Concatenated gradients of all trainable parameters for each batch.
        """
        batch_grad_cache = []
        if params is not None:
            for param in params:
                if param.requires_grad:
                    batch_grad_cache.append(param.grad_batch.reshape(param.grad_batch.shape[0], -1))
        else:
            for param in self.parameters():
                if param.requires_grad:
                    batch_grad_cache.append(param.grad_batch.reshape(param.grad_batch.shape[0], -1))

        batch_grad_cache = torch.cat(batch_grad_cache, dim=1)
        return batch_grad_cache

    def num_of_layers(self) -> int:
        """
        Counts the number of layers in the model.

        Returns:
            int: Number of layers.
        """
        return sum(1 for param in self.parameters() if param.requires_grad)

    def separate_batch_grad(self, batch_grad_cache: torch.Tensor, params: Optional[List[torch.nn.Parameter]] = None) -> List[torch.Tensor]:
        """
        Separates the gradients of all trainable parameters in the model for each layer.

        Args:
            batch_grad_cache (torch.Tensor): Concatenated gradients of all trainable parameters for each batch.
            params (Optional[List[torch.nn.Parameter]]): List of parameters to separate gradients for (optional).

        Returns:
            List[torch.Tensor]: List of gradients of all trainable parameters for each layer.
        """
        num_param_per_layer = []
        if params is not None:
            for param in params:
                if param.requires_grad:
                    num_param_per_layer.append(param.numel())
        else:
            for param in self.parameters():
                if param.requires_grad:
                    num_param_per_layer.append(param.numel())

        grad_per_layer_list = []
        counter = 0
        for num_param in num_param_per_layer:
            if len(batch_grad_cache.shape) == 2:
                grad_per_layer_list.append(batch_grad_cache[:, counter:counter + num_param])
            else:
                grad_per_layer_list.append(batch_grad_cache[counter:counter + num_param])
            counter += num_param

        return grad_per_layer_list


class LinearModel(AbstractModel):
    def __init__(self, input_size: int, num_classes: int) -> None:
        """
        Initializes an instance of the LinearModel class.

        Args:
            input_size (int): Size of input.
            num_classes (int): Number of classes.
        """
        super(LinearModel, self).__init__()
        self.input_size = input_size
        self.output_size = 1 if num_classes == 2 else num_classes
        self.fc1 = torch.nn.Linear(self.input_size, self.output_size)
        self.ce_loss = CrossEntropyLoss()
        self.ce_loss_sum = CrossEntropyLoss(reduction='sum')

        torch.nn.init.zeros_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass on the input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        output = self.fc1(x.reshape(x.shape[0], -1))
        if self.output_size == 1:
            output = torch.cat([F.logsigmoid(-output), F.logsigmoid(output)], dim=-1)
        return output


def train_model(
        learning_rate: float, 
        weight_decay: float, 
        num_epochs: int, 
        input_size: int, 
        num_classes: int, 
        train_dataloader: DataLoader, 
        device: torch.device, 
        verbose: bool = True
    ) -> LinearModel:
    """
    Trains the linear model.

    Args:
        learning_rate (float): Learning rate.
        weight_decay (float): Weight decay.
        num_epochs (int): Number of epochs.
        input_size (int): Size of input.
        num_classes (int): Number of classes.
        train_dataloader (DataLoader): Training data loader.
        device (torch.device): Device to use.
        verbose (bool): Whether to print progress (optional).

    Returns:
        LinearModel: Trained linear model.
    """
    model = LinearModel(input_size, num_classes)
    model.to(device)

    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=weight_decay)
    for epoch in trange(num_epochs, disable=not verbose, desc='model training'):
        for _, x, y in train_dataloader:
            model.train()
            model.zero_grad()
            logit = model(x)
            loss = model.ce_loss(logit, y)
            loss.backward()
            optimizer.step()

    return model


class Explainer:
    def __init__(self, num_labeling_functions: int, num_classes: int, **kwargs: Any) -> None:
        """
        Initializes an instance of the Explainer class.

        Args:
            num_labeling_functions (int): Number of labeling functions.
            num_classes (int): Number of classes.
            **kwargs: Additional keyword arguments.
        """
        self.num_labeling_functions = num_labeling_functions
        self.num_classes = num_classes
        self.weights = None
        self.activation_func = None

    def augment_label_matrix(self, label_matrix: List[int]) -> np.ndarray:
        """
        Augments the label matrix.

        Args:
            label_matrix (List[int]): Label matrix.

        Returns:
            np.ndarray: Augmented label matrix.
        """
        label_matrix = np.array(label_matrix) + 1
        augmented_label_matrix = np.eye(self.num_classes + 1)[label_matrix]
        return augmented_label_matrix

    def approximate_label_model(self, label_matrix: List[int], noisy_labels: np.ndarray, initial_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Approximates the label model using least squares optimization.

        Args:
            label_matrix (List[int]): Label matrix.
            noisy_labels (np.ndarray): Noisy labels predicted by the Label Model.
            initial_weights (Optional[np.ndarray]): Initial weights (optional).

        Returns:
            np.ndarray: Approximate label model weights.
        """
        augmented_label_matrix = self.augment_label_matrix(label_matrix)

        num_samples, num_labeling_functions, num_classes_plus_one = augmented_label_matrix.shape
        num_classes = num_classes_plus_one - 1

        reshaped_label_matrix = augmented_label_matrix.reshape(num_samples, -1)
        if initial_weights is not None:
            initial_weights = initial_weights - initial_weights.min(axis=-1, keepdims=True)
        else:
            initial_weights = np.zeros(shape=(num_labeling_functions, num_classes_plus_one, num_classes))

        def loss_function(weights: np.ndarray) -> np.ndarray:
            weights = weights.reshape(num_labeling_functions * num_classes_plus_one, num_classes)
            probabilities = reshaped_label_matrix @ weights
            normalization_factor = np.sum(probabilities, axis=1, keepdims=True)
            predicted_labels = probabilities / normalization_factor
            return (predicted_labels - noisy_labels).flatten()

        result = least_squares(loss_function, initial_weights.flatten(), bounds=(0, np.inf))

        approximate_weights = result.x.reshape(initial_weights.shape)
        self.register_label_model(approximate_weights, 'identity')
        return approximate_weights

    def register_label_model(self, weights: np.ndarray, activation_func: str = 'identity') -> None:
        """
        Registers the label model.

        Args:
            weights (np.ndarray): Label model weights.
            activation_func (str): Activation function (optional).
        """
        assert activation_func in ['identity', 'exp']
        self.weights = weights
        self.activation_func = activation_func

    def apply_label_model(self, label_matrix: List[int]) -> np.ndarray:
        """
        Applies the label model to the label matrix.

        Args:
            label_matrix (List[int]): Label matrix.

        Returns:
            np.ndarray: Predicted labels.
        """
        augmented_label_matrix = self.augment_label_matrix(label_matrix)
        raw_scores = np.einsum('ijk,jkl->il', augmented_label_matrix, self.weights)
        raw_scores = activation_func_dict[self.activation_func](raw_scores)
        normalization_factor = np.sum(raw_scores, axis=1, keepdims=True)
        predicted_labels = raw_scores / normalization_factor
        return predicted_labels

    def compute_influence_function_score(
            self, 
            train_label_matrix: List[int], 
            train_data: List[np.ndarray], 
            valid_data: List[np.ndarray], 
            valid_labels: List[int], 
            influence_function_type: str, 
            mode: str, 
            learning_rate: float, 
            weight_decay: float, 
            num_epochs: int, 
            batch_size: int, 
            device: Optional[torch.device] = None, 
            damping: float = 1.0, 
            scaling: float = 25.0, 
            num_recursions: int = 2, 
            recursion_depth: int = 100,
        ) -> float:
        """
        Computes the influence function score.

        Args:
            train_label_matrix (List[int]): Training label matrix.
            train_data (List[np.ndarray]): Training data.
            valid_data (List[np.ndarray]): Valid data.
            valid_labels (List[int]): Valid labels.
            influence_function_type (str): Influence function type.
            mode (str): Influence function mode.
            learning_rate (float): Learning rate.
            weight_decay (float): Weight decay.
            num_epochs (int): Number of epochs.
            batch_size (int): Batch size.
            device (Optional[torch.device]): Device to use (optional).
            damping (float): Damping parameter (optional).
            scaling (float): Scaling parameter (optional).
            num_recursions (int): Number of recursions (optional).
            recursion_depth (int): Recursion depth (optional).

        Returns:
            float: Influence function score.
        """
        train_labels = self.apply_label_model(train_label_matrix)

        train_data = np.array(train_data)
        valid_data = np.array(valid_data)
        valid_labels_one_hot = np.eye(self.num_classes)[np.array(valid_labels)]

        # Construct dataloaders
        train_dataset = TensorDataset(
            torch.LongTensor(list(range(train_data.shape[0]))).to(device),
            torch.FloatTensor(train_data).to(device),
            torch.FloatTensor(train_labels).to(device)
        )
        train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset) if batch_size == -1 else batch_size, shuffle=True)

        valid_dataset = TensorDataset(
            torch.LongTensor(list(range(valid_data.shape[0]))).to(device),
            torch.FloatTensor(valid_data).to(device),
            torch.FloatTensor(valid_labels_one_hot).to(device)
        )

        # Train the model
        model = train_model(learning_rate, weight_decay, num_epochs, train_data.shape[1], self.num_classes, train_dataloader, device, verbose=True)

        # Extend the model for influence function computation
        model = extend(model)
        model.ce_loss_sum = extend(model.ce_loss_sum)

        train_dataset_for_if = TensorDataset(
            torch.LongTensor(list(range(train_data.shape[0]))).to(device),
            torch.FloatTensor(train_data).to(device),
            torch.LongTensor(train_label_matrix).to(device),
            torch.FloatTensor(train_labels).to(device)
        )

        # Compute the influence function score
        influence_function = InfluenceFunction(model, train_dataset_for_if, valid_dataset, self.num_labeling_functions, self.num_classes, device, damp=damping, scale=scaling, r=num_recursions, recursion_depth=recursion_depth)
        influence_function_score = influence_function.compute_influence_function(
            if_type=influence_function_type, 
            mode=mode,
            weights=torch.FloatTensor(self.weights).to(device), 
            activation_function=self.activation_func,
            batch_mode=False,
            )

        return influence_function_score