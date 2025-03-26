# TODO check if I really need this level of complexity for a class
# FIXME

import os
import time
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from lightning.pytorch.tuner import Tuner
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, matthews_corrcoef
from torch.utils.data import DataLoader, Dataset

import helpers.classification


class SimpleDiscriminativeNetwork(nn.Module):
    """
    A simple discriminative neural network with customizable architecture.

    Args:
        input_size (int): The size of the input layer.
        output_size (int): The size of the output layer.
        dropout (float, optional): The dropout rate to use in the network. Defaults to 0.5.
        arch (int, optional): The architecture of the network to use. Defaults to 1.
    """
    def __init__(self, input_size: int, output_size: int, dropout: float = 0.5, arch: int = 1):
        super().__init__()

        if arch == 0:
            self.nn = nn.Linear(input_size, output_size)
        else:
            layers = []
            layer_sizes = [input_size] + [input_size // (2 ** (i+1)) for i in range(arch - 1)] + [output_size]
            for i in range(arch):
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
                if i < arch - 1:
                    layers.append(nn.Mish())
                    layers.append(nn.Dropout(dropout))
            self.nn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.

        Args:
            x (torch.Tensor): The input tensor to the network.

        Returns:
            torch.Tensor: The output tensor from the network.
        """
        return self.nn(x)

class DiscriminativeModule(L.LightningModule):
    """
    A PyTorch Lightning module for training a discriminative neural network.

    Args:
        input_size (int): The size of the input layer.
        output_size (int): The size of the output layer.
        learning_rate (float): The learning rate to use for training.
        arch (int, optional): The architecture of the network to use. Defaults to 1.
        dropout (float, optional): The dropout rate to use in the network. Defaults to 0.5.
        class_weights (np.ndarray, optional): The class weights to use for training. Defaults to None.
    """
    def __init__(self, input_size: int, output_size: int, learning_rate: float, arch: int = 1, dropout: float = 0.5, class_weights: np.ndarray = None):
        super().__init__()
        self.model = SimpleDiscriminativeNetwork(input_size=input_size, output_size=output_size, arch=arch, dropout=dropout)
        self.loss_weights = self._convert_class_weights_to_loss_weights(class_weights)
        self.loss = nn.CrossEntropyLoss(weight=self.loss_weights)
        self.num_classes = output_size
        self.learning_rate = learning_rate
        self.class_weights = class_weights
        self.save_hyperparameters()
            
    def _convert_class_weights_to_loss_weights(self, class_weights: np.ndarray) -> torch.Tensor:
        """
        Convert class weights to loss weights.

        Args:
            class_weights (np.ndarray): The class weights to convert.

        Returns:
            torch.Tensor: The converted class weights as a PyTorch tensor, or None if class_weights is None.
        """
        if class_weights is None:
            return None
        return torch.tensor(1 - np.array(class_weights))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.

        Args:
            x (torch.Tensor): The input tensor to the network.

        Returns:
            torch.Tensor: The output tensor from the network.
        """
        return self.model(x)

    def step(self, batch: tuple[torch.Tensor, torch.Tensor], name: str = 'train') -> torch.Tensor:
        """
        Perform a single training/validation/test step.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): A tuple containing the input and target tensors for the batch.
            name (str, optional): The name of the step (train/val/test). Defaults to 'train'.

        Returns:
            torch.Tensor: The loss value for the step.
        """
        x, y = batch
        if name == 'train':
            x, sample_weights = x[:, :-1], x[:, -1]

        # Flatten the input tensor
        x = x.view(x.size(0), -1)

        # Forward pass through the neural network
        y_pred = self.forward(x)

        # Compute metrics and loss
        y_pred_cat = torch.argmax(y_pred, dim=1)
        y_true_cat = torch.argmax(y, dim=1)
        mcc = torchmetrics.functional.matthews_corrcoef(y_pred_cat, y_true_cat, task='multiclass', num_classes=self.num_classes)
        f1 = torchmetrics.functional.f1_score(y_pred_cat, y_true_cat, num_classes=self.num_classes, task='multiclass', average='weighted')
        acc = torchmetrics.functional.accuracy(y_pred_cat, y_true_cat, num_classes=self.num_classes, task='multiclass')
        roc_auc = torchmetrics.functional.auroc(y_pred, y_true_cat, num_classes=self.num_classes, task='multiclass')

        loss = (self.loss(y_pred, y) * sample_weights).mean() if name == 'train' else self.loss(y_pred, y)
            
        # Log metrics and loss
        self.log_dict({
            f'{name}_loss': loss,
            f'{name}_mcc': mcc,
            f'{name}_f1': f1,
            f'{name}_acc': acc,
            f'{name}_roc_auc': roc_auc
        }, on_epoch=True, prog_bar=True, logger=True, on_step=name == 'train')
        
        return loss
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Perform a single training step.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): A tuple containing the input and target tensors for the batch.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value for the step.
        """
        return self.step(batch, name='train')

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Perform a single validation step.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): A tuple containing the input and target tensors for the batch.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value for the step.
        """
        return self.step(batch, name='val')

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Perform a single test step.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): A tuple containing the input and target tensors for the batch.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value for the step.
        """
        return self.step(batch, name='test')
    
    def predict_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """
        Perform a single prediction step.

        Args:
            batch (torch.Tensor): The input tensor for the batch.
            batch_idx (int): The index of the batch.
            dataloader_idx (int, optional): The index of the dataloader. Defaults to 0.

        Returns:
            torch.Tensor: The output tensor from the network.
        """
        return self(batch)
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: The optimizer to use for training.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

class SimpleDataset(Dataset):
    """
    A simple PyTorch dataset class for loading data.

    Args:
        data (list[tuple[np.ndarray, np.ndarray]]): A list of tuples containing the input and target data.
    """
    def __init__(self, data: list[tuple[np.ndarray, np.ndarray]]):
        self.data = data

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single item from the dataset.

        Args:
            idx (int): The index of the item to get.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the input and target data as PyTorch tensors.
        """
        x, y = self.data[idx]

        # Convert the input and target data to PyTorch tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        return x, y

class SimpleDataModule(L.LightningDataModule):
    """
    A PyTorch Lightning data module for loading and preparing data.

    Args:
        train_data (list[tuple[np.ndarray, np.ndarray]]): A list of tuples containing the input and target data for training.
        val_data (list[tuple[np.ndarray, np.ndarray]]): A list of tuples containing the input and target data for validation.
        test_data (list[tuple[np.ndarray, np.ndarray]]): A list of tuples containing the input and target data for testing.
        batch_size (int, optional): The batch size to use for loading data. Defaults to 32.
        num_workers (int, optional): The number of worker processes to use for data loading. Defaults to 1.
    """
    def __init__(self, train_data: list[tuple[np.ndarray, np.ndarray]] = None,
                 val_data: list[tuple[np.ndarray, np.ndarray]] = None,
                 test_data: list[tuple[np.ndarray, np.ndarray]] = None,
                 batch_size: int = 32, num_workers: int = 1):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        """
        Prepare the data for loading. This method is called only on a single GPU.
        """
        # You can add any data preparation steps here if needed
        pass

    def setup(self, stage: str = None):
        """
        Set up the datasets for loading.

        Args:
            stage  str): The stage of training (fit/test/predict). Defaults to None.
        """
        if stage == 'fit' or stage is None:
            if self.train_data is not None:
                self.train_dataset = SimpleDataset(self.train_data)
            if self.val_data is not None:
                self.val_dataset = SimpleDataset(self.val_data)
        if stage == 'test' or stage is None:
            if self.test_data is not None:
                self.test_dataset = SimpleDataset(self.test_data)

    def train_dataloader(self) -> DataLoader:
        """
        Get the dataloader for training data.

        Returns:
            DataLoader: The dataloader for training data, or None if no training data is available.
        """
        if self.train_data is None:
            return None
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """
        Get the dataloader for validation data.

        Returns:
            DataLoader: The dataloader for validation data, or None if no validation data is available.
        """
        if self.val_data is None:
            return None
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self) -> DataLoader:
        """
        Get the dataloader for test data.

        Returns:
            DataLoader: The dataloader for test data, or None if no test data is available.
        """
        if self.test_data is None:
            return None
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class SimpleDiscriminativeClassifier(BaseEstimator, ClassifierMixin):
    """
    A Sklearn-compatible wrapper for DiscriminativeModule.
    """

    def __init__(self, input_size: int, output_size: int, arch: int = 1, learning_rate: float = 3e-4,
                 dropout: float = 0.5, batch_size: int = None, max_epochs: int = 20,
                 patience: int = 3, device: str = 'cpu', auto_lr: bool = False,
                 enable_progress_bar: bool = False, deterministic: bool = True,
                 early_stopping: bool = True, early_stopping_monitor: str = 'val_mcc',
                 early_stopping_dirpath: str = 'tmp/', early_stopping_filename: str = 'model-best',
                 early_stopping_save_top_k: int = 1, early_stopping_mode: str = 'max',
                 X_valid: np.ndarray = None, y_valid: np.ndarray = None,
                 seed: int = 314, class_weights: np.ndarray = None):

        super().__init__()
        self.seed = seed
        L.seed_everything(self.seed, workers=True)
        self.string_time = time.strftime("%Y%m%d-%H%M%S")
        self.input_size = input_size
        self.output_size = output_size
        self.arch = arch
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device
        self.auto_lr = auto_lr
        self.enable_progress_bar = enable_progress_bar
        self.deterministic = deterministic
        self.class_weights = class_weights
        self.model = DiscriminativeModule(input_size=input_size, output_size=output_size, arch=arch, learning_rate=learning_rate, dropout=dropout, class_weights=class_weights)
        self.is_trained = False
        self.early_stopping = early_stopping
        self.cbs = []
        self.early_stopping_monitor = early_stopping_monitor
        self.early_stopping_dirpath = Path(early_stopping_dirpath)
        self.early_stopping_filename = f'{self.string_time}_{early_stopping_filename}'
        self.early_stopping_save_top_k = early_stopping_save_top_k
        self.early_stopping_mode = early_stopping_mode
        if self.early_stopping and X_valid is not None and y_valid is not None:
            self.early_stopping_dirpath.mkdir(parents=True, exist_ok=True)
            self.cbs.append(L.pytorch.callbacks.ModelCheckpoint(monitor=self.early_stopping_monitor, dirpath=str(self.early_stopping_dirpath), filename=self.early_stopping_filename, save_top_k=self.early_stopping_save_top_k, mode=self.early_stopping_mode))
            self.cbs.append(L.pytorch.callbacks.EarlyStopping(monitor=self.early_stopping_monitor, patience=self.patience, mode='max'))
        
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.has_cuda = torch.cuda.is_available()
        self.has_cuda_bf16 = torch.cuda.is_bf16_supported() if self.has_cuda else False
        
        if self.device in ['gpu', 'cuda'] and self.has_cuda:
            self.accelerator = 'gpu'
            self.precision = 'bf16-mixed' if self.has_cuda_bf16 else '16-mixed'
        else:
            self.accelerator = 'cpu'
            self.precision = 32
    
    def one_hot_encode(self, y: np.ndarray) -> np.ndarray:
        """
        One-hot encode the input labels.

        Args:
            y (np.ndarray): Input labels.

        Returns:
            np.ndarray: One-hot encoded labels.
        """
        if y.ndim == 1:
            self.cardinality = len(np.unique(y))
            return np.eye(self.cardinality)[y]
        else:
            self.cardinality = y.shape[1]
            return y
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, sample_weights: np.ndarray = None) -> 'SimpleDiscriminativeClassifier':
        """
        Fit the model to the input data.

        Args:
            X_train (np.ndarray): The input data.
            y_train (np.ndarray): The target data.
            sample_weights (np.ndarray, optional): The sample weights to use for training. Defaults to None.

        Returns:
            SimpleDiscriminativeClassifier: The fitted classifier.
        """
        if self.batch_size is None:
            self.batch_size = min(32, len(X_train)//100)

        y_probs = self.one_hot_encode(y_train)

        # Add sample weights as a new column to X_train
        if sample_weights is None:
            sample_weights = np.ones(len(X_train))
        X_train = np.hstack((X_train, sample_weights.reshape(-1, 1)))

        # Convert the input data to PyTorch tensors
        train_data = list(zip(X_train, y_probs))
        if self.X_valid is not None and self.y_valid is not None:
            y_valid_probs = self.one_hot_encode(self.y_valid)
            valid_data = list(zip(self.X_valid, y_valid_probs))
        else:
            valid_data = None
        
        # Create the neural network model
        self.data_module = SimpleDataModule(train_data, valid_data, batch_size=self.batch_size)
        self.trainer = L.Trainer(max_epochs=self.max_epochs, devices=1, accelerator=self.accelerator, precision=self.precision, callbacks=self.cbs, deterministic=self.deterministic, enable_progress_bar=self.enable_progress_bar, num_sanity_val_steps=0 if valid_data is None else 2, limit_val_batches=0.0 if valid_data is None else 1.0)

        if self.auto_lr:
            # Use the PyTorch Lightning tuner to find the optimal learning rate
            self.lr_finder = Tuner(self.trainer).lr_find(self.model, self.data_module)
            print(f'Suggested LR: {self.lr_finder.suggestion()}')
        
        # Train the neural network model
        self.trainer.fit(self.model, self.data_module)
        if len(self.cbs) > 0:
            if len(self.cbs[0].best_model_path) > 0:
                self.model = DiscriminativeModule.load_from_checkpoint(self.cbs[0].best_model_path)
                print(f'Loaded best model from {self.cbs[0].best_model_path}')
                try:
                    os.remove(self.cbs[0].best_model_path)
                except:
                    pass
        self.is_trained = True
        return self
    
    def predict_proba(self, X: np.ndarray, batch_size: int = None) -> np.ndarray:
        """
        Get the predicted probabilities for the input data.

        Args:
            X (np.ndarray): The input data.
            batch_size (int, optional): The batch size to use for prediction. Defaults to None.

        Returns:
            np.ndarray: The predicted probabilities.
        """
        if not self.is_trained:
            raise Exception('Model has not been trained yet.')
        
        self.model.eval()
        self.model_device = self.model.device

        if batch_size is None:
            batch_size = self.batch_size

        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # Check if it's an observation or a batch
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        if len(X) < batch_size:
            batch_size = len(X)

        y = np.zeros((len(X), self.cardinality))

        data_test = list(zip(X, y))
        data_module_test = SimpleDataModule(test_data=data_test, batch_size=batch_size)
        data_module_test.setup()

        y_activations = []
        for batch in data_module_test.test_dataloader():
            X, y = batch
            X = X.to(self.model_device)
            y_activations.append(self.model(X).detach().cpu().numpy())

        y_activations = np.vstack(y_activations)
        y_pred_prob = torch.tensor(y_activations).softmax(dim=1).numpy().squeeze()
        return y_pred_prob
    
    def predict(self, X: np.ndarray, batch_size: int = None) -> np.ndarray:
        """
        Get the predicted labels for the input data.

        Args:
            X (np.ndarray): The input data.
            batch_size (int, optional): The batch size to use for prediction. Defaults to None.

        Returns:
            np.ndarray: The predicted labels.
        """
        y_pred_prob = self.predict_proba(X, batch_size=batch_size)
        if y_pred_prob.ndim == 1:
            return y_pred_prob.argmax()
        else:
            return y_pred_prob.argmax(axis=1)

def get_basic_model(X: np.ndarray, y: np.ndarray, batch_size: int = 2048, device: str = 'gpu',
                    learning_rate: float = 3e-4, dropout: float = 0.5, arch: int = 2,
                    auto_lr: bool = False, max_epochs: int = 10) -> SimpleDiscriminativeClassifier:
    """
    Create and return a basic SimpleDiscriminativeClassifier model.

    Args:
        X (np.ndarray): Input features.
        y (np.ndarray): Target labels.
        batch_size (int, optional): Batch size for training. Defaults to 2048.
        device (str, optional): Device to use for training ('cpu' or 'gpu'). Defaults to 'gpu'.
        learning_rate (float, optional): Learning rate for training. Defaults to 3e-4.
        dropout (float, optional): Dropout rate. Defaults to 0.5.
        arch (int, optional): Architecture complexity. Defaults to 2.
        auto_lr (bool, optional): Whether to use automatic learning rate finding. Defaults to False.
        max_epochs (int, optional): Maximum number of epochs for training. Defaults to 10.

    Returns:
        SimpleDiscriminativeClassifier: A configured SimpleDiscriminativeClassifier model.
    """
    if device != 'cpu' and not torch.cuda.is_available():
        print('CUDA is not available. Using CPU.')
        device = 'cpu'
        
    cardinality = helpers.classification.get_cardinality_from_labels(y)

    return SimpleDiscriminativeClassifier(
        input_size=X.shape[1],
        output_size=cardinality,
        batch_size=batch_size,
        device=device,
        learning_rate=learning_rate,
        dropout=dropout,
        arch=arch,
        auto_lr=auto_lr,
        max_epochs=max_epochs,
    )


def train_and_get_preds(
    X_train: np.ndarray,
    y_train_prob: np.ndarray,
    X_valid: np.ndarray,
    y_valid_prob: np.ndarray,
    X_test: np.ndarray = None,
    y_test_prob: np.ndarray = None,
    arch: int = 2,
    learning_rate: float = 3e-4,
    dropout: float = 0.5,
    batch_size: int = 4096,
    max_epochs: int = 100,
    preds_from: str = 'valid',
    patience: int = 10,
    device: str = 'cpu',
    auto_lr: bool = False,
    enable_progress_bar: bool = False,
    deterministic: bool = True,
    class_weights: np.ndarray = None,
    sample_weights: np.ndarray = None
) -> dict[str, str | np.ndarray | float]:
    """
    Trains a discriminative neural network on the input data and returns the predicted labels and probabilities, as well as evaluation metrics.

    Args:
        X_train (np.ndarray): The input data for training.
        y_train_prob (np.ndarray): The target probabilities for training.
        X_valid (np.ndarray): The input data for validation.
        y_valid_prob (np.ndarray): The target probabilities for validation.
        X_test (np.ndarray, optional): The input data for testing. Defaults to None.
        y_test_prob (np.ndarray, optional): The target probabilities for testing. Defaults to None.
        arch (int, optional): The architecture of the neural network to use. Defaults to 2.
        learning_rate (float, optional): The learning rate to use for training. Defaults to 3e-4.
        dropout (float, optional): The dropout rate to use in the network. Defaults to 0.5.
        batch_size (int, optional): The batch size to use for loading data. Defaults to 4096.
        max_epochs (int, optional): The maximum number of epochs to train for. Defaults to 100.
        preds_from (str, optional): Whether to get predictions from the validation or test data. Defaults to 'valid'.
        patience (int, optional): The number of epochs to wait before early stopping. Defaults to 10.
        device (str, optional): The device to use for training. Defaults to 'cpu'.
        auto_lr (bool, optional): Whether to automatically find the optimal learning rate. Defaults to False.
        enable_progress_bar (bool, optional): Whether to enable the progress bar. Defaults to False.
        deterministic (bool, optional): Whether to set the random seed to a fixed value. Defaults to True.
        class_weights (np.ndarray, optional): The class weights to use for training. Defaults to None.
        sample_weights (np.ndarray, optional): Sample weights for training. Defaults to None.

    Returns:
        dict[str, Union[str, np.ndarray, float]]: A dictionary containing the predicted labels and probabilities, as well as evaluation metrics.
    """
    assert preds_from in ['valid', 'test'], 'preds_from must be either "valid" or "test"'
    if X_test is None or y_test_prob is None:
        if preds_from == 'test':
            preds_from = 'valid'
            print('No test data provided. Getting predictions from validation data instead.')
    
    if device == 'cuda':
        device = 'gpu'
    
    # Add sample weights as a new column to X_train
    if sample_weights is None:
        sample_weights = np.ones(len(X_train))
    X_train = np.hstack((X_train, sample_weights.reshape(-1, 1)))

    # Convert the input data to PyTorch tensors
    train_data = list(zip(X_train, y_train_prob))
    valid_data = list(zip(X_valid, y_valid_prob))
    test_data = list(zip(X_test, y_test_prob)) if X_test is not None and y_test_prob is not None else None

    # Create the neural network model
    model = DiscriminativeModule(input_size=X_train.shape[1], output_size=y_train_prob.shape[1], arch=arch, learning_rate=learning_rate, dropout=dropout, class_weights=class_weights)

    # Print the details about all layers of the model
    for name, param in model.named_parameters():
        print(f"{name}: {param.size()}")
    print(model)
        
    # Set up the PyTorch Lightning callbacks
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(monitor='val_mcc', dirpath='tmp/', filename='model-best', save_top_k=1, mode='max')
    early_stop_callback = L.pytorch.callbacks.EarlyStopping(monitor='val_mcc', patience=patience, mode='max')

    # Set up the accelerator and precision based on the device
    accelerator = 'gpu' if device == 'gpu' and torch.cuda.is_available() else 'cpu'
    precision = 'bf16-mixed' if accelerator == 'gpu' and torch.cuda.is_bf16_supported() else ('16-mixed' if accelerator == 'gpu' else 32)
    
    data_module = SimpleDataModule(train_data, valid_data, test_data, batch_size=batch_size)
    trainer = L.Trainer(max_epochs=max_epochs, devices=1, accelerator=accelerator, precision=precision, callbacks=[checkpoint_callback, early_stop_callback], deterministic=deterministic, enable_progress_bar=enable_progress_bar)

    if auto_lr:
        # Use the PyTorch Lightning tuner to find the optimal learning rate
        lr_finder = Tuner(trainer).lr_find(model, data_module)
        print(f'Suggested LR: {lr_finder.suggestion()}')    

    # Train the neural network model
    trainer.fit(model, data_module)
    model = DiscriminativeModule.load_from_checkpoint(checkpoint_callback.best_model_path)

    # Delete the temporary directory created for the best model
    try:
        os.remove(checkpoint_callback.best_model_path)
    except:
        pass

    # Get the predicted labels for the validation or test data
    model.eval()
    model_device = model.device
    y_activations = []
    dataloader = data_module.val_dataloader() if preds_from == 'valid' else data_module.test_dataloader()
    for batch in dataloader:
        X, y = batch
        X = X.to(model_device)
        y_activations.append(model(X).detach().cpu())
    y_activations = torch.cat(y_activations)
    y_pred_prob = y_activations.softmax(dim=1).numpy()
    y_pred_cat = y_pred_prob.argmax(axis=1)

    # Calculate evaluation metrics based on the predicted labels and the true labels
    true_labels = y_valid_prob.argmax(axis=1) if preds_from == 'valid' else y_test_prob.argmax(axis=1)
    mcc = matthews_corrcoef(true_labels, y_pred_cat)
    acc = accuracy_score(true_labels, y_pred_cat)
    cr = classification_report(true_labels, y_pred_cat)
    cm = confusion_matrix(true_labels, y_pred_cat)

    # Return the predicted labels and probabilities, as well as evaluation metrics
    return {
        'preds_from': preds_from,
        'y_pred_cat': y_pred_cat,
        'y_pred_prob': y_pred_prob,
        'accuracy': acc,
        'mcc': mcc,
        'classification_report': cr,
        'confusion_matrix': cm
    }

