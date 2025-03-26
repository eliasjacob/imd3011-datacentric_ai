import copy
import random
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from plotly.subplots import make_subplots
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


# PyTorch neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_dimension: int, hidden_layer1_size: int, hidden_layer2_size: int, hidden_layer3_size: int, output_size: int, dropout_rate: float):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dimension, hidden_layer1_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_layer1_size, hidden_layer2_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_layer2_size, hidden_layer3_size)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(hidden_layer3_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


def train_and_evaluate_classification_models(X: pd.DataFrame, y: pd.Series, plot_calibration: bool = True) -> tuple[pd.DataFrame, list[tuple[str, str, np.ndarray]], go.Figure | None]:
    """
    Train multiple classification models and evaluate their performance.

    Args:
        X (pd.DataFrame): The training data.
        y (pd.Series): The training labels.
        plot_calibration (bool): Whether to generate calibration plots. Default is True.

    Returns:
        tuple[pd.DataFrame, list[tuple[str, str, np.ndarray]], Optional[go.Figure]]: 
            A DataFrame with the performance metrics of each model, 
            a list of classification reports, and 
            a figure with consolidated calibration plots (if plot_calibration is True).
    """
    random_state = 271828
    
    # Define the models to be trained
    models = [
        ('Calibrated-LSVC', CalibratedClassifierCV(LinearSVC(random_state=random_state, class_weight='balanced', dual='auto'))),
        ('Logistic Regression', LogisticRegression(random_state=random_state, n_jobs=-1, class_weight='balanced')),
        ('Random Forest', RandomForestClassifier(random_state=random_state, n_jobs=-1, class_weight='balanced')),
        ('XGBoost', XGBClassifier(random_state=random_state, n_jobs=-1, class_weight='balanced', verbosity=0)),
        ('SGD', SGDClassifier(random_state=random_state, n_jobs=-1, class_weight='balanced')),
        ('Naive Bayes', MultinomialNB()),
        ('K-Nearest Neighbors', KNeighborsClassifier(n_jobs=-1)),
        ('Decision Tree', DecisionTreeClassifier(random_state=random_state, class_weight='balanced')),
        ('Extra Trees', ExtraTreesClassifier(random_state=random_state, n_jobs=-1, class_weight='balanced'))
    ]
    
    performance_results = []
    classification_reports = []
    
    # Create a Plotly figure for calibration plots
    calibration_fig = None
    if plot_calibration:
        calibration_fig = make_subplots(
            rows=2, 
            cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Calibration Curves for All Classifiers', 'Probability Distributions'),
            vertical_spacing=0.15
        )
        
        # Add the reference line for perfect calibration
        calibration_fig.add_trace(
            go.Scatter(
                x=[0, 1], 
                y=[0, 1], 
                mode='lines', 
                name='Perfectly calibrated',
                line=dict(dash='dash', color='black', width=1),
            ),
            row=1, col=1
        )
    
    # StratifiedKFold cross-validator to ensure each fold has the same proportion of classes
    cross_validation = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    colors = px.colors.qualitative.Plotly
    
    for i, (model_name, model) in enumerate(models):
        start_time = time.time()
        model_color = colors[i % len(colors)]
        try:
            predicted_labels = cross_val_predict(estimator=model, X=X, y=y, cv=cross_validation, method="predict", n_jobs=2)
            if plot_calibration:
                try:
                    y_prob = cross_val_predict(estimator=model, X=X, y=y, cv=cross_validation, method="predict_proba", n_jobs=2)
                    if len(np.unique(y)) > 2:  # Multiclass
                        n_classes = y_prob.shape[1]
                        class_counts = np.bincount(np.asarray(y))
                        main_class = np.argmax(class_counts) 
                        y_binary = (np.array(y) == main_class).astype(int)
                        prob_pos = y_prob[:, main_class]
                        from sklearn.calibration import calibration_curve
                        fraction_of_positives, mean_predicted_value = calibration_curve(y_binary, prob_pos, n_bins=10)
                        calibration_fig.add_trace(
                            go.Scatter(
                                x=mean_predicted_value, 
                                y=fraction_of_positives, 
                                mode='lines+markers', 
                                name=f"{model_name}",
                                line=dict(color=model_color),
                                marker=dict(size=8)
                            ),
                            row=1, col=1
                        )
                        calibration_fig.add_trace(
                            go.Histogram(
                                x=prob_pos,
                                name=f"{model_name}",
                                opacity=0.6,
                                marker_color=model_color,
                                xbins=dict(start=0, end=1, size=0.1),
                                showlegend=False
                            ),
                            row=2, col=1
                        )
                    else:  # Binary classification
                        prob_pos = y_prob[:, 1]
                        from sklearn.calibration import calibration_curve
                        fraction_of_positives, mean_predicted_value = calibration_curve(y, prob_pos, n_bins=10)
                        calibration_fig.add_trace(
                            go.Scatter(
                                x=mean_predicted_value, 
                                y=fraction_of_positives, 
                                mode='lines+markers', 
                                name=f"{model_name}",
                                line=dict(color=model_color),
                                marker=dict(size=8)
                            ),
                            row=1, col=1
                        )
                        calibration_fig.add_trace(
                            go.Histogram(
                                x=prob_pos,
                                name=f"{model_name}",
                                opacity=0.6,
                                marker_color=model_color,
                                xbins=dict(start=0, end=1, size=0.1),
                                showlegend=False
                            ),
                            row=2, col=1
                        )
                except Exception as e:
                    print(f'Could not generate calibration plot for {model_name}: {e}')
        except Exception as e:
            print(f'Error {model_name} - {e}')
            continue 
        
        f1 = f1_score(y, predicted_labels, average='micro')
        balanced_accuracy = balanced_accuracy_score(y, predicted_labels)
        accuracy = accuracy_score(y, predicted_labels)
        classification_report_str = classification_report(y, predicted_labels)
        matthews_corr_coeff = matthews_corrcoef(y, predicted_labels)
        confusion_mat = confusion_matrix(y, predicted_labels)
        classification_reports.append((model_name, classification_report_str, confusion_mat))
        elapsed_time = time.time() - start_time
        performance_results.append([
            model_name, f1, balanced_accuracy, accuracy, matthews_corr_coeff, elapsed_time, confusion_mat, classification_report_str
        ])
        print(f'Model: {model_name} - F1: {f1:.4f} - Balanced Accuracy: {balanced_accuracy:.4f} - Accuracy: {accuracy:.4f} - Matthews Correlation Coefficient: {matthews_corr_coeff:.4f} - Elapsed time: {elapsed_time:.2f}s')
        print(classification_report_str)
        print(confusion_mat)
        print('*' * 20, '\n')
    
    if plot_calibration and calibration_fig is not None:
        calibration_fig.update_layout(
            title='Model Calibration Comparison',
            title_x=0.5,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            height=800,
            template='plotly_white'
        )
        calibration_fig.update_xaxes(title_text="Mean predicted probability", row=1, col=1)
        calibration_fig.update_yaxes(title_text="Fraction of positives", row=1, col=1)
        calibration_fig.update_xaxes(title_text="Predicted probability", row=2, col=1)
        calibration_fig.update_yaxes(title_text="Count", row=2, col=1)
        calibration_fig.update_yaxes(range=[-0.05, 1.05], row=1, col=1)
    
    results_df = pd.DataFrame(performance_results, columns=[
        'Model', 'F1', 'Balanced Accuracy', 'Accuracy', 'Matthews Correlation Coefficient', 'Elapsed Time', 'Confusion Matrix', 'Classification Report'
    ])
    results_df['Confusion Matrix'] = results_df['Confusion Matrix'].apply(lambda x: str(x))
    return results_df, classification_reports, calibration_fig


def train_neural_network(X_train: np.ndarray, y_train: np.ndarray, X_dev: np.ndarray, y_dev: np.ndarray, verbose: bool = True, random_state: int = 271828) -> NeuralNetwork:
    """
    Train a neural network model using PyTorch with early stopping.

    Args:
        X_train (ndarray): The training data.
        y_train (ndarray): The training labels (one-hot encoded).
        X_dev (ndarray): The development data.
        y_dev (ndarray): The development labels (one-hot encoded).
        verbose (int): Verbosity mode (0 = silent, 1 = progress bar).

    Returns:
        NeuralNetwork: The trained PyTorch model.
    """
    
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)

    # Check if y_train and y_dev are one-hot encoded
    if y_train.ndim == 1 or y_train.shape[1] == 1:
        num_classes = int(np.max(y_train) + 1)
        y_train = np.eye(num_classes)[y_train.astype(int)]
    
    if y_dev.ndim == 1 or y_dev.shape[1] == 1:
        num_classes = int(np.max(y_dev) + 1)
        y_dev = np.eye(num_classes)[y_dev.astype(int)]

    input_dimension = X_train.shape[1]
    hidden_layer1_size = 384
    hidden_layer2_size = 192
    hidden_layer3_size = 96
    output_size = y_train.shape[1]
    dropout_rate = 0.1
    learning_rate = 0.0003
    num_epochs = 20
    batch_size = 256

    model = NeuralNetwork(input_dimension, hidden_layer1_size, hidden_layer2_size, hidden_layer3_size, output_size, dropout_rate)
    device = torch.device("cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.KLDivLoss(reduction='batchmean')

    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_dev_tensor = torch.from_numpy(X_dev).float()
    y_dev_tensor = torch.from_numpy(y_dev).float()

    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_mcc = -1.0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        epoch_loss /= len(dataloader.dataset)

        # Validation phase
        model.eval()
        with torch.no_grad():
            dev_outputs = model(X_dev_tensor.to(device))
            dev_loss = criterion(dev_outputs, y_dev_tensor.to(device)).item()
            
            # Calculate validation mcc
            _, predicted = torch.max(dev_outputs, 1)
            _, true_labels = torch.max(y_dev_tensor, 1)
            mcc = matthews_corrcoef(true_labels.cpu().numpy(), predicted.cpu().numpy())

        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f} / Validation Loss: {dev_loss:.4f} / Validation MCC: {mcc:.4f}")

        if mcc > best_mcc:
            best_mcc = mcc
            best_model_state = copy.deepcopy(model.state_dict())

    if best_model_state is not None:
        if verbose:
            print(f'Loading the best model with MCC: {best_mcc:.4f}')
        model.load_state_dict(best_model_state)
    return model


def print_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, return_metrics: bool = False) -> None:
    """
    Display various classification metrics in a formatted manner.

    Args:
        y_true (np.ndarray): True labels, can be 1D or 2D array.
        y_pred (np.ndarray): Predicted labels, can be 1D or 2D array.

    Returns:
        None
    """
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    print(f"{'Metric':<35} {'Score':>10}")
    print("=" * 45)
    print(f"{'Accuracy Score:':<35} {accuracy_score(y_true, y_pred):>10.5f}")
    print(f"{'Balanced Accuracy Score:':<35} {balanced_accuracy_score(y_true, y_pred):>10.5f}")
    print(f"{'F1 Score (weighted):':<35} {f1_score(y_true, y_pred, average='weighted'):>10.5f}")
    print(f"{'Cohen Kappa Score:':<35} {cohen_kappa_score(y_true, y_pred):>10.5f}")
    print(f"{'Matthews Correlation Coefficient:':<35} {matthews_corrcoef(y_true, y_pred):>10.5f}")
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))
    print("\nConfusion Matrix:\n")
    for c in np.unique(y_true):
        print(f"Class {c} has {np.sum((y_true == c) & (y_pred != c))} false negatives and {np.sum((y_true != c) & (y_pred == c))} false positives.")
    print(f'The total number of errors is {np.sum(y_true != y_pred)} out of {len(y_true)} samples (error rate: {np.sum(y_true != y_pred) / len(y_true):.4f}).')
    print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True).map("{:,}".format))

    if return_metrics:
        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
            'F1 Score (weighted)': f1_score(y_true, y_pred, average='weighted'),
            'Cohen Kappa Score': cohen_kappa_score(y_true, y_pred),
            'Matthews Correlation Coefficient': matthews_corrcoef(y_true, y_pred)
        }
    

def get_cardinality_from_labels(labels: np.ndarray) -> int:
    """
    Get the cardinality (number of unique classes) from the labels.

    Args:
        labels (np.ndarray): A numpy array representing the labels.

    Returns:
        int: An integer representing the cardinality.
    """
    decoded_labels = one_hot_decode(labels)
    return len(np.unique(decoded_labels))


def one_hot_encode(labels: np.ndarray) -> np.ndarray:
    """
    One-hot encode the labels.

    Args:
        labels (np.ndarray): A numpy array representing the labels.

    Returns:
        np.ndarray: A numpy array representing the one-hot encoded labels.
    """
    if labels.ndim == 1:
        num_classes = len(np.unique(labels))
        one_hot_labels = np.eye(num_classes)[labels]
        return one_hot_labels
    else:
        return labels


def one_hot_decode(one_hot_labels: np.ndarray) -> np.ndarray:
    """
    Decode the one-hot encoded labels to their original form.

    Args:
        one_hot_labels (np.ndarray): A numpy array representing the one-hot encoded labels.

    Returns:
        np.ndarray: A numpy array representing the decoded labels.
    """
    if one_hot_labels.ndim == 1:
        return one_hot_labels
    else:
        decoded_labels = np.argmax(one_hot_labels, axis=1)
        return decoded_labels

def predict_pytorch(model: NeuralNetwork, X: np.ndarray) -> np.ndarray:
    """
    Generate predictions (probabilities) using the trained PyTorch model.

    Args:
        model (NeuralNetwork): The trained PyTorch model.
        X (np.ndarray): Input features.

    Returns:
        np.ndarray: Array of predicted probabilities.
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).float()
        log_probs = model(X_tensor)
        probs = torch.exp(log_probs)
    return probs.cpu().numpy()
