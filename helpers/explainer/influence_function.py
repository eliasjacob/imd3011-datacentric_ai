import numpy as np
import torch
import torch.utils.data as tdata
from backpack import backpack
from backpack.extensions import BatchGrad
from torch import nn
from torch.autograd import grad
from tqdm import tqdm, trange


def modify_training_labels(
    predicted_labels: np.ndarray,
    label_matrix: np.ndarray,
    approximated_label_model: np.ndarray,
    influence_scores: np.ndarray,
    sample_ratio: float,
    sample_method: str = 'weight',
    normalize_influence: bool = False,
    activation_function: str = 'identity',
    normalize_labels: bool = False
) -> np.ndarray:
    """
    Modifies the training labels based on the influence function scores. It takes in predicted labels, a labeling function matrix, a weight matrix, influence function scores, and several other parameters, and returns the modified labels. The function can modify the labels in different ways depending on the sample method and activation function used.

    Args:
        predicted_labels (np.ndarray): Predicted labels array.
        label_matrix (np.ndarray): Labeling function matrix.
        approximated_label_model (np.ndarray): Approximated label model.
        influence_scores (np.ndarray): Influence function scores.
        sample_ratio (float): Ratio of samples to modify.
        sample_method (str, optional): Method for selecting samples to modify. Defaults to 'weight'.
        normalize_influence (bool, optional): Whether to normalize influence scores. Defaults to False.
        activation_function (str, optional): Activation function to use ('identity' or 'exp'). Defaults to 'identity'.
        normalize_labels (bool, optional): Whether to normalize the modified labels. Defaults to False.

    Returns:
        np.ndarray: Modified labels.
    """
    if normalize_influence:
        # Flatten influence scores and select top samples based on sample ratio
        influence_scores_flat = influence_scores.flatten()
        num_samples = int(len(influence_scores_flat) * sample_ratio)
        threshold = np.partition(influence_scores_flat, -num_samples)[-num_samples]
        # Modify predicted labels where influence scores are above the threshold
        modified_labels = predicted_labels * (influence_scores >= threshold)
    else:
        if sample_method == 'relabel':
            sample_method = 'term'
            normalize_labels = True

        num_label_functions = label_matrix.shape[1]
        label_function_indices = np.arange(num_label_functions)

        # Compute raw scores using weight matrix and label matrix
        raw_scores = approximated_label_model[label_function_indices, label_matrix + 1, :]

        # Calculate the normalizer based on the activation function
        normalizer = np.sum(raw_scores, axis=1)
        if activation_function == 'exp':
            normalizer = np.exp(normalizer)
        normalizer = np.sum(normalizer, axis=1, keepdims=True)

        if sample_method == 'term':
            # Select top samples based on influence scores
            influence_scores_flat = influence_scores[influence_scores != 0]
            num_samples = int(len(influence_scores_flat) * sample_ratio)
            threshold = np.partition(influence_scores_flat, -num_samples)[-num_samples]
            selected_indices = influence_scores >= threshold
        elif sample_method == 'weight':
            assert activation_function == 'identity'
            weight_to_remove = sample_ratio * len(predicted_labels)
            influence_scores_flat = influence_scores.flatten()
            sorted_indices = np.argsort(-influence_scores_flat)
            normalized_scores = raw_scores / normalizer.reshape(-1, 1, 1)
            cumulative_sum = np.cumsum(normalized_scores.flatten()[sorted_indices])
            unravel_indices = np.unravel_index(
                sorted_indices[cumulative_sum <= weight_to_remove],
                shape=influence_scores.shape
            )
            selected_indices = np.zeros_like(influence_scores)
            selected_indices[unravel_indices] = 1
        elif sample_method == 'LF':
            # For LF, sample size is an integer (the number of LFs to use)
            normalize_labels = False
            lf_influence_scores = influence_scores.sum(axis=(0, 2))
            lf_indices_to_use = (-lf_influence_scores).argsort()[:sample_ratio]
            selected_indices = np.zeros_like(influence_scores)
            selected_indices[:, lf_indices_to_use] = 1
        else:
            raise NotImplementedError("Sample method not implemented.")

        filtered_scores = raw_scores * selected_indices
        modified_labels = np.sum(filtered_scores, axis=1)
        if activation_function == 'exp':
            modified_labels = np.exp(modified_labels)

        if normalize_labels:
            label_sum = np.sum(modified_labels, axis=1, keepdims=True)
            modified_labels = np.divide(modified_labels, label_sum, where=label_sum != 0)
            modified_labels[label_sum.flatten() == 0] = 0
        else:
            modified_labels = modified_labels / normalizer

    return modified_labels


class InfluenceFunction(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        train_dataset: tdata.Dataset,
        valid_dataset: tdata.Dataset,
        num_label_functions: int,
        num_classes: int,
        device: torch.device = None,
        damp: float = 1.0,
        scale: float = 25.0,
        r: int = 2,
        recursion_depth: int = 100
    ):
        """
        Influence Function class that calculates the influence of each training sample on the model's predictions.

        Args:
            model (nn.Module): PyTorch model to calculate influence on.
            train_dataset (Dataset): Training dataset.
            valid_dataset (Dataset): Validation dataset.
            num_label_functions (int): Number of labeling functions.
            num_classes (int): Number of classes.
            device (torch.device, optional): Device to run the calculations on. Defaults to None.
            damp (float, optional): Damping factor. Defaults to 1.0.
            scale (float, optional): Scaling factor. Defaults to 25.0.
            r (int, optional): Rank of the approximation. Defaults to 2.
            recursion_depth (int, optional): Maximum recursion depth. Defaults to 100.
        """
        super().__init__()
        self.model = model
        self.damp = damp
        self.scale = scale
        self.r = r
        self.recursion_depth = recursion_depth
        self.train_dataloader = tdata.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
        self.valid_dataloader = tdata.DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False)
        self.num_train_samples = len(train_dataset)
        self.num_label_functions = num_label_functions
        self.num_classes = num_classes

        self.device = device
        self.eye_matrix = torch.eye(self.num_classes).to(device)
        self.weight_indices = torch.arange(self.num_label_functions)

    def generate_inputs_and_labels(
        self,
        weights: torch.Tensor,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        targets: torch.Tensor,
        activation_function: str = 'identity',
        return_raw_score: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates the input and label tensors for the influence function calculation.

        Args:
            weights (torch.Tensor): Weight tensor.
            inputs (torch.Tensor): Input tensor.
            labels (torch.Tensor): Label indices.
            targets (torch.Tensor): Target labels.
            activation_function (str, optional): Activation function to use ('identity' or 'exp'). Defaults to 'identity'.
            return_raw_score (bool, optional): Whether to return the raw score tensor. Defaults to False.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Expanded input tensor and label tensor.
        """
        raw_scores = weights[self.weight_indices, labels + 1, :]
        if activation_function == 'identity':
            # Repeat inputs for each label function and class
            expanded_inputs = inputs.repeat_interleave(self.num_label_functions * self.num_classes, dim=0)
            normalized_scores = raw_scores / torch.sum(raw_scores, dim=(1, 2), keepdim=True)
            normalized_scores = normalized_scores.view(-1, self.num_classes)
            expanded_labels = (
                normalized_scores.unsqueeze(dim=2)
                .repeat_interleave(self.num_classes, dim=-1) * self.eye_matrix
            )
            expanded_labels = expanded_labels.view(-1, self.num_classes)
            return expanded_inputs, expanded_labels
        elif activation_function == 'exp':
            expanded_inputs = inputs.repeat_interleave(self.num_classes, dim=0)
            expanded_labels = (
                targets.unsqueeze(dim=2)
                .repeat_interleave(self.num_classes, dim=-1) * self.eye_matrix
            )
            expanded_labels = expanded_labels.view(-1, self.num_classes)
            if return_raw_score:
                return expanded_inputs, expanded_labels, raw_scores
            else:
                return expanded_inputs, expanded_labels
        else:
            raise NotImplementedError("Activation function not implemented.")

    def generate_renormalized_inputs_and_labels(
        self,
        weights: torch.Tensor,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        targets: torch.Tensor,
        activation_function: str = 'identity'
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates the input and label tensors for the renormalized influence function calculation.

        Args:
            weights (torch.Tensor): Weight tensor.
            inputs (torch.Tensor): Input tensor.
            labels (torch.Tensor): Label indices.
            targets (torch.Tensor): Target labels.
            activation_function (str, optional): Activation function to use ('identity' or 'exp'). Defaults to 'identity'.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Expanded input tensor and label tensor.
        """
        expanded_inputs = inputs.repeat_interleave(self.num_label_functions * self.num_classes, dim=0)
        raw_scores = weights[self.weight_indices, labels + 1, :]
        expanded_raw_scores = raw_scores.unsqueeze(1).expand(
            -1, self.num_label_functions * self.num_classes, -1, -1
        ).clone()

        # Indices for label functions and classes
        lf_range = torch.arange(self.num_label_functions).repeat_interleave(self.num_classes)
        class_range = torch.arange(self.num_classes).repeat(self.num_label_functions)

        # Zero out specific positions in raw scores
        expanded_raw_scores[:, torch.arange(self.num_label_functions * self.num_classes), lf_range, class_range] = 0
        expanded_raw_scores = torch.sum(expanded_raw_scores, dim=-2)

        if activation_function == 'exp':
            expanded_raw_scores = torch.exp(expanded_raw_scores)

        sum_raw_scores = torch.sum(expanded_raw_scores, dim=-1, keepdim=True)
        normalized_scores = expanded_raw_scores / sum_raw_scores
        normalized_scores[sum_raw_scores.squeeze() == 0] = 0
        normalized_scores = targets.unsqueeze(1) - normalized_scores
        expanded_labels = normalized_scores.view(-1, self.num_classes)

        return expanded_inputs, expanded_labels

    def outer_product_inverse(self, input_vector: torch.Tensor) -> torch.Tensor:
        """
        Calculates the inverse of the outer product of the input vector.

        Args:
            input_vector (torch.Tensor): Input vector.

        Returns:
            torch.Tensor: Inverse of the outer product of the input vector.
        """
        outer_result = input_vector @ input_vector.t()
        inverse_result = torch.inverse(
            outer_result.cpu() + self.damp * torch.eye(outer_result.shape[0])
        ).to(input_vector.device)
        return inverse_result

    def batch_hessian_vector_product(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        params_list: list[torch.Tensor],
        batch_grad_list: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """
        Calculates the batch Hessian-vector product.

        Args:
            inputs (torch.Tensor): Input tensor.
            labels (torch.Tensor): Label tensor.
            params_list (list[torch.Tensor]): list of model parameters.
            batch_grad_list (list[torch.Tensor]): list of batch gradients.

        Returns:
            list[torch.Tensor]: list of second-order gradients.
        """
        logits = self.model(inputs.to(self.device))
        loss = self.model.ce_loss(logits, labels)
        if len(params_list) != len(batch_grad_list):
            raise ValueError("Parameter list and gradient list must have the same length.")

        one_sample_grad_list = list(grad(loss, params_list, create_graph=True, retain_graph=True))
        elemwise_products = sum(
            (grad_elem.view(-1) * v_elem).sum() for grad_elem, v_elem in zip(one_sample_grad_list, batch_grad_list)
        )
        second_order_grads = list(grad(elemwise_products, params_list))
        return [p.clone().detach().view(-1) for p in second_order_grads]

    def batch_s_test(
        self,
        batch_v: list[torch.Tensor],
        batch_h_estimate: list[torch.Tensor],
        mode: str,
        weights: torch.Tensor,
        activation_function: str
    ) -> torch.Tensor:
        """
        Calculates the batch S-test vector.

        Args:
            batch_v (list[torch.Tensor]): list of validation gradients.
            batch_h_estimate (list[torch.Tensor]): list of batch Hessian estimates.
            mode (str): Mode of the influence function calculation ('normal', 'RW', or 'WM').
            weights (torch.Tensor): Weight tensor.
            activation_function (str): Activation function.

        Returns:
            torch.Tensor: Batch S-test vector.
        """
        for _ in range(self.recursion_depth):
            if mode == 'normal':
                for _, inputs, _, targets in self.train_dataloader:
                    self.model.zero_grad()
                    params = [p for p in self.model.parameters() if p.requires_grad]
                    batch_hv = self.batch_hessian_vector_product(inputs, targets, params, batch_h_estimate)
                    batch_temp_list = [
                        v + (1 - self.damp) * h_e - hv / self.scale
                        for v, h_e, hv in zip(batch_v, batch_h_estimate, batch_hv)
                    ]
                    batch_h_estimate = [item.clone() for item in batch_temp_list]
                    break  # Only process one batch
            else:
                for _, inputs, labels, targets in self.train_dataloader:
                    self.model.zero_grad()
                    params = [p for p in self.model.parameters() if p.requires_grad]
                    if mode == 'RW':
                        ex_inputs, ex_labels = self.generate_inputs_and_labels(
                            weights, inputs, labels, targets, activation_function
                        )
                    elif mode == 'WM':
                        ex_inputs, ex_labels = self.generate_renormalized_inputs_and_labels(
                            weights, inputs, labels, targets, activation_function
                        )
                    else:
                        raise NotImplementedError("Mode not implemented.")

                    batch_hv = self.batch_hessian_vector_product(ex_inputs, ex_labels, params, batch_h_estimate)
                    batch_temp_list = [
                        v + (1 - self.damp) * h_e - hv / self.scale
                        for v, h_e, hv in zip(batch_v, batch_h_estimate, batch_hv)
                    ]
                    batch_h_estimate = [item.clone() for item in batch_temp_list]
                    break  # Only process one batch

        h_estimate_vec = torch.cat(batch_h_estimate, dim=0)
        return h_estimate_vec

    def compute_hv(
        self,
        val_grad_list: list[torch.Tensor],
        mode: str,
        weights: torch.Tensor,
        activation_function: str
    ) -> torch.Tensor:
        """
        Calculates the Hessian-vector product. The Hessian-vector product is an approximation of the Hessian of the loss function with respect to the model parameters. The Hessian is a square matrix of second-order partial derivatives of the loss function with respect to the model parameters. It describes the curvature of the loss function.

        Args:
            val_grad_list (list[torch.Tensor]): list of validation gradients.
            mode (str): Mode of the influence function calculation.
            weights (torch.Tensor): Weight tensor.
            activation_function (str): Activation function.

        Returns:
            torch.Tensor: S-test vector.
        """
        s_test_vec_list = []
        for _ in range(self.r):
            batch_v = val_grad_list
            batch_h_estimate = [h.clone().detach() for h in batch_v]
            s_test_vec = self.batch_s_test(batch_v, batch_h_estimate, mode, weights, activation_function)
            s_test_vec_list.append(s_test_vec.unsqueeze(0))
        s_test_vec = torch.cat(s_test_vec_list, dim=0).mean(dim=0)
        return s_test_vec

    def compute_valid_grad_and_hv(
        self,
        mode: str,
        weights: torch.Tensor,
        activation_function: str,
        batch_mode: bool
    ) -> torch.Tensor:
        """
        Computes the gradient over the validation set and the Hessian-vector product.

        Args:
            mode (str): Mode of the influence function calculation.
            weights (torch.Tensor): Weight tensor.
            activation_function (str): Activation function.
            batch_mode (bool): Whether to use batch mode.

        Returns:
            torch.Tensor: Hessian-vector product.
        """
        print('Computing gradient over validation set')
        self.model.zero_grad()

        if batch_mode:
            for _, inputs, targets in tqdm(self.valid_dataloader):
                logits = self.model(inputs)
                loss = self.model.ce_loss_sum(logits, targets)
                with backpack(BatchGrad()):
                    loss.backward()
                    batch_train_grad_vec = self.model.collect_batch_grad().detach()
                val_grad_list = self.model.separate_batch_grad(batch_train_grad_vec)
        else:
            for _, inputs, targets in tqdm(self.valid_dataloader):
                logits = self.model(inputs)
                loss = self.model.ce_loss_sum(logits, targets)
                loss.backward()
            val_grad_vec = self.model.collect_grad()
            val_grad_list = self.model.separate_batch_grad(val_grad_vec)

        self.model.zero_grad()

        print('Computing Hessian-vector product')
        if batch_mode:
            s_test_vec_list = []
            for i in trange(len(val_grad_list[0])):
                s_test_vec = self.compute_hv([v[i] for v in val_grad_list], mode, weights, activation_function)
                s_test_vec_list.append(s_test_vec)
            s_test_vec = torch.stack(s_test_vec_list).T
        else:
            s_test_vec = self.compute_hv(val_grad_list, mode, weights, activation_function)

        return s_test_vec

    def compute_influence_function(
        self,
        if_type: str,
        mode: str = 'normal',
        weights: torch.Tensor = None,
        activation_function: str = 'identity',
        batch_mode: bool = False
    ) -> np.ndarray:
        """
        Computes the influence function.

        Args:
            if_type (str): Type of influence function ('if', 'sif', 'relatif', or 'all').
            mode (str, optional): Mode of the influence function calculation. Defaults to 'normal'.
            weights (torch.Tensor, optional): Weight tensor. Defaults to None.
            activation_function (str, optional): Activation function. Defaults to 'identity'.
            batch_mode (bool, optional): Whether to use batch mode. Defaults to False.

        Returns:
            np.ndarray: Calculated influence function.
        """
        assert activation_function in ['identity', 'exp']
        assert mode in ['normal', 'RW', 'WM']

        if if_type == 'if':
            influence = self.compute_original_if(mode, weights, activation_function, batch_mode)
        elif if_type == 'sif':
            influence = self.compute_self_if(mode, weights, activation_function, batch_mode)
        elif if_type == 'relatif':
            influence = self.compute_relative_if(mode, weights, activation_function, batch_mode, return_all=False)
        elif if_type == 'all':
            influence = self.compute_relative_if(mode, weights, activation_function, batch_mode, return_all=True)
        else:
            raise NotImplementedError("Influence function type not implemented.")

        return influence

    def compute_original_if(
        self,
        mode: str,
        weights: torch.Tensor = None,
        activation_function: str = 'identity',
        batch_mode: bool = False
    ) -> np.ndarray:
        """
        Computes the original influence function.

        Args:
            mode (str): Mode of the influence function calculation.
            weights (torch.Tensor, optional): Weight tensor. Defaults to None.
            activation_function (str, optional): Activation function. Defaults to 'identity'.
            batch_mode (bool, optional): Whether to use batch mode. Defaults to False.

        Returns:
            np.ndarray: Original influence function.
        """
        s_test_vec = self.compute_valid_grad_and_hv(mode, weights, activation_function, batch_mode)

        print('Computing influence over training set')
        if mode == 'normal':
            if batch_mode:
                n_test = s_test_vec.shape[1]
                train_if = torch.zeros(self.num_train_samples, n_test).to(self.device)
            else:
                train_if = torch.zeros(self.num_train_samples, 1).to(self.device)
                s_test_vec = s_test_vec.view(-1, 1)
            for idx, inputs, _, targets in tqdm(self.train_dataloader):
                self.model.zero_grad()
                logits = self.model(inputs)
                loss = self.model.ce_loss_sum(logits, targets)

                with backpack(BatchGrad()):
                    loss.backward()
                    batch_train_grad_vec = self.model.collect_batch_grad().detach()

                if_score = batch_train_grad_vec @ s_test_vec
                train_if[idx] = if_score.detach()
        else:
            if batch_mode:
                n_test = s_test_vec.shape[1]
                train_if = torch.zeros(
                    self.num_train_samples, self.num_label_functions, self.num_classes, n_test
                ).to(self.device)
            else:
                train_if = torch.zeros(
                    self.num_train_samples, self.num_label_functions, self.num_classes
                ).to(self.device)

            if mode == 'RW':
                if activation_function == 'identity':
                    for idx, inputs, labels, targets in tqdm(self.train_dataloader):
                        self.model.zero_grad()
                        ex_inputs, ex_labels = self.generate_inputs_and_labels(
                            weights, inputs, labels, targets, activation_function
                        )
                        logits = self.model(ex_inputs)
                        loss = self.model.ce_loss_sum(logits, ex_labels)

                        with backpack(BatchGrad()):
                            loss.backward()
                            batch_train_grad_vec = self.model.collect_batch_grad().detach()

                        if_score = batch_train_grad_vec.view(
                            -1, self.num_label_functions, self.num_classes, len(s_test_vec)
                        ) @ s_test_vec
                        train_if[idx] = if_score.detach()

                elif activation_function == 'exp':
                    for idx, inputs, labels, targets in tqdm(self.train_dataloader):
                        self.model.zero_grad()
                        ex_inputs, ex_labels, raw_scores = self.generate_inputs_and_labels(
                            weights, inputs, labels, targets, activation_function, return_raw_score=True
                        )
                        logits = self.model(ex_inputs)
                        loss = self.model.ce_loss_sum(logits, ex_labels)

                        with backpack(BatchGrad()):
                            loss.backward()
                            batch_train_grad_vec = self.model.collect_batch_grad().detach()

                        if batch_mode:
                            if_score = (
                                batch_train_grad_vec @ s_test_vec
                            ).view(-1, self.num_classes, n_test).unsqueeze(1) * raw_scores.unsqueeze(-1)
                        else:
                            if_score = (
                                batch_train_grad_vec @ s_test_vec
                            ).view(-1, self.num_classes).unsqueeze(1) * raw_scores
                        train_if[idx] = if_score.detach()
                else:
                    raise NotImplementedError("Activation function not implemented.")

            elif mode == 'WM':
                for idx, inputs, labels, targets in tqdm(self.train_dataloader):
                    self.model.zero_grad()
                    ex_inputs, ex_labels = self.generate_renormalized_inputs_and_labels(
                        weights, inputs, labels, targets, activation_function
                    )
                    logits = self.model(ex_inputs)
                    loss = self.model.ce_loss_sum(logits, ex_labels)

                    with backpack(BatchGrad()):
                        loss.backward()
                        batch_train_grad_vec = self.model.collect_batch_grad().detach()
                    batch_train_grad_vec = batch_train_grad_vec.view(
                        len(inputs), self.num_label_functions * self.num_classes, -1
                    )
                    if_score = batch_train_grad_vec.view(
                        -1, self.num_label_functions, self.num_classes, len(s_test_vec)
                    ) @ s_test_vec
                    train_if[idx] = if_score.detach()
            else:
                raise NotImplementedError("Mode not implemented.")

        train_if = train_if / self.num_train_samples
        return train_if.cpu().numpy()

    def compute_self_if(
        self,
        mode: str,
        weights: torch.Tensor = None,
        activation_function: str = 'identity',
        batch_mode: bool = False
    ) -> np.ndarray:
        """
        Computes the self-influence of the model on the training data.
        From the paper:
        "For the computation of self-influence, the influence estimation is required for each training sample. For real-world dataset, estimating each sample separately using the LiSSA method is intolerable. Instead of applying the stochastic method for each training sample, we leverage the relation between Hessian matrix and Fisher matrix, and use the K-FAC method [27] directly compute the inverse Hessian matrix. We refer interested readers to Barshan et al. [3] for details regarding the K-FAC approximation for the computation of inverse Hessian."
        
        Args:
            mode (str): Mode of the influence function calculation.
            weights (torch.Tensor, optional): Weight tensor. Defaults to None.
            activation_function (str, optional): Activation function. Defaults to 'identity'.
            batch_mode (bool, optional): Whether to use batch mode. Defaults to False.

        Returns:
            np.ndarray: Self-influence scores.
        """
        # Compute inverse Fisher information matrix
        num_layers = self.model.num_of_layers()
        inverse_block_diag = [0 for _ in range(num_layers)]

        if mode == 'normal':
            for _, inputs, _, targets in self.train_dataloader:
                logits = self.model(inputs)
                for i in tqdm(range(inputs.shape[0])):
                    self.model.zero_grad()
                    loss = self.model.ce_loss_sum(logits[i].unsqueeze(0), targets[i].unsqueeze(0))
                    loss.backward(retain_graph=True)
                    train_grad = self.model.collect_grad()
                    train_grad_list = self.model.separate_batch_grad(train_grad)
                    for j in range(num_layers):
                        inverse_block_diag[j] += self.outer_product_inverse(train_grad_list[j].view(-1, 1))
                for j in range(num_layers):
                    inverse_block_diag[j] /= inputs.shape[0]
                break  # Only process one batch
        else:
            for _, inputs, labels, targets in self.train_dataloader:
                ex_inputs, ex_labels = self.generate_inputs_and_labels(
                    weights, inputs, labels, targets, activation_function
                )
                logits = self.model(ex_inputs)
                count = 0
                for i in tqdm(range(ex_inputs.shape[0])):
                    if ex_labels[i].sum() != 0:
                        count += 1
                        self.model.zero_grad()
                        loss = self.model.ce_loss_sum(logits[i].unsqueeze(0), ex_labels[i].unsqueeze(0))
                        loss.backward(retain_graph=True)
                        train_grad = self.model.collect_grad()
                        train_grad_list = self.model.separate_batch_grad(train_grad)
                        for j in range(num_layers):
                            inverse_block_diag[j] += self.outer_product_inverse(train_grad_list[j].view(-1, 1))
                for j in range(num_layers):
                    inverse_block_diag[j] /= count
                break  # Only process one batch

        if mode == 'normal':
            train_sif = torch.zeros(self.num_train_samples, 1).to(self.device)
            for idx, inputs, _, targets in tqdm(self.train_dataloader):
                self.model.zero_grad()
                logits = self.model(inputs)
                loss = self.model.ce_loss_sum(logits, targets)
                with backpack(BatchGrad()):
                    loss.backward()
                    batch_train_grad_vec = self.model.collect_batch_grad().detach()
                    train_batch_grad_list = self.model.separate_batch_grad(batch_train_grad_vec)
                    temp = sum(
                        (torch.mm(grad, inv_block) * grad).sum(dim=-1)
                        for grad, inv_block in zip(train_batch_grad_list, inverse_block_diag)
                    )
                train_sif[idx] = temp.view(-1, 1)
        else:
            train_sif = torch.zeros(
                self.num_train_samples, self.num_label_functions, self.num_classes
            ).to(self.device)

            if activation_function == 'identity':
                for idx, inputs, labels, targets in tqdm(self.train_dataloader):
                    self.model.zero_grad()
                    ex_inputs, ex_labels = self.generate_inputs_and_labels(
                        weights, inputs, labels, targets, activation_function
                    )
                    logits = self.model(ex_inputs)
                    loss = self.model.ce_loss_sum(logits, ex_labels)
                    with backpack(BatchGrad()):
                        loss.backward()
                        batch_train_grad_vec = self.model.collect_batch_grad().detach()
                        train_batch_grad_list = self.model.separate_batch_grad(batch_train_grad_vec)
                        temp = sum(
                            (torch.mm(grad, inv_block) * grad).sum(dim=-1)
                            for grad, inv_block in zip(train_batch_grad_list, inverse_block_diag)
                        )
                    train_sif[idx] = temp.view(-1, self.num_label_functions, self.num_classes)
            elif activation_function == 'exp':
                for idx, inputs, labels, targets in tqdm(self.train_dataloader):
                    self.model.zero_grad()
                    ex_inputs, ex_labels, raw_scores = self.generate_inputs_and_labels(
                        weights, inputs, labels, targets, activation_function, return_raw_score=True
                    )
                    logits = self.model(ex_inputs)
                    loss = self.model.ce_loss_sum(logits, ex_labels)
                    with backpack(BatchGrad()):
                        loss.backward(retain_graph=True)
                        batch_train_grad_vec = self.model.collect_batch_grad().detach()
                        train_batch_grad_list = self.model.separate_batch_grad(batch_train_grad_vec)
                        hv = [torch.mm(grad, inv_block) for grad, inv_block in zip(train_batch_grad_list, inverse_block_diag)]
                    hv = torch.cat(hv, dim=1)
                    raw_scores += 1 / self.num_label_functions
                    ex_inputs = inputs.repeat_interleave(self.num_label_functions, dim=0)
                    self.model.zero_grad()
                    logits = self.model(ex_inputs)
                    finegrained_loss = self.model.ce_loss_sum(logits, raw_scores.view(-1, self.num_classes))
                    with backpack(BatchGrad()):
                        finegrained_loss.backward(retain_graph=True)
                        batch_train_grad_vec = self.model.collect_batch_grad().detach().view(
                            len(inputs), self.num_label_functions, -1
                        )
                    hv = hv.view(len(inputs), self.num_classes, -1)
                    sif = (batch_train_grad_vec @ hv.transpose(2, 1)) * raw_scores
                    train_sif[idx] = sif
            else:
                raise NotImplementedError("Activation function not implemented.")

            if batch_mode:
                train_sif = train_sif.unsqueeze(-1)

        train_sif = train_sif.cpu().numpy()
        if np.any(train_sif < 0):
            train_sif -= np.min(train_sif)
        train_sif = np.sqrt(train_sif)
        return train_sif

    def compute_relative_if(
        self,
        mode: str,
        weights: torch.Tensor = None,
        activation_function: str = 'identity',
        batch_mode: bool = False,
        return_all: bool = False
    ) -> np.ndarray:
        """
        Computes the relative influence of the model on the training data.

        Args:
            mode (str): Mode of the influence function calculation.
            weights (torch.Tensor, optional): Weight tensor. Defaults to None.
            activation_function (str, optional): Activation function. Defaults to 'identity'.
            batch_mode (bool, optional): Whether to use batch mode. Defaults to False.
            return_all (bool, optional): Whether to return all components. Defaults to False.

        Returns:
            np.ndarray: Relative influence scores.
        """
        train_if = self.compute_original_if(mode, weights, activation_function, batch_mode)
        train_sif = self.compute_self_if(mode, weights, activation_function, batch_mode)
        train_rel_if = np.divide(train_if, train_sif, where=train_sif != 0)
        train_rel_if[train_sif == 0] = 0

        if return_all:
            return train_if, train_rel_if, train_sif
        else:
            return train_rel_if
