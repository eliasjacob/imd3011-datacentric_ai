from typing import Callable, Iterable

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs, parallel_backend
from joblib.parallel import effective_n_jobs
from lightgbm import early_stopping
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from tqdm import tqdm


class SelfTrainingLeaner(BaseEstimator, ClassifierMixin):
    """
    A semi-supervised learning classifier that uses self-training to leverage unlabeled data.

    This classifier iteratively labels high-confidence predictions from unlabeled data
    and incorporates them into the training set to improve the model.

    Args:
        base_classifier (BaseEstimator): The base classifier to use for predictions.
        confidence_threshold (float): The minimum confidence level to accept a prediction.
        max_iterations (int): The maximum number of self-training iterations.

    Attributes:
        classes_ (np.ndarray): The classes seen during fitting.
        final_classifier_ (BaseEstimator): The final trained classifier.
    """

    def __init__(self, base_classifier: BaseEstimator, confidence_threshold: float = 0.7, max_iterations: int = 10):
        self.base_classifier = base_classifier
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations

    def fit(self, X_labeled: np.ndarray, y_labeled: np.ndarray, X_unlabeled: np.ndarray) -> 'SelfTrainingLeaner':
        """
        Fit the semi-supervised learner using labeled and unlabeled data.

        Args:
            X_labeled (np.ndarray): The input samples for labeled data.
            y_labeled (np.ndarray): The target values for labeled data.
            X_unlabeled (np.ndarray): The input samples for unlabeled data.

        Returns:
            self: The fitted SelfTrainingLeaner instance.

        Raises:
            ValueError: If input data is not valid.
        """
        # Validate input data
        X_labeled, y_labeled = check_X_y(X_labeled, y_labeled)
        X_unlabeled = check_array(X_unlabeled)

        # Store the classes seen during fit
        self.classes_ = np.unique(y_labeled)

        # Initialize the working sets
        X_train, y_train = X_labeled.copy(), y_labeled.copy()
        unlabeled_pool = X_unlabeled.copy()

        for iteration in range(self.max_iterations):
            # Create a new instance of the classifier for this iteration
            current_classifier = clone(self.base_classifier)
            
            # Fit the classifier on the current labeled data
            current_classifier.fit(X_train, y_train)

            if len(unlabeled_pool) == 0:
                break

            # Predict probabilities for unlabeled data
            prediction_probabilities = current_classifier.predict_proba(unlabeled_pool)

            # Find most confident predictions
            max_probabilities = np.max(prediction_probabilities, axis=1)
            confident_predictions_mask = max_probabilities >= self.confidence_threshold

            if not np.any(confident_predictions_mask):
                break

            # Add confident predictions to the labeled dataset
            new_labeled_samples = unlabeled_pool[confident_predictions_mask]
            new_labeled_targets = current_classifier.predict(new_labeled_samples)

            X_train = np.vstack((X_train, new_labeled_samples))
            y_train = np.concatenate((y_train, new_labeled_targets))

            # Remove newly labeled data from the unlabeled pool
            unlabeled_pool = unlabeled_pool[~confident_predictions_mask]

            print(f"Iteration {iteration + 1}: Labeled {len(new_labeled_samples)} new samples. "
                  f"Remaining unlabeled: {len(unlabeled_pool)}")

        # Train the final classifier on all labeled data
        self.final_classifier_ = clone(self.base_classifier)
        self.final_classifier_.fit(X_train, y_train)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Args:
            X (np.ndarray): The input samples.

        Returns:
            np.ndarray: Predicted class labels.

        Raises:
            NotFittedError: If the model has not been fitted yet.
        """
        self._check_is_fitted()
        X = check_array(X)
        return self.final_classifier_.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Args:
            X (np.ndarray): The input samples.

        Returns:
            np.ndarray: The class probabilities of the input samples.

        Raises:
            NotFittedError: If the model has not been fitted yet.
        """
        self._check_is_fitted()
        X = check_array(X)
        return self.final_classifier_.predict_proba(X)

    def _check_is_fitted(self) -> None:
        """
        Check if the model has been fitted.

        Raises:
            NotFittedError: If the model has not been fitted yet.
        """
        check_is_fitted(self, ['final_classifier_', 'classes_'])

class MultiViewCoTrainingClassifier(BaseEstimator, ClassifierMixin):
    """
    Co-Training Classifier adapting its strategy based on the number of views.

    - If n_views == 2: Implements the original Blum & Mitchell co-training
      algorithm with separate conceptual training sets and cross-teaching.
    - If n_views > 2: Implements a multi-view variant where predictions
      are aggregated for pseudo-labeling and a shared training pool is used.
    """
    def __init__(self, base_estimator: BaseEstimator = None,
                 n_views: int = 2, n_iter: int = 50, pool_size: int = 75,
                 n_to_add: int = 10):
        """
        Args:
            base_estimator (BaseEstimator): The base estimator to use for each view.
                                            If None, defaults to RandomForestClassifier(n_estimators=100).
            n_views (int): Number of views to split the features into. Determines strategy.
            n_iter (int): Number of iterations for the co-training process.
            pool_size (int): Size of the pool of unlabeled data to sample from in each iteration.
            n_to_add (int): Total number of examples (most confident positive and negative combined)
                             to attempt to add per view/classifier in each iteration.
                             For n_views=2, each classifier selects n_to_add/2 positive and n_to_add/2 negative.
                             For n_views>2, each classifier selects based on this, but labeling aggregates.
        """
        self.base_estimator = base_estimator
        self.n_views = n_views
        self.n_iter = n_iter
        self.pool_size = pool_size
        self.n_to_add = n_to_add # Simplified: total to add per view

    def fit(self, X: np.ndarray, y: np.ndarray, X_unlabeled: np.ndarray) -> 'MultiViewCoTrainingClassifier':
        """
        Fit the multi-view co-training classifier.

        Args:
            X (np.ndarray): Labeled training data features.
            y (np.ndarray): Labeled training data labels.
            X_unlabeled (np.ndarray): Unlabeled data features.

        Returns:
            MultiViewCoTrainingClassifier: The fitted classifier.
        """
        # Input Validation and Initialization 
        X, y = check_X_y(X, y)
        X_unlabeled = check_array(X_unlabeled)
        self.classes_ = unique_labels(y)
        if len(self.classes_) != 2:
             print("Warning: Co-Training selection logic is often designed for binary "
                   "problems. Performance on multiclass might vary.")
            # For multiclass n_to_add might need different interpretation

        n_features = X.shape[1]
        if n_features < self.n_views:
            raise ValueError(f"Number of features ({n_features}) must be >= n_views ({self.n_views})")
        self.feature_splits_ = np.array_split(range(n_features), self.n_views)

        # Store initial labeled data
        X_initial = X.copy()
        y_initial = y.copy()
        X_unlabeled_current = X_unlabeled.copy()

        # Determine selection counts per class (approximate balance)
        n_pos_per_view = self.n_to_add // 2
        n_neg_per_view = self.n_to_add - n_pos_per_view

        # Train classifiers on initial labeled data
        self.classifiers_ = [clone(self._get_base_estimator()).fit(X_initial[:, split], y_initial)
                             for split in self.feature_splits_]

        # Co-rraining Loop
        for iteration in range(self.n_iter):
            if len(X_unlabeled_current) == 0:
                print(f"Stopping early: No more unlabeled data at iteration {iteration+1}.")
                break

            # Select a pool of unlabeled data
            pool_size_actual = min(self.pool_size, len(X_unlabeled_current))
            pool_indices_orig = np.random.choice(len(X_unlabeled_current), pool_size_actual, replace=False)
            X_pool = X_unlabeled_current[pool_indices_orig]

            # Get Predictions on Pool
            # Store predictions {view_index: {'proba': array, 'pred': array}}
            predictions_pool = {}
            for i, (clf, split) in enumerate(zip(self.classifiers_, self.feature_splits_)):
                if len(X_pool) > 0:
                    proba = clf.predict_proba(X_pool[:, split])
                    pred = self.classes_[np.argmax(proba, axis=1)] # Use fitted classes_
                    predictions_pool[i] = {'proba': proba, 'pred': pred}
                else:
                    predictions_pool[i] = {'proba': np.empty((0, len(self.classes_))), 'pred': np.empty(0)}


            # Data to add in this iteration {view_index_adding: {'X': list, 'y': list, 'indices_in_pool': list}}
            data_to_add_this_iter = {i: {'X': [], 'y': [], 'indices_in_pool': []} for i in range(self.n_views)}
            indices_to_remove_from_unlabeled = set() # Track original indices in X_unlabeled_current

            # Select Examples and Determine Labels (Strategy depends on n_views)
            if self.n_views == 2:
                # Original Co-Training Logic (Blum & Mitchell)
                for view_idx in range(self.n_views):
                    other_view_idx = 1 - view_idx
                    preds_view = predictions_pool[view_idx]['proba']
                    labels_view = predictions_pool[view_idx]['pred']

                    if preds_view.shape[0] == 0: continue # Pool is empty

                    # sort by probability of class 1 for pos, class 0 for neg
                    conf_pos = preds_view[:, 1] # Confidence for class 1
                    conf_neg = preds_view[:, 0] # Confidence for class 0

                    idx_sorted_pos = np.argsort(conf_pos)[::-1] # Descending conf for class 1
                    idx_sorted_neg = np.argsort(conf_neg)[::-1] # Descending conf for class 0

                    selected_indices_pos = idx_sorted_pos[:n_pos_per_view]
                    selected_indices_neg = idx_sorted_neg[:n_neg_per_view]

                    # Combine and ensure uniqueness within this view's selection
                    selected_indices_in_pool = np.unique(np.concatenate((selected_indices_pos, selected_indices_neg)))

                    if len(selected_indices_in_pool) > 0:
                        # Get data and labels PREDICTED BY THIS VIEW (view_idx)
                        X_add = X_pool[selected_indices_in_pool]
                        y_add = labels_view[selected_indices_in_pool] # Label from *this* classifier

                        # Store it to be used for training the OTHER view
                        data_to_add_this_iter[other_view_idx]['X'].append(X_add)
                        data_to_add_this_iter[other_view_idx]['y'].append(y_add)
                        data_to_add_this_iter[other_view_idx]['indices_in_pool'].extend(selected_indices_in_pool.tolist())
                        indices_to_remove_from_unlabeled.update(pool_indices_orig[selected_indices_in_pool])

            else: # self.n_views > 2
                # == Multi-View Logic (Shared Pool, Aggregated Labels) ==
                candidate_indices_in_pool = set()
                # Gather candidates from all views
                for view_idx in range(self.n_views):
                    preds_view = predictions_pool[view_idx]['proba']
                    if preds_view.shape[0] == 0: continue

                    conf_pos = preds_view[:, 1]
                    conf_neg = preds_view[:, 0]
                    idx_sorted_pos = np.argsort(conf_pos)[::-1]
                    idx_sorted_neg = np.argsort(conf_neg)[::-1]
                    selected_indices_pos = idx_sorted_pos[:n_pos_per_view]
                    selected_indices_neg = idx_sorted_neg[:n_neg_per_view]
                    candidate_indices_in_pool.update(selected_indices_pos)
                    candidate_indices_in_pool.update(selected_indices_neg)

                selected_indices_in_pool = list(candidate_indices_in_pool)

                if len(selected_indices_in_pool) > 0:
                    X_add = X_pool[selected_indices_in_pool]

                    # Aggregate predictions for labeling
                    all_probas = np.array([predictions_pool[i]['proba'][selected_indices_in_pool]
                                           for i in range(self.n_views)])
                    avg_proba = np.mean(all_probas, axis=0)
                    y_add = self.classes_[np.argmax(avg_proba, axis=1)] # Label by average confidence

                    # Store it to be added to the shared pool (conceptually view_idx 0 here)
                    data_to_add_this_iter[0]['X'].append(X_add)
                    data_to_add_this_iter[0]['y'].append(y_add)
                    data_to_add_this_iter[0]['indices_in_pool'].extend(selected_indices_in_pool) # Not strictly needed after aggregation
                    indices_to_remove_from_unlabeled.update(pool_indices_orig[selected_indices_in_pool])

            # Remove Added Examples from Unlabeled Pool            
            if indices_to_remove_from_unlabeled:
                remove_mask = np.ones(len(X_unlabeled_current), dtype=bool)
                # Ensure indices are within bounds before trying to set mask
                valid_indices = [idx for idx in indices_to_remove_from_unlabeled if idx < len(remove_mask)]
                if valid_indices:
                    remove_mask[valid_indices] = False
                X_unlabeled_current = X_unlabeled_current[remove_mask]
            else:
                 # No examples added in this iteration by any view
                 print(f"Iteration {iteration+1}: No confident examples found to add.")
                 continue # Skip retraining if nothing was added

            # Retrain Classifiers
            if self.n_views == 2:
                # Original co-training retraining
                new_classifiers = []
                for view_idx in range(self.n_views):
                    # Data for this view = Initial + Data added by the OTHER view
                    X_to_add = np.vstack(data_to_add_this_iter[view_idx]['X']) if data_to_add_this_iter[view_idx]['X'] else np.empty((0, X_initial.shape[1]))
                    y_to_add = np.concatenate(data_to_add_this_iter[view_idx]['y']) if data_to_add_this_iter[view_idx]['y'] else np.empty(0)

                    if X_to_add.shape[0] > 0:
                         X_train_view = np.vstack((X_initial, X_to_add))
                         y_train_view = np.concatenate((y_initial, y_to_add))
                    else:
                         X_train_view = X_initial
                         y_train_view = y_initial

                    # Clone and retrain on combined data using this view's features
                    clf = clone(self._get_base_estimator())
                    clf.fit(X_train_view[:, self.feature_splits_[view_idx]], y_train_view)
                    new_classifiers.append(clf)
                self.classifiers_ = new_classifiers

            else: # self.n_views > 2
                # Multi-view retraining (shared pool)
                # Aggregate all added data into the main pool
                X_added_all = np.vstack(data_to_add_this_iter[0]['X']) if data_to_add_this_iter[0]['X'] else np.empty((0, X_initial.shape[1]))
                y_added_all = np.concatenate(data_to_add_this_iter[0]['y']) if data_to_add_this_iter[0]['y'] else np.empty(0)

                if X_added_all.shape[0] > 0:
                    # Update the shared labeled pool for the next iteration's reference
                    # Note: For pure multi-view retraining, we'd just use this combined pool.
                    # Let's re-initialize X/y for retraining for simplicity here.
                    X_train_shared = np.vstack((X_initial, X_added_all))
                    y_train_shared = np.concatenate((y_initial, y_added_all))
                     # Update X_initial, y_initial conceptually for the next iteration
                     # This makes the multi-view behave more like the original where added data persists
                    X_initial = X_train_shared.copy()
                    y_initial = y_train_shared.copy()

                else:
                    X_train_shared = X_initial
                    y_train_shared = y_initial

                # Retrain all classifiers on the updated shared pool
                self.classifiers_ = [clone(self._get_base_estimator()).fit(X_train_shared[:, split], y_train_shared)
                                     for split in self.feature_splits_]

        # Final check: Ensure classifiers were trained
        if not hasattr(self, 'classifiers_') or not self.classifiers_:
             raise RuntimeError("Classifier fitting failed.")

        # Store the final state if needed (e.g., final combined X,y for n_views>2)
        # For n_views=2, the state is captured in the final classifiers trained on their respective augmented sets.
        # For n_views>2, the state is captured by classifiers trained on the final aggregated X_initial, y_initial

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities by averaging predictions from all views.
        """
        check_is_fitted(self)
        X = check_array(X)
        predictions = []
        for clf, split in zip(self.classifiers_, self.feature_splits_):
            # Handle cases where a view might have 0 features after split (unlikely but safe)
            if X[:, split].shape[1] > 0:
                 predictions.append(clf.predict_proba(X[:, split]))
            else:
                 # Cannot predict with 0 features, return uniform probability? Or error?
                 # For simplicity, let's assume uniform if this happens.
                 num_samples = X.shape[0]
                 num_classes = len(self.classes_)
                 predictions.append(np.full((num_samples, num_classes), 1.0 / num_classes))

        # Average probabilities across views
        avg_prediction = np.mean(predictions, axis=0)
        return avg_prediction

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels based on the highest average probability.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def _get_base_estimator(self):
        # Helper to get a fresh clone or the default estimator
        if self.base_estimator is None:
            # Define default here to avoid storing a fitted default in __init__
            return RandomForestClassifier(n_estimators=100, random_state=271828)
        else:
            return clone(self.base_estimator)

    # Add _more_tags for sklearn compatibility
    def _more_tags(self):
        return {'requires_y': True}
        


###### ImPULSE Classifier - Adapted from https://github.com/woldemarg/impulse_classifier ######

class ParallelMixin:
    @staticmethod
    def do_parallel(
            func: Callable,
            iterable: Iterable,
            concatenate_result: bool = True,
            **kwargs: dict[str, str | int]) -> np.array:
        """
        Applies a function to each element of an iterable in parallel and
        returns the results as a numpy array.

        Args:
            func (Callable): The function to apply to each element of the iterable.
            iterable (Iterable): The iterable to apply the function to.
            concatenate_result (bool, optional): If True, concatenates the results
                along the second axis. Default is True.
            **kwargs (dict): Additional keyword arguments to pass to the function.

        Returns:
            np.array: The results of applying the function to the iterable,
                in a numpy array.

        Examples:
            def square(x, **kwargs):
                return x**2
            arr = [1, 2, 3, 4, 5]
            result = do_parallel(square, arr, concatenate_result=False)
            print(result)
            # Output: [1, 4, 9, 16, 25]
        """
        # Set the backend for parallel processing
        backend = kwargs.get('backend', 'loky')

        with parallel_backend(backend, n_jobs=effective_n_jobs()):
            lst_processed = Parallel()(
                delayed(func)(el, **kwargs)
                for el in iterable)

        if concatenate_result:
            return np.concatenate(
                [arr.reshape(-1, 1) for arr in lst_processed], axis=1)

        return lst_processed
    

class ImPULSEClassifier(ParallelMixin):
    def __init__(self,
                 estimator: object,
                 min_learning_rate: float,
                 max_learning_rate: float,
                 num_iterations: int,
                 hold_out_ratio: float,
                 random_state: int,
                 n_jobs: int = None) -> None:
        """
        Initialize the ImPULSEClassifier.

        Args:
            estimator (object): The base estimator to be used.
            min_learning_rate (float): Minimum learning rate.
            max_learning_rate (float): Maximum learning rate.
            num_iterations (int): Number of iterations for learning rate adjustment.
            hold_out_ratio (float): Ratio of data to hold out for evaluation.
            random_state (int): Random state for reproducibility.
            n_jobs (int): Number of jobs for parallel processing.
        """
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.num_iterations = num_iterations
        self.hold_out_ratio = hold_out_ratio
        self.update_estimator = self._create_estimator_updater(estimator)
        self.random_state = random_state
        self.n_jobs = n_jobs if n_jobs else effective_n_jobs()
        self.model = None
        self.prior = None

    @staticmethod
    def _create_estimator_updater(estimator: object) -> Callable:
        """
        Create a function to update the hyperparameters of the estimator.

        Args:
            estimator (object): The base estimator.

        Returns:
            Callable: A function that updates the estimator's hyperparameters.
        """
        def update_hyperparameters(**kwargs):
            return estimator.__class__(**{**estimator.get_params(), **kwargs})
        return update_hyperparameters

    @staticmethod
    def _custom_score(y_true: np.array, y_pred: np.array) -> tuple[str, float, bool]:
        """
        Compute a custom score for evaluation.

        Args:
            y_true (np.array): True labels.
            y_pred (np.array): Predicted labels.

        Returns:
            tuple[str, float, bool]: Custom score name, value, and a flag indicating its importance.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            score = average_precision_score(y_true, np.round(y_pred))
        return 'custom_score', score, True

    def _train_model(self,
                     X_train: np.array,
                     X_eval: np.array,
                     y_train: np.array,
                     y_eval: np.array,
                     estimator_updater: Callable,
                     sample_weights: np.array,
                     learning_rate: float,
                     **kwargs) -> object:
        """
        Train the model with the given data and parameters.

        Args:
            X_train (np.array): Training features.
            X_eval (np.array): Evaluation features.
            y_train (np.array): Training labels.
            y_eval (np.array): Evaluation labels.
            estimator_updater (Callable): Function to update the estimator.
            sample_weights (np.array): Sample weights.
            learning_rate (float): Learning rate.
            **kwargs: Additional parameters for the estimator.

        Returns:
            object: Trained model.
        """
        class_counts = np.bincount(y_train)
        class_weights = {c: (len(y_train) / (len(set(y_train)) * class_counts[c])) for c in set(y_train)}

        model = estimator_updater(learning_rate=learning_rate, class_weight=class_weights, **kwargs)
        model.fit(X_train, y_train, sample_weight=sample_weights, eval_metric=self._custom_score,
                  eval_set=(X_eval, y_eval), callbacks=[early_stopping(25, verbose=0)])
        return model

    def _iterate_learning_rates(self,
                                learning_rates: Iterable[float],
                                X_train: np.array,
                                X_eval: np.array,
                                y_train: np.array,
                                y_eval: np.array,
                                **kwargs) -> bool:
        """
        Iterate over learning rates to train the model.

        Args:
            learning_rates (Iterable[float]): Learning rates to iterate over.
            X_train (np.array): Training features.
            X_eval (np.array): Evaluation features.
            y_train (np.array): Training labels.
            y_eval (np.array): Evaluation labels.
            **kwargs: Additional parameters for the estimator.

        Returns:
            bool: True if the model was successfully trained, False otherwise.
        """
        sample_weights = np.full(len(y_train), 1)
        y_train_copy = y_train.copy()
        delta_positives, delta_confidence = 1, 1

        for learning_rate in tqdm(learning_rates):
            if self.model:
                sum_positives, sum_confidence = y_train_copy.sum(), sample_weights.sum()
                chunks = np.array_split(X_train, self.n_jobs)
                preds = np.concatenate(self.do_parallel(self.model.predict_proba, chunks, concatenate_result=False))[:, 1]
                bins = np.percentile(preds, q=np.linspace(0, 100, 11))
                bins[-1] += np.finfo(float).eps
                bin_indices = np.digitize(preds, bins)
                ones_indices = bin_indices >= 9
                zeros_indices = bin_indices <= 2
                y_train_copy[ones_indices] = 1
                true_indices = np.where(y_train == 1)[0]
                sample_weights = np.full(X_train.shape[0], 0.5)
                sample_weights[true_indices] = 1
                sample_weights[ones_indices] = preds[ones_indices]
                sample_weights[zeros_indices] = 1 - preds[zeros_indices]
                delta_positives = y_train_copy.sum() - sum_positives
                delta_confidence = sample_weights.sum() - sum_confidence

            if delta_positives > 0 or delta_confidence > 0:
                try:
                    model = self._train_model(X_train, X_eval, y_train_copy, y_eval, estimator_updater=self.update_estimator,
                                              sample_weights=sample_weights, learning_rate=learning_rate, **kwargs)
                except ValueError:
                    self.model = None
                    self.prior = None
                    return False

                prior = np.average(a=np.ma.masked_array(y_train_copy, mask=y_train),
                                   weights=np.ma.masked_array(sample_weights, mask=y_train))
                self.model = model
                self.prior = prior

        print(f'Added {y_train_copy.sum() - y_train.sum()} new labels.')
        return True

    def fit(self, X: np.array, y: np.array) -> None:
        """
        Fit the model to the given data.

        Args:
            X (np.array): Features.
            y (np.array): Labels.
        """
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=self.hold_out_ratio,
                                                            random_state=self.random_state, stratify=y)
        learning_rates = np.geomspace(self.max_learning_rate, self.min_learning_rate, self.num_iterations)
        fitted = self._iterate_learning_rates(learning_rates, X_train, X_eval, y_train, y_eval)

        n = 1
        while not fitted:
            if n >= 10:
                raise ValueError('The model training process failed to converge.')
            print('Adjusting parameters for another attempt.')
            self.max_learning_rate *= np.exp(-0.1)
            self.min_learning_rate *= np.exp(-0.1)
            learning_rates = np.geomspace(self.max_learning_rate, self.min_learning_rate, self.num_iterations)
            X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=self.hold_out_ratio,
                                                                random_state=n, stratify=y)
            fitted = self._iterate_learning_rates(learning_rates, X_train, X_eval, y_train, y_eval)
            n += 1

    def __getattr__(self, attrname: str):
        """
        Delegate attribute access to the underlying model if the attribute exists.

        Args:
            attrname (str): Attribute name.

        Returns:
            Any: Attribute value.
        """
        if hasattr(self.model, attrname):
            return getattr(self.model, attrname)
        raise AttributeError(f"{self.__class__.__name__} object has no attribute '{attrname}'")

    def compute_confusion_matrix(self,
                                 X_train: np.array,
                                 y_train: np.array,
                                 X_test: np.array,
                                 y_test: np.array) -> np.array:
        """
        Compute the confusion matrix for the given test data.

        Args:
            X_train (np.array): Training features.
            y_train (np.array): Training labels.
            X_test (np.array): Test features.
            y_test (np.array): Test labels.

        Returns:
            np.array: Confusion matrix.
        """
        test_size = len(y_test)
        pos_indices = np.where(y_train)[0]
        train_prob = self.predict_proba(X_train[pos_indices])[:, 1]
        pos_labelled_prob = train_prob.mean()
        test_prob = self.predict_proba(X_test)[:, 1]
        conf_matrix = confusion_matrix(y_test, (test_prob > 0.5))
        true_neg, false_pos, false_neg, true_pos = conf_matrix.ravel()

        if self.prior > 0:
            false_pos_odds = ((np.abs(y_test - 1)).sum() * self.prior * pos_labelled_prob)
            true_pos_odds = ((np.abs(y_test - 1)).sum() * self.prior * (1 - pos_labelled_prob))
            false_pos_adj = max(0, false_pos - (false_pos_odds + true_pos_odds))
            true_pos_adj = true_pos + false_pos_odds
            false_neg_adj = false_neg + true_pos_odds
            conf_matrix = np.array([[true_neg, false_pos_adj], [false_neg_adj, true_pos_adj]])
            conf_matrix = (np.round(conf_matrix * test_size / conf_matrix.sum()) * conf_matrix.sum() // test_size)
            true_neg_odds = test_size - conf_matrix.sum()
            true_neg_adj = true_neg + true_neg_odds
            conf_matrix[0, 0] = true_neg_adj

        return conf_matrix.astype(int)
    
###### End of ImPULSE Classifier ######