import numpy as np
import scipy
from typing import List, Union, Optional, Any


class GroupingAndOverlappingMultiTasker:
    def __init__(
        self,
        activation: str = "relu",
        shared_basis_shape: tuple[int, ...] = (100,),
        k: int = 7,
        task_layers_shape: tuple[int, ...] = (50,),
        solver: str = "adam",
        alpha: float = 0.0001,
        batch_size: Union[int, str] = "auto",
        learning_rate: str = "constant",
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        tol: float = 1e-4,
        verbose: bool = False,
        warm_start: bool = True,
        momentum: float = 0.9,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-08,
        n_iter_no_change: int = 10,
    ) -> None:
        self.activation: str = activation
        self.shared_basis_shape: tuple[int, ...] = shared_basis_shape
        self.k: int = k
        self.task_layers_shape: tuple[int, ...] = task_layers_shape
        self.solver: str = solver
        self.alpha: float = alpha
        self.batch_size: Union[int, str] = batch_size
        self.learning_rate: str = learning_rate
        self.learning_rate_init: float = learning_rate_init
        self.shuffle: bool = shuffle
        self.max_iter: int = max_iter
        self.random_state: Optional[int] = random_state
        self.tol: float = tol
        self.verbose: bool = verbose
        self.warm_start: bool = warm_start
        self.momentum: float = momentum
        self.early_stopping: bool = early_stopping
        self.validation_fraction: float = validation_fraction
        self.beta_1: float = beta_1
        self.beta_2: float = beta_2
        self.epsilon: float = epsilon
        self.n_iter_no_change: int = n_iter_no_change

        self.shared_bases_: Optional[np.ndarray] = None
        self.task_weights_: Optional[np.ndarray] = None

    def fit(
        self,
        X: List[np.ndarray],
        y: List[np.ndarray],
    ) -> "GroupingAndOverlappingMultiTasker":
        """
        Fit the Go-MTL model to multiple tasks.

        Parameters
        ----------
        X : list of (n_t, d) arrays
            Feature matrices for each task.
        y : list of (n_t,) arrays
            Target vectors for each task.

        Returns
        -------
        self : GroupingAndOverlappingMultiTasker
            The fitted model.
        """
        self.current_X: List[np.ndarray] = X
        self.current_y: List[np.ndarray] = y
        # TODO: implement alternating-minimization here
        return self

    def predict(
        self,
        X: np.ndarray,
        task_id: int = 0,
    ) -> np.ndarray:
        """
        Predict using the fitted model for a single task.

        Parameters
        ----------
        X : (n_samples, d) array
            New data.
        task_id : int
            Index of the task to predict for.

        Returns
        -------
        y_pred : (n_samples,) array
            Predicted targets.
        """
        # TODO: compute X @ (L @ S[:, task_id])
        return np.zeros(X.shape[0])

    def get_params(self) -> dict[str, Any]:
        """
        Get model hyperparameters (for compatibility with sklearn).

        Returns
        -------
        params : dict
            Mapping of parameter names to their values.
        """
        return {
            "activation": self.activation,
            "shared_basis_shape": self.shared_basis_shape,
            "k": self.k,
            "task_layers_shape": self.task_layers_shape,
            "solver": self.solver,
            "alpha": self.alpha,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "learning_rate_init": self.learning_rate_init,
            "max_iter": self.max_iter,
            "shuffle": self.shuffle,
            "random_state": self.random_state,
            "tol": self.tol,
            "verbose": self.verbose,
            "warm_start": self.warm_start,
            "momentum": self.momentum,
            "early_stopping": self.early_stopping,
            "validation_fraction": self.validation_fraction,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "epsilon": self.epsilon,
            "n_iter_no_change": self.n_iter_no_change,
        }

    def partial_fit(
        self,
        X: List[np.ndarray],
        y: List[np.ndarray],
        task_id: int = 0,
    ) -> "GroupingAndOverlappingMultiTasker":
        """
        Incrementally fit model on data for one task.

        Parameters
        ----------
        X : list of (n_t, d) arrays
        y : list of (n_t,) arrays
        task_id : int

        Returns
        -------
        self : GroupingAndOverlappingMultiTasker
        """
        # TODO: update only the parameters for `task_id`
        return self

    def predict_log_proba(
        self,
        X: np.ndarray,
        task_id: int = 0,
    ) -> np.ndarray:
        """
        Return log-probability estimates for classification tasks.

        Parameters
        ----------
        X : (n_samples, d) array
        task_id : int

        Returns
        -------
        log_proba : (n_samples, n_classes) array
        """
        # TODO: implement if applicable
        raise NotImplementedError

    def predict_proba(
        self,
        X: np.ndarray,
        task_id: int = 0,
    ) -> np.ndarray:
        """
        Return probability estimates for classification tasks.

        Parameters
        ----------
        X : (n_samples, d) array
        task_id : int

        Returns
        -------
        proba : (n_samples, n_classes) array
        """
        # TODO: implement if applicable
        raise NotImplementedError

    def score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_id: int = 0,
    ) -> float:
        """
        Return a score (e.g., R^2 or accuracy) for one task.

        Parameters
        ----------
        X : (n_samples, d) array
        y : (n_samples,) array
        task_id : int

        Returns
        -------
        score : float
        """
        # TODO: compute performance metric
        return 0.0

    def predict_all(
        self,
        X_list: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Predict for all tasks in one call.

        Parameters
        ----------
        X_list : list of (n_t, d) arrays

        Returns
        -------
        y_preds : list of (n_t,) arrays
        """
        return [self.predict(X, i) for i, X in enumerate(X_list)]
