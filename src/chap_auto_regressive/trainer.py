"""The training loop.

[`Trainer`][chap_auto_regressive.trainer.Trainer] fits a flax network by minimizing a
caller-supplied loss with the Adam optimizer. The per-step update is JIT-compiled
with ``jax.jit`` and uses ``jax.value_and_grad`` for the gradient; a small L2
penalty on the weight matrices is added for regularization.
"""

import logging
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from more_itertools import peekable

from .data_loader import DataLoader

logger = logging.getLogger(__name__)


class TrainState(train_state.TrainState):
    """Flax training state extended with a PRNG key for dropout.

    Attributes:
        key: The PRNG key folded per step to drive dropout.
    """

    key: jax.Array


def l2_regularization(params: Any, scale: float = 1.0) -> Any:
    """Sum of squared weight-matrix entries, used as an L2 penalty.

    Only rank-2 parameters (weight matrices) are penalized; biases and other
    parameters are left out.

    Args:
        params: A pytree of model parameters.
        scale: Multiplier applied to the summed squared weights.

    Returns:
        The scaled L2 penalty as a scalar.
    """
    return sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params) if p.ndim == 2) * scale


class Trainer:
    """Fits a flax model to windowed data with Adam.

    The trainer initializes the model from the first window, then for ``n_iter``
    epochs runs a JIT-compiled gradient step over every training window. When a
    validation loader is supplied, the validation loss is reported every ten
    epochs.
    """

    def __init__(
        self,
        model: Any,
        n_iter: int = 3000,
        learning_rate: float = 1e-5,
        validation_loader: Optional[DataLoader] = None,
        seed: int = 0,
        eval_every: int = 5,
        patience: int = 6,
    ):
        """Configure the trainer.

        Args:
            model: The flax module to fit.
            n_iter: Number of epochs (full passes over the windows).
            learning_rate: Adam learning rate.
            validation_loader: Optional loader providing a held-out window for
                periodic validation-loss reporting.
            seed: Seed for parameter initialization and dropout, so independently
                seeded trainers yield distinct ensemble members.
            eval_every: Epochs between validation evaluations.
            patience: Stop after this many consecutive evaluations without a new
                best validation loss. Only active when a validation loader is
                supplied; ``0`` disables stopping and restores the previous
                fixed-length behaviour.
        """
        self.model = model
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self._validation_loader = validation_loader
        self.seed = seed
        self.eval_every = eval_every
        self.patience = patience
        #: Epoch at which the best validation loss was seen, set by ``train``.
        self.best_epoch: Optional[int] = None
        #: The best validation loss seen, set by ``train``.
        self.best_validation_loss: Optional[float] = None

    def train(self, data_loader: DataLoader, loss_fn: Callable) -> "TrainState":
        """Train the model and return the final state.

        Initializes parameters from the first window, then repeatedly applies a
        JIT-compiled Adam step over all windows. Each step adds an L2 penalty and
        folds a fresh dropout key from the step counter for stochastic
        regularization.

        Args:
            data_loader: Yields ``(x, ar_y, y)`` training windows.
            loss_fn: Maps ``(eta, y)`` to a scalar loss (the negative
                log-likelihood under the model's output distribution).

        Returns:
            The final [`TrainState`][chap_auto_regressive.trainer.TrainState] holding the
            trained parameters.
        """
        ix, iar_y, iy = peekable(iter(data_loader)).peek()
        params = self.model.init(jax.random.PRNGKey(self.seed), ix, iar_y, training=False)
        dropout_key = jax.random.PRNGKey(self.seed + 40)

        training_state = TrainState.create(
            apply_fn=self.model.apply, params=params, tx=optax.adam(self.learning_rate), key=dropout_key
        )

        @jax.jit
        def train_step(state: TrainState, dropout_key, x, ar_y, y) -> Tuple[TrainState, jnp.ndarray]:
            dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

            def loss_func(params):
                eta = state.apply_fn(params, x, ar_y, training=True, rngs={"dropout": dropout_train_key})
                return loss_fn(eta, y) + l2_regularization(params, 0.001)

            grad_func = jax.value_and_grad(loss_func)
            loss, grad = grad_func(state.params)
            state = state.apply_gradients(grads=grad)
            return state, loss

        @jax.jit
        def get_validation_loss(state: TrainState, x, ar_y, y):
            return loss_fn(state.apply_fn(state.params, x, ar_y, training=False), y)

        def validation_loss() -> float:
            total = 0.0
            for v_x, v_ar, v_y in iter(self._validation_loader):
                total += float(get_validation_loss(training_state, v_x, v_ar, v_y))
            return total

        best_state, best_loss, best_epoch, since_best = training_state, float("inf"), 0, 0
        for i in range(self.n_iter):
            for x, ar_y, y in iter(data_loader):
                training_state, cur_loss = train_step(training_state, dropout_key, x, ar_y, y)
            if self._validation_loader is None:
                if i % 10 == 0:
                    logger.info("epoch %d: loss=%s", i, cur_loss)
                continue
            if i % self.eval_every:
                continue
            current = validation_loss()
            logger.info("epoch %d: loss=%s validation_loss=%s", i, cur_loss, current)
            if current < best_loss:
                # Keeping the best parameters, not the last, is the whole point:
                # held-out loss on these series bottoms out early and then climbs
                # for the rest of the run.
                best_state, best_loss, best_epoch, since_best = training_state, current, i, 0
                continue
            since_best += 1
            if self.patience and since_best >= self.patience:
                logger.info("early stopping at epoch %d; best was epoch %d (%.4f)", i, best_epoch, best_loss)
                break

        if self._validation_loader is None:
            return training_state
        self.best_epoch, self.best_validation_loss = best_epoch, best_loss
        return best_state
