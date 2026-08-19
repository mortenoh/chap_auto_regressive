"""The public model and its trained predictor.

[`AutoRegressiveModel`][chap_auto_regressive.model.AutoRegressiveModel] is the estimator CHAP
trains; its ``train`` method returns a
[`FlaxPredictor`][chap_auto_regressive.model.FlaxPredictor] that produces probabilistic
forecasts. Both share the same output head — the network emits two channels per
period (``eta``) which [`nb_head`][chap_auto_regressive.distributions.nb_head] turns into a
negative-binomial distribution that is then sampled.

The public API speaks tidy :class:`pandas.DataFrame` objects (one row per location
and time period); the model itself has no dependency on chap-core.
"""

import logging
import pickle
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .data_loader import DataSet as DLDataSet
from .data_loader import SimpleDataLoader
from .distributions import nb_head
from .rnn_model import build_network
from .trainer import Trainer
from .transforms import REQUIRED_COVARIATES, ZScaler, get_series, location_groups

logger = logging.getLogger(__name__)


def _check_predict_inputs(
    historic: pd.DataFrame,
    future: pd.DataFrame,
    prediction_length: int,
    context_length: int,
    training_locations: list | None = None,
) -> None:
    """Validate the history/future frames before forecasting.

    Args:
        historic: Recent history including observed cases.
        future: The periods to forecast, with covariates but no cases.
        prediction_length: The fixed forecast horizon the model was built for.
        context_length: Number of past periods the model reads as context; each
            location's history must provide at least this many.
        training_locations: The canonical (sorted) locations the model was trained
            on. When provided, the frames must cover exactly this set — the model
            learns per-location embeddings by sorted index, so an unseen location
            would silently borrow another location's embedding.

    Raises:
        ValueError: If the two frames cover different location sets, if the
            locations differ from ``training_locations``, if any location provides
            fewer than ``context_length`` history periods, or if any location asks
            for more than ``prediction_length`` future periods.
    """
    historic_locations = set(historic["location"].unique())
    if historic_locations != set(future["location"].unique()):
        raise ValueError("historic and future must cover the same set of locations")
    if training_locations is not None and sorted(historic_locations) != sorted(training_locations):
        raise ValueError(
            "prediction locations must match the training locations "
            f"{sorted(training_locations)}, but got {sorted(historic_locations)}"
        )
    historic_counts = historic.groupby("location").size()
    if (historic_counts < context_length).any():
        raise ValueError(
            f"each location's history must have at least context_length={context_length} periods, "
            f"got period counts {sorted(set(historic_counts.tolist()))}"
        )
    # The model forecasts up to its trained horizon; callers (e.g. a chap eval
    # backtest) may request fewer periods, but more than prediction_length would
    # extrapolate beyond what the network was trained for.
    counts = future.groupby("location").size()
    if (counts > prediction_length).any():
        raise ValueError(
            f"each location's future must have at most prediction_length={prediction_length} "
            f"periods, got period counts {sorted(set(counts.tolist()))}"
        )


def _forecast_frame(future: pd.DataFrame, samples: Any) -> pd.DataFrame:
    """Assemble the prediction output frame from per-location samples.

    Args:
        future: The future-covariate frame the forecast was made for.
        samples: Sampled counts with a leading location axis, each entry shaped
            ``(periods, n_samples)`` (aligned with
            [`location_groups`][chap_auto_regressive.transforms.location_groups]).

    Returns:
        A long frame with columns ``time_period``, ``location`` and one
        ``sample_i`` column per draw.
    """
    groups = list(location_groups(future))
    frames = []
    for (location, sub), location_samples in zip(groups, samples):
        location_samples = np.asarray(location_samples)
        frame = pd.DataFrame({f"sample_{i}": location_samples[:, i] for i in range(location_samples.shape[-1])})
        frame.insert(0, "location", location)
        # Label each location with its own forecast periods. location_groups
        # sorts each group by time_period, matching the order the samples were
        # produced in, so locations with differing future periods stay correct.
        frame.insert(0, "time_period", sub["time_period"].to_numpy())
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class FlaxPredictor:
    """A trained model that turns history and future covariates into samples.

    Produced by [`AutoRegressiveModel.train`][chap_auto_regressive.model.AutoRegressiveModel.train].
    It holds the fitted network parameters and the feature scaler, and can be
    pickled to disk and reloaded.

    Attributes:
        distribution_head: Maps the network's two-channel output to a sampling
            distribution (the negative-binomial head).
    """

    distribution_head = staticmethod(nb_head)

    def __init__(
        self,
        params: Any,
        transform: Callable,
        architecture: dict,
        prediction_length: int,
        context_length: int,
        locations: list | None = None,
        covariates: Sequence[str] = REQUIRED_COVARIATES,
    ):
        # The network is rebuilt from the saved architecture spec, so a loaded
        # predictor is self-contained and does not depend on how the caller's
        # AutoRegressiveModel happens to be configured.
        self.architecture = dict(architecture)
        # Canonical (sorted) training locations; predictions must cover this exact
        # set because the network's per-location embeddings are indexed by sorted
        # position. None only for predictors loaded from a legacy pickle.
        self.locations = list(locations) if locations is not None else None
        self.model = build_network(len(self.locations) if self.locations else 1, **self.architecture)
        # Accept either a single parameter pytree or a list of them (a deep
        # ensemble). Stored as a list; predict pools the members' samples.
        self._params_list = params if isinstance(params, list) else [params]
        self._params = self._params_list[0]
        self._transform = transform
        self.prediction_length = prediction_length
        self.context_length = context_length
        # The covariate columns (in order) the network was trained on; predict
        # must build features the same way. Defaults to the required covariates
        # for predictors loaded from a legacy pickle that predate this field.
        self.covariates = tuple(covariates)
        self.rng_key = jax.random.PRNGKey(1234)

    def get_samples(self, eta: Any, n_samples: int) -> Any:
        """Draw samples from the output distribution for each period.

        Args:
            eta: The network's per-period output, shape ``(..., periods, 2)``.
            n_samples: Number of samples to draw per period.

        Returns:
            Sampled counts, with a leading sample axis of length ``n_samples``.
        """
        self.rng_key, sample_key = jax.random.split(self.rng_key)
        return self.distribution_head(eta).sample(sample_key, (n_samples,))

    def predict(self, historic: pd.DataFrame, future: pd.DataFrame, num_samples: int = 100) -> pd.DataFrame:
        """Forecast the future periods as samples per location.

        The last ``context_length`` periods of history (with observed cases) are
        concatenated with the future covariates, run through the network, and the
        forecast-horizon outputs are sampled.

        Args:
            historic: Recent history including observed ``disease_cases``.
            future: The periods to forecast, with covariates but no cases.
            num_samples: Number of samples to draw per location and period.

        Returns:
            A frame with columns ``time_period``, ``location`` and one ``sample_i``
            column per draw.
        """
        _check_predict_inputs(historic, future, self.prediction_length, self.context_length, self.locations)
        x, _ = get_series(future, self.covariates)
        prev_values, prev_y = get_series(historic, self.covariates)
        prev_values = prev_values[:, -self.context_length :]
        prev_y = prev_y[:, -self.context_length :]
        full_x = jnp.concatenate([prev_values, x], axis=1)
        dataset = DLDataSet(full_x, prev_y, forecast_length=self.prediction_length, context_length=self.context_length)
        dataset.set_transform(self._transform)
        x, y = dataset.prediction_instance()
        n_prev = prev_values.shape[1]
        # Pool samples across the ensemble members. Each member draws an equal
        # share of num_samples (the first members absorb the remainder) so the
        # output keeps exactly num_samples columns.
        n_members = len(self._params_list)
        shares = [num_samples // n_members + (1 if k < num_samples % n_members else 0) for k in range(n_members)]
        member_samples = []
        for params, share in zip(self._params_list, shares):
            if share == 0:
                continue
            eta = self.model.apply(params, x, y)
            member_samples.append(np.asarray(self.get_samples(eta[:, n_prev - 1 :], share)))
        samples = np.concatenate(member_samples, axis=-1)  # pool along the trailing sample axis
        return _forecast_frame(future, samples)

    def save(self, path: str) -> None:
        """Pickle everything needed to reload a self-contained predictor.

        The payload carries the ensemble parameters, the feature scaler, the
        training locations and covariates, **and the network architecture and
        window lengths** — so ``load`` rebuilds the exact network without needing
        the original model configuration.

        Args:
            path: Destination file path.
        """
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "params_list": self._params_list,
                    "transform": self._transform,
                    "locations": self.locations,
                    "covariates": self.covariates,
                    "architecture": self.architecture,
                    "prediction_length": self.prediction_length,
                    "context_length": self.context_length,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "FlaxPredictor":
        """Reload a self-contained predictor previously written by ``save``.

        Args:
            path: Path to the pickled predictor.

        Returns:
            The reconstructed ``FlaxPredictor``.
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return cls(
            payload["params_list"],
            payload["transform"],
            payload["architecture"],
            payload["prediction_length"],
            payload["context_length"],
            locations=payload.get("locations"),
            covariates=payload.get("covariates", REQUIRED_COVARIATES),
        )


class AutoRegressiveModel:
    """Deep auto-regressive forecaster for disease case counts.

    The estimator wraps the [`ARModel2`][chap_auto_regressive.rnn_model.ARModel2] network: it
    extracts and scales features, slices each location's series into
    ``context_length + prediction_length`` windows, and fits the network by
    maximizing the negative-binomial likelihood of the observed cases. Calling
    ``train`` returns a [`FlaxPredictor`][chap_auto_regressive.model.FlaxPredictor].

    All input/output is via tidy :class:`pandas.DataFrame` objects with the columns
    ``location``, ``time_period``, ``rainfall``, ``mean_temperature``,
    ``population`` and (for training) ``disease_cases``. Listing extra column
    names in ``additional_covariates`` feeds them to the network as further
    features, on top of the always-present required three.

    The defaults are the tuned configuration: a GRU encoder/decoder, a 3-year
    context, and a 5-member deep ensemble.

    Attributes:
        prediction_length: Number of periods to forecast ahead.
        n_iter: Number of training epochs (per ensemble member).
        context_length: Number of past periods read as context.
        learning_rate: Adam learning rate.
        n_ensemble: Independently seeded models to train and pool at predict time
            (a deep ensemble; improves predictive calibration / CRPS).
        additional_covariates: Extra covariate columns to use as features,
            appended after the required ``rainfall``, ``mean_temperature`` and
            ``population``. Empty by default.
        cell: Recurrent cell type (``"gru"`` or ``"simple"``).
        rnn_features: Hidden width of the encoder/decoder cells.
        preprocess_hidden / preprocess_output / embedding_dim / head_features:
            Network layer widths.
        distribution_head: Maps the network output to the sampling distribution.
    """

    # --- training ---
    prediction_length = 3
    n_iter: int = 400
    context_length = 36
    learning_rate = 1e-3
    n_ensemble: int = 5
    # Offsets every ensemble member's seed. The model is otherwise fully
    # deterministic, so without this two runs of the same configuration are
    # bit-identical and the seed variance -- the noise floor any change has to
    # clear -- cannot be measured at all.
    seed_offset: int = 0
    # --- early stopping ---
    # Held-out loss on these series bottoms out around epoch 30 and then climbs
    # for the remaining 370, so training to a fixed n_iter costs 1.3-2.5 nats.
    # Each member first fits on all but the held-out tail to find its own best
    # epoch, then refits on the full series for that many epochs -- so no data is
    # permanently given up, and the usual run is several times cheaper than the
    # fixed-length one it replaces.
    early_stopping: bool = True
    #: Extra periods the held-out block gets on top of the one window it must
    #: contain, so the block spans ``context_length + prediction_length +
    #: validation_periods`` periods and yields ``validation_periods + 1`` windows.
    validation_periods: int = 12
    patience: int = 6
    eval_every: int = 5
    additional_covariates: Sequence[str] = ()
    # --- network architecture (persisted in the saved predictor) ---
    cell: str = "gru"
    rnn_features: int = 16
    preprocess_hidden: int = 16
    preprocess_output: int = 8
    embedding_dim: int = 8
    head_features: int = 24
    rnn_layers: int = 1
    recursive_decode: bool = False
    dropout_rate: float = 0.2
    input_dropout_rate: float = 0.0
    distribution_head = staticmethod(nb_head)

    def __init__(self, rng_key=jax.random.PRNGKey(100)):
        self.rng_key = rng_key
        self._model = None
        self._n_locations = None
        self._params = None
        self._validation_loader = None
        self._locations: list | None = None
        self._validation_locations: list | None = None

    @property
    def covariates(self) -> tuple[str, ...]:
        """The feature covariates in order: the required three then any extras.

        Additional covariates that duplicate a required one are ignored, so the
        required covariates always lead exactly once.
        """
        extra = tuple(c for c in self.additional_covariates if c not in REQUIRED_COVARIATES)
        return REQUIRED_COVARIATES + extra

    @property
    def architecture(self) -> dict:
        """The network hyperparameters, persisted so predict rebuilds the same net."""
        return {
            "cell": self.cell,
            "rnn_features": self.rnn_features,
            "preprocess_hidden": self.preprocess_hidden,
            "preprocess_output": self.preprocess_output,
            "embedding_dim": self.embedding_dim,
            "head_features": self.head_features,
            "rnn_layers": self.rnn_layers,
            "recursive_decode": self.recursive_decode,
            "dropout_rate": self.dropout_rate,
            "input_dropout_rate": self.input_dropout_rate,
        }

    def set_model(self, model: Any) -> None:
        """Override the network instance instead of building one lazily.

        Args:
            model: A flax module to use in place of the default architecture.
        """
        self._model = model

    @property
    def model(self) -> Any:
        """The flax network, built lazily from the architecture on first access."""
        if self._model is None:
            self._model = build_network(self._n_locations, **self.architecture)
        return self._model

    def _get_dataset(self, data: pd.DataFrame) -> DLDataSet:
        """Build a windowed dataset from the input frame."""
        x, y = get_series(data, self.covariates)
        return DLDataSet(x, y, forecast_length=self.prediction_length, context_length=self.context_length)

    def set_validation_data(self, historic: pd.DataFrame, future: pd.DataFrame) -> None:
        """Provide a held-out window for validation-loss reporting during training.

        The supplied history and future are concatenated into a single window and
        wrapped in a loader the trainer evaluates periodically. Unlike forecasting,
        the validation ``future`` must include observed ``disease_cases`` — they
        are the labels the validation loss is computed against.

        The fitted feature scaler is attached later, inside
        [`train`][chap_auto_regressive.model.AutoRegressiveModel.train], so the
        validation features are standardized the same way as the training ones.

        Args:
            historic: History including observed cases.
            future: The matching future periods, with both covariates and observed
                ``disease_cases``.

        Raises:
            ValueError: If ``future`` has no ``disease_cases`` column.
        """
        if "disease_cases" not in future.columns:
            raise ValueError(
                "validation future must include observed disease_cases (they are the "
                "labels the validation loss is computed against)"
            )
        if set(historic["location"].unique()) != set(future["location"].unique()):
            raise ValueError("validation historic and future must cover the same set of locations")
        self._validation_locations = sorted(historic["location"].unique())
        x, y = get_series(historic, self.covariates)
        x = x[:, -self.context_length :]
        y = y[:, -self.context_length :]
        fx, fy = get_series(future, self.covariates)
        full_x = np.concatenate([x, fx], axis=1)
        full_y = np.concatenate([y, fy], axis=1)
        self._validation_loader = SimpleDataLoader(
            DLDataSet(full_x, full_y, forecast_length=self.prediction_length, context_length=self.context_length)
        )

    def _early_stopping_loaders(self, data: pd.DataFrame) -> tuple | None:
        """Split the tail off the training series to select a stopping epoch.

        The split is contiguous and final in time -- a random split would let the
        auto-regressive input see the future -- and the feature scaler is fitted on
        the earlier part only. Returns ``None`` when early stopping is switched
        off or when the caller supplied its own validation window.

        A window spans ``context_length + prediction_length`` periods, and both
        sides of the split need at least one, so early stopping needs at least
        twice that many periods -- 78 months at the default 36 + 3. Too short a
        series raises rather than falling back to fixed-length training: the
        fallback is the overtraining this feature exists to prevent, and it would
        happen silently, since chap runs training in a subprocess whose logs the
        caller never sees.

        Args:
            data: The training frame.

        Raises:
            ValueError: If the series is too short to hold out a validation window.

        Returns:
            A ``(fit_loader, validation_loader)`` pair, or ``None``.
        """
        if not self.early_stopping or self._validation_loader is not None:
            return None
        x, y = get_series(data, self.covariates)
        total_length = self.context_length + self.prediction_length
        # validation_periods is the *extra* history the held-out block gets beyond
        # the single window it must contain, so the knob has an effect at every
        # setting rather than being swallowed by a max() against the window size.
        n_validation = total_length + max(self.validation_periods, 0)
        split = x.shape[1] - n_validation
        if split < total_length:
            raise ValueError(
                f"early stopping needs at least {n_validation + total_length} periods "
                f"(a {total_length}-period window to train on and {n_validation} held out), "
                f"but the shortest series has {x.shape[1]}. Either shorten context_length, "
                f"lower validation_periods (currently {self.validation_periods}), or set "
                f"early_stopping=False to train for a fixed n_iter={self.n_iter}."
            )
        fit_set = DLDataSet(
            x[:, :split], y[:, :split], forecast_length=self.prediction_length, context_length=self.context_length
        )
        scaler = ZScaler.from_data(fit_set)
        fit_set.set_transform(scaler)
        validation_set = DLDataSet(
            x[:, split:], y[:, split:], forecast_length=self.prediction_length, context_length=self.context_length
        )
        validation_set.set_transform(scaler)
        return SimpleDataLoader(fit_set), SimpleDataLoader(validation_set)

    def train(self, data: pd.DataFrame) -> FlaxPredictor:
        """Fit the model and return a predictor.

        Extracts features, fits the feature scaler, builds the windowed loader,
        and runs the [`Trainer`][chap_auto_regressive.trainer.Trainer] to minimize the negative
        log-likelihood.

        Args:
            data: Training frame, one row per location and period, with observed
                ``disease_cases``.

        Returns:
            A [`FlaxPredictor`][chap_auto_regressive.model.FlaxPredictor] holding the trained
            parameters and the fitted scaler.
        """
        self._locations = sorted(data["location"].unique())
        self._n_locations = data["location"].nunique()
        data_set = self._get_dataset(data)
        self._transform = ZScaler.from_data(data_set)
        data_set.set_transform(self._transform)
        early_stopping_loaders = self._early_stopping_loaders(data)
        # Standardize the validation window with the same fitted scaler, otherwise
        # its features stay on the raw scale and the reported validation loss is
        # not comparable to the training loss.
        if self._validation_loader is not None:
            if self._validation_locations != self._locations:
                raise ValueError(
                    "validation locations must match the training locations "
                    f"{self._locations}, but got {self._validation_locations}"
                )
            self._validation_loader.dataset.set_transform(self._transform)
        data_loader = SimpleDataLoader(data_set)
        params_list = []
        for member in range(max(1, self.n_ensemble)):
            seed = self.seed_offset + member
            n_iter = self.n_iter
            if early_stopping_loaders is not None:
                fit_loader, validation_loader = early_stopping_loaders
                probe = Trainer(
                    self.model,
                    self.n_iter,
                    learning_rate=self.learning_rate,
                    validation_loader=validation_loader,
                    seed=seed,
                    eval_every=self.eval_every,
                    patience=self.patience,
                )
                probe.train(fit_loader, self._loss)
                n_iter = max(probe.best_epoch or 0, 1)
                logger.info(
                    "member %d: best validation loss %.4f at epoch %d; refitting on the full series",
                    member,
                    probe.best_validation_loss,
                    n_iter,
                )
            trainer = Trainer(
                self.model,
                n_iter,
                learning_rate=self.learning_rate,
                validation_loader=None if early_stopping_loaders is not None else self._validation_loader,
                seed=seed,
                eval_every=self.eval_every,
                patience=self.patience,
            )
            params_list.append(trainer.train(data_loader, self._loss).params)
        self._params = params_list[0]
        return FlaxPredictor(
            params_list,
            self._transform,
            self.architecture,
            self.prediction_length,
            self.context_length,
            locations=self._locations,
            covariates=self.covariates,
        )

    def load_predictor(self, path: str) -> FlaxPredictor:
        """Load a saved predictor. Self-contained — rebuilt entirely from the file.

        Args:
            path: Path to a file written by
                [`FlaxPredictor.save`][chap_auto_regressive.model.FlaxPredictor.save].

        Returns:
            The reconstructed [`FlaxPredictor`][chap_auto_regressive.model.FlaxPredictor].
        """
        return FlaxPredictor.load(path)

    def predict(self, historic: pd.DataFrame, future: pd.DataFrame, num_samples: int = 100) -> pd.DataFrame:
        """Forecast the future periods directly from this (trained) model.

        Equivalent to the predictor's ``predict``; useful when forecasting from a
        model trained in the same process.

        Args:
            historic: Recent history including observed ``disease_cases``.
            future: The periods to forecast, with covariates but no cases.
            num_samples: Number of samples per location and period.

        Returns:
            A frame with columns ``time_period``, ``location`` and one ``sample_i``
            column per draw.
        """
        _check_predict_inputs(historic, future, self.prediction_length, self.context_length, self._locations)
        x, _ = get_series(future)
        prev_values, prev_y = get_series(historic)
        prev_values = prev_values[:, -self.context_length :]
        prev_y = prev_y[:, -self.context_length :]
        full_x = jnp.concatenate([prev_values, x], axis=1)
        dataset = DLDataSet(full_x, prev_y, forecast_length=self.prediction_length, context_length=self.context_length)
        dataset.set_transform(self._transform)
        x, y = dataset.prediction_instance()
        eta = self.model.apply(self._params, x, y)
        n_prev = prev_values.shape[1]
        samples = self.get_samples(eta[:, n_prev - 1 :], num_samples)
        return _forecast_frame(future, samples)

    def loss_func(self, eta_pred: Any, y_true: Any) -> jnp.ndarray:
        """Per-period negative log-likelihood of the observed cases.

        Args:
            eta_pred: The network output, shape ``(..., periods, 2)``.
            y_true: The window's target; ``y_true[..., 0]`` is the lagged value, so
                the likelihood is evaluated against ``y_true[..., 1:]``.

        Returns:
            The element-wise negative log-likelihood under the negative binomial.
        """
        return -self.distribution_head(eta_pred).log_prob(y_true[..., 1:])

    def _loss(self, y_pred: Any, y_true: Any) -> Any:
        """Aggregate the per-period loss over the whole window, emphasizing the horizon.

        Training on every period (not just the forecast horizon) gives the
        auto-regressive encoder one-step-ahead supervision across the whole
        context, which is far more signal than the 3 horizon periods alone; the
        extra horizon term keeps the forecast region weighted up.
        """
        L = self.loss_func(y_pred, y_true)
        return jnp.mean(L) + jnp.mean(L[:, -self.prediction_length :])

    def get_samples(self, eta: Any, n_samples: int) -> Any:
        """Draw samples from the output distribution for each period.

        Args:
            eta: The network's per-period output, shape ``(..., periods, 2)``.
            n_samples: Number of samples to draw per period.

        Returns:
            Sampled counts, with a leading sample axis of length ``n_samples``.
        """
        self.rng_key, sample_key = jax.random.split(self.rng_key)
        return self.distribution_head(eta).sample(sample_key, (n_samples,))
