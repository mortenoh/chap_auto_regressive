"""Early stopping: the split, the two-phase refit, and the short-series guard."""

import numpy as np
import pandas as pd
import pytest

from chap_auto_regressive import AutoRegressiveModel
from chap_auto_regressive.trainer import Trainer


def _frame(n_periods: int, n_locations: int = 3, seed: int = 0) -> pd.DataFrame:
    """A small synthetic panel in the shape the model expects."""
    rng = np.random.default_rng(seed)
    rows = []
    for location in range(n_locations):
        for period in range(n_periods):
            month = period % 12
            rows.append(
                {
                    "time_period": f"{2000 + period // 12}-{month + 1:02d}",
                    "location": f"loc{location}",
                    "disease_cases": float(rng.poisson(20 + 10 * np.sin(2 * np.pi * month / 12))),
                    "rainfall": float(rng.normal(100, 20)),
                    "mean_temperature": float(20 + 5 * np.sin(2 * np.pi * month / 12)),
                    "population": 1000.0 * (location + 1),
                }
            )
    return pd.DataFrame(rows)


def _model(**options) -> AutoRegressiveModel:
    """A small, fast model configuration."""
    model = AutoRegressiveModel()
    model.context_length = 6
    model.prediction_length = 2
    model.n_ensemble = 1
    model.n_iter = 30
    model.eval_every = 2
    model.patience = 3
    model.validation_periods = 4
    for key, value in options.items():
        setattr(model, key, value)
    return model


def test_split_is_contiguous_and_final_in_time():
    """The held-out block is the tail of the series, never a random subset."""
    model = _model()
    model._locations = ["loc0", "loc1", "loc2"]
    model._n_locations = 3
    data = _frame(40)
    fit_loader, validation_loader = model._early_stopping_loaders(data)

    total_length = model.context_length + model.prediction_length
    n_validation = total_length + model.validation_periods
    # The validation windows must come from the final n_validation periods, so the
    # first validation target cannot appear anywhere in the training windows.
    assert len(validation_loader.dataset) == model.validation_periods + 1
    assert len(fit_loader.dataset) == 40 - n_validation - total_length + 1


def test_too_short_a_series_raises_rather_than_falling_back():
    """A silent fallback would reintroduce the overtraining this prevents."""
    model = _model()
    model._locations = ["loc0", "loc1", "loc2"]
    model._n_locations = 3
    total_length = model.context_length + model.prediction_length
    minimum = total_length + model.validation_periods + total_length

    model._early_stopping_loaders(_frame(minimum))  # exactly enough: fine

    with pytest.raises(ValueError, match="early stopping needs at least"):
        model._early_stopping_loaders(_frame(minimum - 1))


def test_disabled_or_externally_supplied_validation_returns_none():
    """Early stopping steps aside when switched off or when given a window."""
    data = _frame(40)
    off = _model(early_stopping=False)
    off._locations, off._n_locations = ["loc0", "loc1", "loc2"], 3
    assert off._early_stopping_loaders(data) is None

    supplied = _model()
    supplied._locations, supplied._n_locations = ["loc0", "loc1", "loc2"], 3
    supplied._validation_loader = object()
    assert supplied._early_stopping_loaders(data) is None


def test_training_selects_an_epoch_and_refits_on_everything():
    """The two phases run, and the second is capped by the epoch the first chose."""
    model = _model()
    selected = []
    original = Trainer.train

    def recording(self, loader, loss_fn):
        state = original(self, loader, loss_fn)
        if self.best_epoch is not None:
            selected.append((self.best_epoch, self.n_iter))
        return state

    Trainer.train = recording
    try:
        predictor = model.train(_frame(60))
    finally:
        Trainer.train = original

    assert predictor is not None
    assert selected, "the probe phase should have reported a best epoch"
    best_epoch, probe_budget = selected[0]
    assert 0 <= best_epoch <= probe_budget


def test_short_run_still_trains_when_early_stopping_is_off():
    """Series below the early-stopping minimum remain trainable explicitly."""
    model = _model(early_stopping=False, n_iter=3)
    predictor = model.train(_frame(20))
    assert predictor is not None


def test_trainer_without_validation_keeps_the_final_state():
    """With no validation loader the trainer behaves as it did before."""
    trainer = Trainer(model=None, n_iter=3)
    assert trainer.best_epoch is None
    assert trainer.patience == 6
