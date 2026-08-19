import numpy as np
import pandas as pd
import pytest

from chap_auto_regressive import AutoRegressiveModel
from chap_auto_regressive.transforms import REQUIRED_COVARIATES, get_series


def _small_trained_model():
    train_df = _frame(["A", "B"], [f"2020-{m:02d}" for m in range(1, 13)])
    model = AutoRegressiveModel()
    model.early_stopping = False  # tiny fixtures: exercise IO, not training length
    model.context_length = 4
    model.prediction_length = 2
    model.n_iter = 2
    model.n_ensemble = 1  # keep the fixture fast; ensembling is covered separately
    return model, model.train(train_df), train_df


def _frame(locations, periods, with_target=True, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for index, location in enumerate(locations):
        for period in periods:
            row = {
                "location": location,
                "time_period": period,
                "rainfall": float(rng.rand() * 100),
                "mean_temperature": float(20 + rng.rand() * 10),
                "population": 1000.0 * (index + 1),  # vary by location so the scaler's std > 0
            }
            if with_target:
                row["disease_cases"] = float(rng.poisson(5))
            rows.append(row)
    return pd.DataFrame(rows)


def test_get_series_shapes_and_target():
    months = [f"2020-{m:02d}" for m in range(1, 7)]
    x, y = get_series(_frame(["A", "B"], months))
    assert x.shape == (2, 6, 8)  # 2 locations, 6 periods, 3 covariates + 5 seasonal features
    assert y.shape == (2, 6)


def test_get_series_future_has_no_target():
    months = [f"2020-{m:02d}" for m in range(1, 7)]
    x, y = get_series(_frame(["A", "B"], months, with_target=False))
    assert x.shape == (2, 6, 8)
    assert y.size == 0


def test_train_predict_roundtrip_returns_sample_frame():
    train_months = [f"2020-{m:02d}" for m in range(1, 13)]
    future_months = ["2021-01", "2021-02"]
    train_df = _frame(["A", "B"], train_months)
    future_df = _frame(["A", "B"], future_months, with_target=False, seed=1)

    model = AutoRegressiveModel()
    model.early_stopping = False  # tiny fixtures: exercise IO, not training length
    model.context_length = 4
    model.prediction_length = 2
    model.n_iter = 2
    model.n_ensemble = 1
    predictor = model.train(train_df)

    out = predictor.predict(train_df, future_df, num_samples=10)

    sample_cols = [c for c in out.columns if c.startswith("sample_")]
    assert {"time_period", "location"}.issubset(out.columns)
    assert len(sample_cols) == 10
    assert len(out) == len(["A", "B"]) * len(future_months)
    assert set(out["location"]) == {"A", "B"}
    assert np.isfinite(out[sample_cols].to_numpy()).all()


def test_predict_is_insensitive_to_future_location_row_order():
    # Locations are grouped canonically, so a future frame whose locations appear
    # in a different row order than the history must still be accepted and labeled
    # correctly (the validation used to be order-sensitive and would reject this).
    model, predictor, train_df = _small_trained_model()
    future_months = ["2021-01", "2021-02"]
    future_reversed = _frame(["B", "A"], future_months, with_target=False, seed=1)

    out = predictor.predict(train_df, future_reversed, num_samples=8)

    assert set(out["location"]) == {"A", "B"}
    assert len(out) == 2 * len(future_months)
    assert np.isfinite(out[[c for c in out.columns if c.startswith("sample_")]].to_numpy()).all()


def test_predict_rejects_future_longer_than_horizon():
    # The model can forecast at most prediction_length periods (its trained
    # horizon); asking for more must fail loudly rather than extrapolate silently.
    model, predictor, train_df = _small_trained_model()  # prediction_length = 2
    too_many = _frame(["A", "B"], ["2021-01", "2021-02", "2021-03"], with_target=False, seed=1)

    with pytest.raises(ValueError, match="prediction_length"):
        predictor.predict(train_df, too_many, num_samples=8)


def test_predict_rejects_history_shorter_than_context():
    # The model reads context_length periods of history; a shorter history would
    # silently build a truncated window, so it must be rejected with a clear error.
    model, predictor, train_df = _small_trained_model()  # context_length = 4
    short_history = _frame(["A", "B"], ["2020-11", "2020-12"])  # only 2 periods
    future = _frame(["A", "B"], ["2021-01", "2021-02"], with_target=False, seed=1)

    with pytest.raises(ValueError, match="context_length"):
        predictor.predict(short_history, future, num_samples=8)


def test_predict_accepts_future_shorter_than_horizon():
    # A chap eval backtest forecasts fewer periods than prediction_length; a short
    # future must be accepted and produce one row per location and period.
    model, predictor, train_df = _small_trained_model()  # prediction_length = 2
    short_future = _frame(["A", "B"], ["2021-01"], with_target=False, seed=1)

    out = predictor.predict(train_df, short_future, num_samples=8)

    assert len(out) == 2  # 2 locations x 1 period
    assert sorted(out["time_period"].unique()) == ["2021-01"]
    assert np.isfinite(out[[c for c in out.columns if c.startswith("sample_")]].to_numpy()).all()


def test_predict_labels_each_location_with_its_own_periods():
    # Each location must be labeled with its own forecast periods, not the first
    # location's (the output used to copy group-0's time_period onto every group).
    model, predictor, train_df = _small_trained_model()
    future = pd.concat(
        [
            _frame(["A"], ["2021-01", "2021-02"], with_target=False, seed=1),
            _frame(["B"], ["2021-03", "2021-04"], with_target=False, seed=2),
        ],
        ignore_index=True,
    )

    out = predictor.predict(train_df, future, num_samples=6)

    assert sorted(out[out["location"] == "A"]["time_period"]) == ["2021-01", "2021-02"]
    assert sorted(out[out["location"] == "B"]["time_period"]) == ["2021-03", "2021-04"]


def test_set_validation_data_requires_observed_cases():
    # Validation loss is computed against observed cases, so a future without a
    # disease_cases column must fail with a clear message (not a cryptic concat).
    model = AutoRegressiveModel()
    model.early_stopping = False  # tiny fixtures: exercise IO, not training length
    model.context_length, model.prediction_length = 4, 2
    historic = _frame(["A", "B"], [f"2020-{m:02d}" for m in range(1, 7)])
    future_no_target = _frame(["A", "B"], ["2020-07", "2020-08"], with_target=False, seed=3)

    with pytest.raises(ValueError, match="disease_cases"):
        model.set_validation_data(historic, future_no_target)


def test_predict_rejects_unseen_locations():
    # The network learns per-location embeddings by sorted index; predicting on a
    # location set the model was not trained on would silently reuse another
    # location's embedding, so it must be rejected.
    model, predictor, train_df = _small_trained_model()  # trained on A, B
    unseen_hist = _frame(["C", "D"], [f"2020-{m:02d}" for m in range(1, 13)], seed=9)
    unseen_future = _frame(["C", "D"], ["2021-01", "2021-02"], with_target=False, seed=10)

    with pytest.raises(ValueError, match="training locations"):
        predictor.predict(unseen_hist, unseen_future, num_samples=8)


def test_saved_predictor_preserves_and_enforces_locations(tmp_path):
    # The training locations must survive a save/load round-trip so a reloaded
    # predictor still rejects unseen locations.
    model, predictor, train_df = _small_trained_model()  # trained on A, B
    path = str(tmp_path / "model.pkl")
    predictor.save(path)
    loaded = model.load_predictor(path)

    assert loaded.locations == ["A", "B"]
    out = loaded.predict(train_df, _frame(["A", "B"], ["2021-01", "2021-02"], with_target=False, seed=1), num_samples=4)
    assert set(out["location"]) == {"A", "B"}
    with pytest.raises(ValueError, match="training locations"):
        loaded.predict(
            _frame(["C", "D"], [f"2020-{m:02d}" for m in range(1, 13)], seed=9),
            _frame(["C", "D"], ["2021-01", "2021-02"], with_target=False, seed=10),
            num_samples=4,
        )


def test_validation_features_are_scaled_after_train():
    # The fitted scaler must be applied to the validation window too, otherwise
    # the reported validation loss is computed on raw features and is not
    # comparable to the training loss.
    model = AutoRegressiveModel()
    model.early_stopping = False  # tiny fixtures: exercise IO, not training length
    model.context_length, model.prediction_length, model.n_iter = 4, 2, 2
    model.n_ensemble = 1
    historic = _frame(["A", "B"], [f"2020-{m:02d}" for m in range(1, 7)])
    future = _frame(["A", "B"], ["2020-07", "2020-08"], with_target=True, seed=4)
    model.set_validation_data(historic, future)

    before = model._validation_loader.dataset[0][0]
    model.train(_frame(["A", "B"], [f"2020-{m:02d}" for m in range(1, 13)]))
    after = model._validation_loader.dataset[0][0]

    assert not np.allclose(before, after)  # raw features were standardized in place


def test_ensemble_trains_pools_and_roundtrips(tmp_path):
    # n_ensemble>1 trains several seeded members and pools their samples; the
    # output still has exactly num_samples columns and survives save/load.
    train_df = _frame(["A", "B"], [f"2020-{m:02d}" for m in range(1, 13)])
    future = _frame(["A", "B"], ["2021-01", "2021-02"], with_target=False, seed=1)
    model = AutoRegressiveModel()
    model.early_stopping = False  # tiny fixtures: exercise IO, not training length
    model.context_length, model.prediction_length, model.n_iter = 4, 2, 2
    model.n_ensemble = 3
    predictor = model.train(train_df)
    assert len(predictor._params_list) == 3

    out = predictor.predict(train_df, future, num_samples=12)
    sample_cols = [c for c in out.columns if c.startswith("sample_")]
    assert len(sample_cols) == 12  # pooled to exactly num_samples
    assert np.isfinite(out[sample_cols].to_numpy()).all()

    path = str(tmp_path / "ens.pkl")
    predictor.save(path)
    loaded = model.load_predictor(path)
    assert len(loaded._params_list) == 3


def _add_covariate(df, name="relative_humidity", seed=7):
    """Append a non-NaN extra covariate column to a ``_frame`` dataframe."""
    rng = np.random.RandomState(seed)
    return df.assign(**{name: rng.rand(len(df)) * 100})


def test_covariates_property_appends_additional_after_required():
    # The required covariates always lead, exactly once; an additional name that
    # duplicates a required one is ignored so it is never counted twice.
    model = AutoRegressiveModel()
    model.early_stopping = False  # tiny fixtures: exercise IO, not training length
    assert model.covariates == REQUIRED_COVARIATES  # default: just the required three
    model.additional_covariates = ["relative_humidity", "rainfall"]  # rainfall is required
    assert model.covariates == REQUIRED_COVARIATES + ("relative_humidity",)


def test_get_series_appends_additional_covariate_feature():
    # An extra covariate adds one feature column after the required ones (and the
    # year_position feature still comes last).
    months = [f"2020-{m:02d}" for m in range(1, 7)]
    df = _add_covariate(_frame(["A", "B"], months))
    x, _ = get_series(df, list(REQUIRED_COVARIATES) + ["relative_humidity"])
    assert x.shape == (2, 6, 9)  # 3 required + 1 extra covariate + 5 seasonal features


def test_get_series_missing_covariate_raises():
    months = [f"2020-{m:02d}" for m in range(1, 7)]
    df = _frame(["A", "B"], months)  # no relative_humidity column
    with pytest.raises(ValueError, match="missing covariate column"):
        get_series(df, list(REQUIRED_COVARIATES) + ["relative_humidity"])


def test_train_predict_roundtrip_with_additional_covariate(tmp_path):
    # End-to-end: an additional covariate trains and predicts, survives save/load,
    # and is then required at predict time (dropping it is a clear error).
    train_months = [f"2020-{m:02d}" for m in range(1, 13)]
    future_months = ["2021-01", "2021-02"]
    train_df = _add_covariate(_frame(["A", "B"], train_months))
    future_df = _add_covariate(_frame(["A", "B"], future_months, with_target=False, seed=1))

    model = AutoRegressiveModel()
    model.early_stopping = False  # tiny fixtures: exercise IO, not training length
    model.context_length, model.prediction_length, model.n_iter = 4, 2, 2
    model.n_ensemble = 1
    model.additional_covariates = ["relative_humidity"]
    predictor = model.train(train_df)
    assert predictor.covariates == REQUIRED_COVARIATES + ("relative_humidity",)

    out = predictor.predict(train_df, future_df, num_samples=8)
    assert np.isfinite(out[[c for c in out.columns if c.startswith("sample_")]].to_numpy()).all()

    path = str(tmp_path / "model.pkl")
    predictor.save(path)
    loaded = model.load_predictor(path)
    assert loaded.covariates == REQUIRED_COVARIATES + ("relative_humidity",)
    with pytest.raises(ValueError, match="missing covariate column"):
        loaded.predict(
            train_df.drop(columns="relative_humidity"),
            future_df.drop(columns="relative_humidity"),
            num_samples=4,
        )


def test_saved_predictor_rebuilds_its_architecture(tmp_path):
    # The architecture is persisted, so a predictor saved by one model loads and
    # predicts correctly even when the loading model is configured differently
    # (this is what predict.py does — it has no access to the training config).
    train_df = _frame(["A", "B"], [f"2020-{m:02d}" for m in range(1, 13)])
    future = _frame(["A", "B"], ["2021-01", "2021-02"], with_target=False, seed=1)
    model = AutoRegressiveModel()
    model.early_stopping = False  # tiny fixtures: exercise IO, not training length
    model.context_length, model.prediction_length, model.n_iter, model.n_ensemble = 4, 2, 2, 1
    model.cell, model.rnn_features, model.head_features = "simple", 5, 7  # non-default architecture
    predictor = model.train(train_df)
    path = str(tmp_path / "m.pkl")
    predictor.save(path)

    other = AutoRegressiveModel()  # defaults: gru / 16 / 24
    loaded = other.load_predictor(path)
    assert loaded.architecture["cell"] == "simple"
    assert loaded.architecture["rnn_features"] == 5
    assert loaded.architecture["head_features"] == 7
    out = loaded.predict(train_df, future, num_samples=6)
    assert np.isfinite(out[[c for c in out.columns if c.startswith("sample_")]].to_numpy()).all()


def test_stacked_rnn_layers_train_and_predict():
    # n_layers > 1 stacks encoder/decoder RNNs; it must train and predict, and the
    # choice is persisted in the architecture so predict rebuilds the same depth.
    train_df = _frame(["A", "B"], [f"2020-{m:02d}" for m in range(1, 13)])
    future = _frame(["A", "B"], ["2021-01", "2021-02"], with_target=False, seed=1)
    model = AutoRegressiveModel()
    model.early_stopping = False  # tiny fixtures: exercise IO, not training length
    model.context_length, model.prediction_length, model.n_iter, model.n_ensemble = 4, 2, 2, 1
    model.rnn_layers = 2
    predictor = model.train(train_df)
    assert predictor.architecture["rnn_layers"] == 2
    out = predictor.predict(train_df, future, num_samples=6)
    assert np.isfinite(out[[c for c in out.columns if c.startswith("sample_")]].to_numpy()).all()
