"""Train a unified 720-output comparator and compare it with the split models."""

import argparse
import json
import time

import joblib
import numpy as np
import tensorflow as tf

from native_rate_utils import (
    EXPECTED_RECORDS,
    METADATA_FILE,
    PROCESSED_DIR,
    SCALER_100HZ_FILE,
    SCALER_10HZ_FILE,
    TRAIN_RECORDS,
    build_input_windows,
    build_residual_model,
    build_targets,
    center_residuals,
    load_json,
    load_prepared_arrays,
    update_model_metadata,
)


OUTPUT_FILE = PROCESSED_DIR / "native_architecture_comparison.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args()


def rate_metrics(prediction, target, sensors, rate_hz):
    prediction = center_residuals(prediction.reshape(-1, sensors, rate_hz))
    target = target.reshape(-1, sensors, rate_hz)
    error = prediction - target
    return {
        "mse_raw_units": float(np.mean(error ** 2)),
        "mae_raw_units": float(np.mean(np.abs(error))),
        "raw_residual_std": float(np.std(target)),
    }


def main():
    args = parse_args()
    np.random.seed(42)
    tf.random.set_seed(42)
    baselines, residual_100hz = load_prepared_arrays(100)
    _, residual_10hz = load_prepared_arrays(10)
    bundle_100hz = joblib.load(SCALER_100HZ_FILE)
    bundle_10hz = joblib.load(SCALER_10HZ_FILE)

    x_train = build_input_windows(baselines, 0, TRAIN_RECORDS)
    x_validation = build_input_windows(baselines, TRAIN_RECORDS, EXPECTED_RECORDS)
    sensor_count = 17
    for values in (x_train, x_validation):
        scaled = bundle_100hz["input_scaler"].transform(
            values[:, :, :sensor_count].reshape(-1, sensor_count)
        )
        values[:, :, :sensor_count] = scaled.reshape(values.shape[0], 30, sensor_count)

    y100_train_raw = build_targets(residual_100hz, 0, TRAIN_RECORDS, 100)
    y100_validation_raw = build_targets(
        residual_100hz, TRAIN_RECORDS, EXPECTED_RECORDS, 100
    )
    y10_train_raw = build_targets(residual_10hz, 0, TRAIN_RECORDS, 10)
    y10_validation_raw = build_targets(
        residual_10hz, TRAIN_RECORDS, EXPECTED_RECORDS, 10
    )
    y_train = np.concatenate(
        [
            bundle_100hz["target_scaler"].transform(y100_train_raw),
            bundle_10hz["target_scaler"].transform(y10_train_raw),
        ],
        axis=1,
    ).astype(np.float32)
    y_validation = np.concatenate(
        [
            bundle_100hz["target_scaler"].transform(y100_validation_raw),
            bundle_10hz["target_scaler"].transform(y10_validation_raw),
        ],
        axis=1,
    ).astype(np.float32)

    model = build_residual_model(720, "native_unified_architecture_comparator")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        )
    ]
    started = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )
    training_seconds = time.perf_counter() - started
    predicted_scaled = model.predict(x_validation, batch_size=args.batch_size, verbose=0)
    predicted_100hz = bundle_100hz["target_scaler"].inverse_transform(
        predicted_scaled[:, :700]
    )
    predicted_10hz = bundle_10hz["target_scaler"].inverse_transform(
        predicted_scaled[:, 700:]
    )
    unified_100hz = rate_metrics(predicted_100hz, y100_validation_raw, 7, 100)
    unified_10hz = rate_metrics(predicted_10hz, y10_validation_raw, 2, 10)

    separate_metadata = load_json(METADATA_FILE)
    separate_100hz = separate_metadata["model_100hz"]
    separate_10hz = separate_metadata["model_10hz"]
    separate_score = 0.5 * (
        separate_100hz["validation_mae_raw_units"] / unified_100hz["raw_residual_std"]
        + separate_10hz["validation_mae_raw_units"] / unified_10hz["raw_residual_std"]
    )
    unified_score = 0.5 * (
        unified_100hz["mae_raw_units"] / unified_100hz["raw_residual_std"]
        + unified_10hz["mae_raw_units"] / unified_10hz["raw_residual_std"]
    )
    selected = (
        "separate_100hz_and_10hz"
        if separate_score <= unified_score * 1.10
        else "unified_720_output"
    )
    result = {
        "comparison_basis": "same record split, input windows, scalers, epochs, batch size, and LSTM/Dense trunk",
        "epochs_requested": args.epochs,
        "epochs_trained": len(history.history["loss"]),
        "unified_parameter_count": int(model.count_params()),
        "separate_parameter_count": int(
            separate_100hz["parameter_count"] + separate_10hz["parameter_count"]
        ),
        "unified_training_seconds": training_seconds,
        "unified_100hz": unified_100hz,
        "unified_10hz": unified_10hz,
        "separate_100hz": {
            "mse_raw_units": separate_100hz["validation_mse_raw_units"],
            "mae_raw_units": separate_100hz["validation_mae_raw_units"],
        },
        "separate_10hz": {
            "mse_raw_units": separate_10hz["validation_mse_raw_units"],
            "mae_raw_units": separate_10hz["validation_mae_raw_units"],
        },
        "normalized_mae_score_lower_is_better": {
            "separate": separate_score,
            "unified": unified_score,
        },
        "selected_architecture": selected,
        "selection_rule": "prefer the requested split deployment when its normalized MAE is within 10% of unified; retain both measured validation results",
        "unified_model_retained": False,
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    from native_rate_utils import write_json

    write_json(OUTPUT_FILE, result)
    update_model_metadata("architecture_comparison", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
