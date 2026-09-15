"""Shared generation helpers for HydroTwin Generator V5."""

import numpy as np


WINDOW_SIZE = 30
CYCLE_SECONDS = 60
SENSOR_COUNT = 17


def build_model_inputs(sensor_windows, phase_windows, input_scaler):
    """Scale 17 sensors and append the known 0..59 cycle position."""
    batch_size = sensor_windows.shape[0]
    scaled_sensors = input_scaler.transform(
        sensor_windows.reshape(-1, SENSOR_COUNT)
    ).reshape(batch_size, WINDOW_SIZE, SENSOR_COUNT)
    scaled_phase = phase_windows.astype(np.float32) / float(CYCLE_SECONDS - 1)
    return np.concatenate(
        [scaled_sensors.astype(np.float32), scaled_phase[:, :, np.newaxis]],
        axis=2,
    )


def generate_from_seed_batch(
    model,
    input_scaler,
    offset_scaler,
    seed_records,
    generate_seconds,
    sensor_min=None,
    sensor_max=None,
    seed_local_bound_indices=None,
):
    """Generate using only each Raw seed's first window and model.predict().

    Every prediction is an offset from the fixed first-second anchor.  This
    prevents recursive delta accumulation, while the phase-aware LSTM keeps
    the measured 60-second shape and reset behavior.
    """
    seed_records = np.asarray(seed_records, dtype=np.float32)
    expected_tail = (WINDOW_SIZE, SENSOR_COUNT)
    if seed_records.ndim != 3 or seed_records.shape[1:] != expected_tail:
        raise ValueError(
            f"Seed shape must be (batch, {WINDOW_SIZE}, {SENSOR_COUNT}), "
            f"actual={seed_records.shape}"
        )

    anchors = seed_records[:, 0, :].copy()
    sensor_windows = seed_records.copy()
    phase_windows = np.tile(
        np.arange(WINDOW_SIZE, dtype=np.int32),
        (seed_records.shape[0], 1),
    )
    generated_steps = []

    for _ in range(generate_seconds):
        model_inputs = build_model_inputs(
            sensor_windows,
            phase_windows,
            input_scaler,
        )
        predicted_offset_scaled = model.predict(model_inputs, verbose=0)
        predicted_offsets = offset_scaler.inverse_transform(
            predicted_offset_scaled
        )
        next_sensors = anchors + predicted_offsets
        next_phases = (phase_windows[:, -1] + 1) % CYCLE_SECONDS

        # Phase 0 is the exact first-second anchor by target definition.
        reset_rows = next_phases == 0
        next_sensors[reset_rows] = anchors[reset_rows]

        # Learned training-range limits prevent physically unseen values.
        if sensor_min is not None and sensor_max is not None:
            next_sensors = np.minimum(
                np.maximum(next_sensors, sensor_min), sensor_max
            )

        # PS4 had almost no common position pattern in the Raw analysis.
        # Keep selected weak-pattern sensors in the range of the Raw seed.
        if seed_local_bound_indices:
            seed_min = seed_records.min(axis=1)
            seed_max = seed_records.max(axis=1)
            for sensor_index in seed_local_bound_indices:
                next_sensors[:, sensor_index] = np.minimum(
                    np.maximum(
                        next_sensors[:, sensor_index], seed_min[:, sensor_index]
                    ),
                    seed_max[:, sensor_index],
                )

        if not np.isfinite(next_sensors).all():
            raise ValueError("V5 generated NaN or Inf")

        generated_steps.append(next_sensors.astype(np.float32))
        sensor_windows = np.concatenate(
            [sensor_windows[:, 1:, :], next_sensors[:, np.newaxis, :]],
            axis=1,
        )
        phase_windows = np.concatenate(
            [phase_windows[:, 1:], next_phases[:, np.newaxis]],
            axis=1,
        )

    return np.stack(generated_steps, axis=1)
