"""Validate 300 V5+native batches and measure runtime, CPU, memory, and size."""

import argparse
import json
import resource
import time

import numpy as np

from native_rate_utils import (
    PROCESSED_DIR,
    SENSOR_NAMES,
    SENSORS_100HZ,
    SENSORS_10HZ,
    NativeRateRuntime,
    V5BaselineRuntime,
    create_native_message,
    load_json,
    DATA_METADATA_FILE,
    write_json,
)


GENERATED_FILE = PROCESSED_DIR / "generated_native_300s.npz"
PERFORMANCE_FILE = PROCESSED_DIR / "native_performance_300s.json"


def current_rss_mb():
    with open("/proc/self/status", "r", encoding="utf-8") as status_file:
        for line in status_file:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return float("nan")


def summary(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
        "min": float(array.min()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--seed-record", type=int, default=1764)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seconds <= 0:
        raise ValueError("--seconds must be positive")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data_metadata = load_json(DATA_METADATA_FILE)
    raw_min = np.asarray(
        [data_metadata["sensor_statistics"][sensor]["raw_min"] for sensor in SENSOR_NAMES]
    )
    raw_max = np.asarray(
        [data_metadata["sensor_statistics"][sensor]["raw_max"] for sensor in SENSOR_NAMES]
    )

    load_started = time.perf_counter()
    v5_runtime = V5BaselineRuntime(args.seed_record)
    native_runtime = NativeRateRuntime()
    model_load_seconds = time.perf_counter() - load_started

    # Exclude TensorFlow graph warm-up from steady-state one-second timing.
    warm_context, warm_phases = v5_runtime.context()
    warm_baseline = v5_runtime.predict_next()
    native_runtime.generate(warm_baseline, warm_context, warm_phases)

    baselines = np.empty((args.seconds, len(SENSOR_NAMES)), dtype=np.float64)
    generated_100hz = np.empty(
        (args.seconds, len(SENSORS_100HZ), 100), dtype=np.float64
    )
    generated_10hz = np.empty(
        (args.seconds, len(SENSORS_10HZ), 10), dtype=np.float64
    )
    v5_timings = []
    native_100hz_timings = []
    native_10hz_timings = []
    serialization_timings = []
    total_timings = []
    message_sizes = []
    mean_errors = []
    rss_samples = [current_rss_mb()]
    safety_violations = []

    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    for second in range(args.seconds):
        total_started = time.perf_counter()
        baseline_window, phase_window = v5_runtime.context()

        v5_started = time.perf_counter()
        baseline = v5_runtime.predict_next()
        v5_timings.append(time.perf_counter() - v5_started)

        sensors, timings = native_runtime.generate(
            baseline, baseline_window, phase_window
        )
        native_100hz_timings.append(timings["native_100hz_inference_seconds"])
        native_10hz_timings.append(timings["native_10hz_inference_seconds"])

        serialization_started = time.perf_counter()
        message = create_native_message(sensors)
        serialized = json.dumps(
            message, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        serialization_timings.append(time.perf_counter() - serialization_started)
        message_sizes.append(len(serialized))
        total_timings.append(time.perf_counter() - total_started)

        baselines[second] = baseline
        for index, sensor in enumerate(SENSORS_100HZ):
            values = np.asarray(message["sensors"][sensor], dtype=np.float64)
            generated_100hz[second, index] = values
            mean_errors.append(abs(float(values.mean()) - baseline[index]))
            if values.min() < raw_min[index] - 1e-6 or values.max() > raw_max[index] + 1e-6:
                safety_violations.append(f"second={second} sensor={sensor} raw_range")
        offset = len(SENSORS_100HZ)
        for index, sensor in enumerate(SENSORS_10HZ):
            values = np.asarray(message["sensors"][sensor], dtype=np.float64)
            generated_10hz[second, index] = values
            mean_errors.append(abs(float(values.mean()) - baseline[offset + index]))
            if values.min() < raw_min[offset + index] - 1e-6 or values.max() > raw_max[offset + index] + 1e-6:
                safety_violations.append(f"second={second} sensor={sensor} raw_range")
        rss_samples.append(current_rss_mb())
        if second == 0 or (second + 1) % 50 == 0:
            print(
                f"[batch {second + 1:3d}/{args.seconds}] "
                f"total={total_timings[-1] * 1000:.3f}ms "
                f"size={message_sizes[-1]}B",
                flush=True,
            )

    cpu_seconds = time.process_time() - cpu_started
    wall_seconds = time.perf_counter() - wall_started
    max_rss_mb = max(
        max(rss_samples), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    )
    mean_error_max = float(max(mean_errors, default=0.0))
    batch_limit_pass = max(total_timings) < 1.0
    result = {
        "status": "PASS" if batch_limit_pass and not safety_violations else "FAIL",
        "equivalent_generated_seconds": args.seconds,
        "warmup_excluded": True,
        "seed_record": args.seed_record,
        "model_load_seconds": model_load_seconds,
        "v5_baseline_inference_seconds": summary(v5_timings),
        "native_100hz_inference_seconds": summary(native_100hz_timings),
        "native_10hz_inference_seconds": summary(native_10hz_timings),
        "kafka_json_serialize_seconds": summary(serialization_timings),
        "total_batch_seconds": summary(total_timings),
        "under_one_second": batch_limit_pass,
        "kafka_message_bytes": summary(message_sizes),
        "samples_per_batch": 728,
        "max_native_mean_vs_v5_baseline_abs_error": mean_error_max,
        "nan_count": int(
            np.isnan(generated_100hz).sum() + np.isnan(generated_10hz).sum()
        ),
        "inf_count": int(
            np.isinf(generated_100hz).sum() + np.isinf(generated_10hz).sum()
        ),
        "raw_range_violations": safety_violations,
        "wall_seconds": wall_seconds,
        "process_cpu_seconds": cpu_seconds,
        "process_cpu_percent_of_one_core": 100.0 * cpu_seconds / wall_seconds,
        "rss_start_mb": rss_samples[0],
        "rss_end_mb": rss_samples[-1],
        "rss_peak_mb": max_rss_mb,
    }
    np.savez_compressed(
        GENERATED_FILE,
        sensor_names=np.asarray(SENSOR_NAMES),
        baselines=baselines,
        samples_100hz=generated_100hz,
        samples_10hz=generated_10hz,
    )
    write_json(PERFORMANCE_FILE, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
