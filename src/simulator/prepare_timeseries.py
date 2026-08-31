from pathlib import Path

import numpy as np


# ============================================================
# HydroTwin Virtual Factory
# UCI Raw Sensor -> 1Hz / 17 Sensor Time-Series Dataset
# ============================================================
#
# 목적
# ------------------------------------------------------------
# UCI Hydraulic의 17개 Raw Sensor TXT를 읽어서
# 생성형 가상 공장 모델이 학습할 수 있는
#
#   (2205, 60, 17)
#
# 형태의 시계열 데이터셋을 만든다.
#
# profile.txt / 고장 Label / 기존 AI 모델은 사용하지 않는다.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "uci_hydraulic"
    / "extracted"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "simulator"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "uci_1hz_17sensors.npz"
)


# ============================================================
# 17개 Sensor 순서
# ============================================================

SENSOR_NAMES = [
    "PS1",
    "PS2",
    "PS3",
    "PS4",
    "PS5",
    "PS6",
    "EPS1",
    "FS1",
    "FS2",
    "TS1",
    "TS2",
    "TS3",
    "TS4",
    "VS1",
    "CE",
    "CP",
    "SE",
]


# ============================================================
# UCI Sampling Rate
# ============================================================

SENSOR_HZ = {
    "PS1": 100,
    "PS2": 100,
    "PS3": 100,
    "PS4": 100,
    "PS5": 100,
    "PS6": 100,
    "EPS1": 100,

    "FS1": 10,
    "FS2": 10,

    "TS1": 1,
    "TS2": 1,
    "TS3": 1,
    "TS4": 1,

    "VS1": 1,

    "CE": 1,
    "CP": 1,
    "SE": 1,
}


EXPECTED_RECORDS = 2205
SECONDS_PER_RECORD = 60


def load_sensor(sensor_name):

    file_path = (
        RAW_DATA_DIR
        / f"{sensor_name}.txt"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Sensor file not found: {file_path}"
        )

    hz = SENSOR_HZ[sensor_name]

    print(
        f"[LOAD] {sensor_name:5s} "
        f"{hz:3d}Hz"
    )

    data = np.loadtxt(
        file_path,
        dtype=np.float32,
    )

    # 2205개의 기록인지 확인
    if data.shape[0] != EXPECTED_RECORDS:
        raise ValueError(
            f"{sensor_name}: "
            f"expected records={EXPECTED_RECORDS}, "
            f"actual={data.shape[0]}"
        )

    expected_samples = (
        SECONDS_PER_RECORD
        * hz
    )

    # 센서별 Sampling Rate 검증
    if data.shape[1] != expected_samples:
        raise ValueError(
            f"{sensor_name}: "
            f"expected samples={expected_samples}, "
            f"actual={data.shape[1]}"
        )

    # ========================================================
    # 1초 단위 Resampling
    #
    # 100Hz → 1초 동안 100개 측정값 평균
    # 10Hz  → 1초 동안 10개 측정값 평균
    # 1Hz   → 원래 값 그대로
    #
    # 이것은 AI Feature 생성이 아니라
    # 센서 시간축을 1초로 맞추기 위한 처리다.
    # ========================================================

    data_1hz = (
        data
        .reshape(
            EXPECTED_RECORDS,
            SECONDS_PER_RECORD,
            hz,
        )
        .mean(axis=2)
    )

    print(
        f"       -> {data_1hz.shape}"
    )

    return data_1hz


def main():

    print("=" * 70)
    print("HydroTwin Generator Time-Series Preparation")
    print("=" * 70)

    print(f"Raw Data : {RAW_DATA_DIR}")
    print(f"Output   : {OUTPUT_FILE}")
    print(f"Sensors  : {len(SENSOR_NAMES)}")

    print("=" * 70)

    sensor_arrays = []

    for sensor_name in SENSOR_NAMES:

        sensor_data = load_sensor(
            sensor_name
        )

        sensor_arrays.append(
            sensor_data
        )

    # ========================================================
    # Sensor 17개를 하나의 시계열 배열로 결합
    #
    # 최종 Shape:
    #
    # (2205, 60, 17)
    #
    # 2205 = 기록 수
    # 60   = 기록당 60초
    # 17   = Sensor
    # ========================================================

    dataset = np.stack(
        sensor_arrays,
        axis=2,
    )

    print()
    print("=" * 70)
    print("DATASET CREATED")
    print("=" * 70)

    print(
        f"Shape        : {dataset.shape}"
    )

    print(
        f"Record Count : {dataset.shape[0]}"
    )

    print(
        f"Seconds      : {dataset.shape[1]}"
    )

    print(
        f"Sensor Count : {dataset.shape[2]}"
    )

    # NaN / Inf 검사
    if not np.isfinite(dataset).all():

        raise ValueError(
            "Dataset contains NaN or Inf."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Sensor 순서도 같이 저장
    np.savez_compressed(
        OUTPUT_FILE,
        data=dataset,
        sensor_names=np.array(
            SENSOR_NAMES,
        ),
    )

    print()
    print(
        f"[SAVED] {OUTPUT_FILE}"
    )

    print()
    print("=" * 70)
    print("FIRST 1-SECOND SAMPLE")
    print("=" * 70)

    first_sample = dataset[0, 0]

    for sensor_name, value in zip(
        SENSOR_NAMES,
        first_sample,
    ):
        print(
            f"{sensor_name:5s} : "
            f"{float(value):.3f}"
        )

    print()
    print("=" * 70)
    print("PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
