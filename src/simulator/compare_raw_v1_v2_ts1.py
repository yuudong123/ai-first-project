
from pathlib import Path

import joblib
import matplotlib

# Ubuntu Server는 GUI가 없으므로
# 파일 저장 전용 Backend 사용
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# ============================================================
# HydroTwin
# Raw vs V1 vs V2 - TS1 Time-Series Comparison
# ============================================================
#
# 비교 방법
# ------------------------------------------------------------
#
# UCI Validation Record 하나를 사용한다.
#
# 실제 1~30초:
#   V1 / V2 Generator의 시작 데이터로 사용
#
# 실제 31~60초:
#   정답 비교 구간
#
# V1:
#   최근 20초를 사용해 31~60초 생성
#
# V2:
#   최근 30초를 사용해 31~60초 생성
#
#
# 중요
# ------------------------------------------------------------
# model.fit() 없음
# 재학습 없음
# 생성 데이터를 학습에 사용하지 않음
#
# 오직 저장된 모델의 predict()만 사용
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "simulator"
    / "uci_1hz_17sensors.npz"
)


MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "simulator"
)


# ============================================================
# V1
# ============================================================

V1_MODEL_FILE = (
    MODEL_DIR
    / "virtual_factory_generator_v1.keras"
)

V1_SCALER_FILE = (
    MODEL_DIR
    / "sensor_scaler_v1.joblib"
)


# ============================================================
# V2
# ============================================================

V2_MODEL_FILE = (
    MODEL_DIR
    / "virtual_factory_generator_v2.keras"
)

V2_SCALER_FILE = (
    MODEL_DIR
    / "sensor_scaler_v2.joblib"
)


# ============================================================
# 그래프 저장 위치
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "simulator"
    / "graphs"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "raw_v1_v2_ts1_31_60s.png"
)


# ============================================================
# 설정
# ============================================================

V1_WINDOW = 20
V2_WINDOW = 30

GENERATE_SECONDS = 30

TRAIN_RATIO = 0.8


# ============================================================
# 연속 생성 함수
# ============================================================

def generate_future(
    model,
    scaler,
    seed_raw,
    generate_seconds,
):

    # Seed를 모델이 학습했던 Scale로 변환
    current_window = (
        scaler
        .transform(seed_raw)
        .astype(np.float32)
    )

    generated_rows = []


    for _ in range(generate_seconds):

        input_batch = (
            current_window[
                np.newaxis,
                ...
            ]
        )


        # ----------------------------------------------------
        # predict만 사용
        # 재학습 없음
        # ----------------------------------------------------

        next_scaled = (
            model.predict(
                input_batch,
                verbose=0,
            )[0]
        )


        # 실제 Sensor 단위로 복원
        next_raw = (
            scaler.inverse_transform(
                next_scaled.reshape(
                    1,
                    -1,
                )
            )[0]
        )


        generated_rows.append(
            next_raw.copy()
        )


        # 가장 오래된 1초 제거
        # +
        # 방금 생성한 1초 추가
        current_window = np.vstack(
            [
                current_window[1:],
                next_scaled,
            ]
        )


    return np.asarray(
        generated_rows,
        dtype=np.float32,
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("HydroTwin Raw vs V1 vs V2 - TS1 Comparison")
    print("=" * 80)


    # --------------------------------------------------------
    # UCI Raw 데이터
    # --------------------------------------------------------

    dataset = np.load(
        DATA_FILE
    )


    data = (
        dataset["data"]
        .astype(np.float32)
    )


    sensor_names = (
        dataset["sensor_names"]
        .astype(str)
        .tolist()
    )


    print(
        f"Dataset Shape : {data.shape}"
    )


    # --------------------------------------------------------
    # TS1 위치 확인
    # --------------------------------------------------------

    ts1_index = (
        sensor_names.index("TS1")
    )


    print(
        f"TS1 Index     : {ts1_index}"
    )


    # --------------------------------------------------------
    # Validation 첫 Record 선택
    #
    # 학습에 사용한 앞쪽 80%가 아니라
    # Validation 영역 첫 Record 사용
    # --------------------------------------------------------

    validation_start = int(
        data.shape[0]
        * TRAIN_RATIO
    )


    record_index = (
        validation_start
    )


    raw_record = (
        data[
            record_index
        ]
        .copy()
    )


    print(
        f"Record Index  : {record_index}"
    )

    print(
        f"Record Shape  : {raw_record.shape}"
    )


    # ========================================================
    # 실제 비교 대상
    #
    # Raw 31~60초
    #
    # Python Index:
    # 30~59
    # ========================================================

    actual_future = (
        raw_record[
            30:60
        ]
    )


    # ========================================================
    # V1 Seed
    #
    # V1은 20초 Window
    #
    # 동일한 1~30초 정보를 기준으로 하기 위해
    # 마지막 20초인 11~30초 사용
    # ========================================================

    v1_seed = (
        raw_record[
            10:30
        ]
    )


    # ========================================================
    # V2 Seed
    #
    # V2는 30초 Window
    #
    # 실제 1~30초 전체 사용
    # ========================================================

    v2_seed = (
        raw_record[
            0:30
        ]
    )


    print()
    print(
        f"V1 Seed Shape : {v1_seed.shape}"
    )

    print(
        f"V2 Seed Shape : {v2_seed.shape}"
    )


    # ========================================================
    # 모델 로드
    #
    # compile=False
    # → 학습하지 않고 predict만 사용
    # ========================================================

    print()
    print("Loading V1...")

    v1_model = (
        tf.keras.models.load_model(
            V1_MODEL_FILE,
            compile=False,
        )
    )

    v1_scaler = joblib.load(
        V1_SCALER_FILE
    )


    print("Loading V2...")

    v2_model = (
        tf.keras.models.load_model(
            V2_MODEL_FILE,
            compile=False,
        )
    )

    v2_scaler = joblib.load(
        V2_SCALER_FILE
    )


    # ========================================================
    # 31~60초 생성
    # ========================================================

    print()
    print("Generating V1 31~60 sec...")

    v1_generated = generate_future(
        model=v1_model,
        scaler=v1_scaler,
        seed_raw=v1_seed,
        generate_seconds=GENERATE_SECONDS,
    )


    print("Generating V2 31~60 sec...")

    v2_generated = generate_future(
        model=v2_model,
        scaler=v2_scaler,
        seed_raw=v2_seed,
        generate_seconds=GENERATE_SECONDS,
    )


    # ========================================================
    # TS1만 추출
    # ========================================================

    raw_ts1 = (
        actual_future[
            :,
            ts1_index
        ]
    )


    v1_ts1 = (
        v1_generated[
            :,
            ts1_index
        ]
    )


    v2_ts1 = (
        v2_generated[
            :,
            ts1_index
        ]
    )


    # ========================================================
    # 수업에서 배운 MSE / MAE
    # ========================================================

    v1_mse = np.mean(
        (
            raw_ts1
            - v1_ts1
        ) ** 2
    )


    v2_mse = np.mean(
        (
            raw_ts1
            - v2_ts1
        ) ** 2
    )


    v1_mae = np.mean(
        np.abs(
            raw_ts1
            - v1_ts1
        )
    )


    v2_mae = np.mean(
        np.abs(
            raw_ts1
            - v2_ts1
        )
    )


    print()
    print("=" * 80)
    print("TS1 RESULT")
    print("=" * 80)

    print(
        f"V1 MSE : {v1_mse:.6f}"
    )

    print(
        f"V1 MAE : {v1_mae:.6f}"
    )

    print()

    print(
        f"V2 MSE : {v2_mse:.6f}"
    )

    print(
        f"V2 MAE : {v2_mae:.6f}"
    )


    # ========================================================
    # Line Plot
    #
    # 실제 31~60초
    # V1 생성 31~60초
    # V2 생성 31~60초
    # ========================================================

    seconds = np.arange(
        31,
        61,
    )


    plt.figure(
        figsize=(14, 6)
    )


    plt.plot(
        seconds,
        raw_ts1,
        marker="o",
        label="Raw TS1",
    )


    plt.plot(
        seconds,
        v1_ts1,
        marker="o",
        label="V1 Generated",
    )


    plt.plot(
        seconds,
        v2_ts1,
        marker="o",
        label="V2 Generated",
    )


    plt.title(
        "HydroTwin TS1 - Raw vs V1 vs V2"
    )


    plt.xlabel(
        "Time (sec)"
    )


    plt.ylabel(
        "TS1"
    )


    plt.legend()


    plt.grid(
        True
    )


    plt.tight_layout()


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    plt.savefig(
        OUTPUT_FILE,
        dpi=150,
    )


    plt.close()


    print()
    print(
        f"[SAVED] {OUTPUT_FILE}"
    )


    print()
    print("=" * 80)
    print("COMPARISON PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
