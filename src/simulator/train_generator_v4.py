import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
)


# ============================================================
# HydroTwin Virtual Factory Generator V4
# ============================================================
#
# 목적
# ------------------------------------------------------------
# V1~V3는 다음 Sensor 값 자체를 예측했다.
#
# V4는:
#
# 최근 30초 x 17 Sensor
#          ↓
#         LSTM
#          ↓
# 다음 1초의 Sensor 변화량(Delta)
#
# 을 학습한다.
#
#
# 다음 Sensor:
#
# 현재 Sensor + 예측 Delta
#
#
# 사용하는 기술
# ------------------------------------------------------------
# - NumPy
# - StandardScaler
# - Sliding Window
# - LSTM
# - Dense
# - MSE
# - MAE
#
#
# 학습 데이터
# ------------------------------------------------------------
# UCI Raw 17 Sensor만 사용
#
#
# 절대 사용하지 않음
# ------------------------------------------------------------
# - generated_300s_v1.csv
# - generated_300s_v2.csv
# - generated_300s_v3.csv
# - 생성 데이터
# - profile.txt
# - 고장 Label
# - 기존 상태예측 모델
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


MODEL_FILE = (
    MODEL_DIR
    / "virtual_factory_generator_v4.keras"
)


INPUT_SCALER_FILE = (
    MODEL_DIR
    / "input_scaler_v4.joblib"
)


DELTA_SCALER_FILE = (
    MODEL_DIR
    / "delta_scaler_v4.joblib"
)


METADATA_FILE = (
    MODEL_DIR
    / "generator_metadata_v4.json"
)


# ============================================================
# V4 설정
# ============================================================

WINDOW_SIZE = 30

TRAIN_RATIO = 0.8

EPOCHS = 50

BATCH_SIZE = 256

RANDOM_SEED = 42


np.random.seed(
    RANDOM_SEED
)

tf.random.set_seed(
    RANDOM_SEED
)


# ============================================================
# Sliding Window 생성
# ============================================================
#
# Input:
#
# 0~29초 Sensor
#
# Target:
#
# 30초 Sensor - 29초 Sensor
#
# 즉 다음 순간의 "변화량"을 학습한다.
#
#
# 서로 다른 60초 Record는 연결하지 않는다.
# ============================================================


def create_windows(
    records_raw,
    records_scaled,
    delta_scaler,
):

    x_list = []
    y_list = []


    for (
        record_raw,
        record_scaled,
    ) in zip(
        records_raw,
        records_scaled,
    ):

        seconds = (
            record_raw.shape[0]
        )


        for start in range(
            seconds - WINDOW_SIZE
        ):

            end = (
                start
                + WINDOW_SIZE
            )


            # 최근 30초 Sensor
            x = (
                record_scaled[
                    start:end
                ]
            )


            # 다음 1초 변화량
            delta_raw = (
                record_raw[end]
                - record_raw[end - 1]
            )


            delta_scaled = (
                delta_scaler
                .transform(
                    delta_raw.reshape(
                        1,
                        -1,
                    )
                )[0]
            )


            x_list.append(
                x
            )


            y_list.append(
                delta_scaled
            )


    return (
        np.asarray(
            x_list,
            dtype=np.float32,
        ),
        np.asarray(
            y_list,
            dtype=np.float32,
        ),
    )


# ============================================================
# Main
# ============================================================


def main():

    print("=" * 75)

    print(
        "HydroTwin Virtual Factory "
        "LSTM Generator V4 - Delta Prediction"
    )

    print("=" * 75)


    # --------------------------------------------------------
    # UCI Raw 기반 1Hz Dataset
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


    sensor_count = (
        data.shape[2]
    )


    print(
        f"Dataset Shape : "
        f"{data.shape}"
    )

    print(
        f"Sensor Count  : "
        f"{sensor_count}"
    )

    print(
        f"Window Size   : "
        f"{WINDOW_SIZE} sec"
    )

    print(
        "Target        : "
        "Next 1-second Sensor Delta"
    )


    # ========================================================
    # Record 단위 Train / Validation
    # ========================================================

    record_count = (
        data.shape[0]
    )


    train_count = int(
        record_count
        * TRAIN_RATIO
    )


    train_raw = (
        data[:train_count]
    )


    val_raw = (
        data[train_count:]
    )


    print()

    print(
        f"Train Records : "
        f"{train_raw.shape[0]}"
    )

    print(
        f"Val Records   : "
        f"{val_raw.shape[0]}"
    )


    # ========================================================
    # Input StandardScaler
    #
    # 현재 Sensor 값 Scale
    # ========================================================

    input_scaler = (
        StandardScaler()
    )


    input_scaler.fit(
        train_raw.reshape(
            -1,
            sensor_count,
        )
    )


    train_scaled = (
        input_scaler
        .transform(
            train_raw.reshape(
                -1,
                sensor_count,
            )
        )
        .reshape(
            train_raw.shape
        )
        .astype(np.float32)
    )


    val_scaled = (
        input_scaler
        .transform(
            val_raw.reshape(
                -1,
                sensor_count,
            )
        )
        .reshape(
            val_raw.shape
        )
        .astype(np.float32)
    )


    # ========================================================
    # Delta 데이터 생성
    #
    # 각 Record 내부에서만:
    #
    # t+1 - t
    #
    # ========================================================

    train_deltas = (
        train_raw[:, 1:, :]
        - train_raw[:, :-1, :]
    )


    delta_scaler = (
        StandardScaler()
    )


    delta_scaler.fit(
        train_deltas.reshape(
            -1,
            sensor_count,
        )
    )


    # ========================================================
    # Sliding Window
    # ========================================================

    print()

    print(
        "Creating V4 Delta windows..."
    )


    x_train, y_train = (
        create_windows(
            train_raw,
            train_scaled,
            delta_scaler,
        )
    )


    x_val, y_val = (
        create_windows(
            val_raw,
            val_scaled,
            delta_scaler,
        )
    )


    print(
        f"X Train : "
        f"{x_train.shape}"
    )

    print(
        f"Y Train : "
        f"{y_train.shape}"
    )

    print(
        f"X Val   : "
        f"{x_val.shape}"
    )

    print(
        f"Y Val   : "
        f"{y_val.shape}"
    )


    # ========================================================
    # LSTM
    #
    # V2와 같은 기본 구조
    #
    # 차이:
    # 다음 Sensor 값이 아니라
    # 다음 Sensor 변화량을 출력
    # ========================================================

    model = Sequential(
        [
            Input(
                shape=(
                    WINDOW_SIZE,
                    sensor_count,
                )
            ),

            LSTM(
                64
            ),

            Dense(
                64,
                activation="relu",
            ),

            Dense(
                sensor_count
            ),
        ]
    )


    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[
            "mae",
        ],
    )


    print()
    print("=" * 75)
    print("V4 MODEL")
    print("=" * 75)

    model.summary()


    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # Callback
    # ========================================================

    early_stopping = (
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        )
    )


    checkpoint = (
        ModelCheckpoint(
            MODEL_FILE,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        )
    )


    # ========================================================
    # 학습
    # ========================================================

    print()
    print("=" * 75)
    print("TRAINING V4 START")
    print("=" * 75)


    model.fit(
        x_train,
        y_train,

        validation_data=(
            x_val,
            y_val,
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=[
            early_stopping,
            checkpoint,
        ],

        verbose=1,
    )


    # ========================================================
    # Validation
    # ========================================================

    val_loss, val_mae = (
        model.evaluate(
            x_val,
            y_val,
            verbose=0,
        )
    )


    print()
    print("=" * 75)
    print("VALIDATION V4")
    print("=" * 75)


    print(
        f"Validation MSE : "
        f"{val_loss:.6f}"
    )


    print(
        f"Validation MAE : "
        f"{val_mae:.6f}"
    )


    # ========================================================
    # 저장
    # ========================================================

    joblib.dump(
        input_scaler,
        INPUT_SCALER_FILE,
    )


    joblib.dump(
        delta_scaler,
        DELTA_SCALER_FILE,
    )


    metadata = {

        "version": "v4",

        "sensor_names":
            sensor_names,

        "sensor_count":
            sensor_count,

        "window_size":
            WINDOW_SIZE,

        "target":
            "next_1_second_delta",

        "training_records":
            int(
                train_raw.shape[0]
            ),

        "validation_records":
            int(
                val_raw.shape[0]
            ),

        "training_source":
            "UCI Raw 17 Sensors Only",

        "generated_data_used_for_training":
            False,
    }


    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )


    model.save(
        MODEL_FILE
    )


    print(
        f"[SAVED] {MODEL_FILE}"
    )

    print(
        f"[SAVED] {INPUT_SCALER_FILE}"
    )

    print(
        f"[SAVED] {DELTA_SCALER_FILE}"
    )

    print(
        f"[SAVED] {METADATA_FILE}"
    )


    # ========================================================
    # 실제 다음 1초 Sample
    # ========================================================

    predicted_delta_scaled = (
        model.predict(
            x_val[:1],
            verbose=0,
        )
    )


    predicted_delta = (
        delta_scaler
        .inverse_transform(
            predicted_delta_scaled
        )[0]
    )


    # Validation 첫 Window
    # 30초째 Sensor
    current_raw = (
        val_raw[
            0,
            WINDOW_SIZE - 1,
            :
        ]
    )


    # 실제 다음 Sensor
    actual_next = (
        val_raw[
            0,
            WINDOW_SIZE,
            :
        ]
    )


    # V4 생성 Sensor
    generated_next = (
        current_raw
        + predicted_delta
    )


    print()
    print("=" * 75)
    print("SAMPLE NEXT-SECOND V4")
    print("=" * 75)


    for (
        sensor,
        actual_value,
        generated_value,
    ) in zip(
        sensor_names,
        actual_next,
        generated_next,
    ):

        print(
            f"{sensor:5s} "
            f"actual={actual_value:10.3f} "
            f"generated={generated_value:10.3f}"
        )


    print()
    print("=" * 75)
    print("TRAINING V4 PASS")
    print("=" * 75)


if __name__ == "__main__":
    main()
