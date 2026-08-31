import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, LSTM, Dense


# ============================================================
# HydroTwin Virtual Factory Generator V3
# ============================================================
#
# V3 목표
# ------------------------------------------------------------
# 최근 30초의 17개 Sensor를 보고
# 다음 10초의 17개 Sensor를 한 번에 생성한다.
#
#
# V1
# 20초 -> 다음 1초
#
# V2
# 30초 -> 다음 1초
#
# V3
# 30초 -> 다음 10초
#
#
# 사용하는 기술
# ------------------------------------------------------------
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
# UCI Raw 17 Sensor -> 1Hz 데이터만 사용
#
#
# 절대 사용하지 않음
# ------------------------------------------------------------
# - generated_300s_v1.csv
# - generated_300s_v2.csv
# - V1/V2 생성 데이터
# - profile.txt
# - 고장 Label
# - 기존 상태예측 model.pkl
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
    / "virtual_factory_generator_v3.keras"
)


SCALER_FILE = (
    MODEL_DIR
    / "sensor_scaler_v3.joblib"
)


METADATA_FILE = (
    MODEL_DIR
    / "generator_metadata_v3.json"
)


# ============================================================
# V3 설정
# ============================================================

INPUT_WINDOW = 30

OUTPUT_STEPS = 10

TRAIN_RATIO = 0.8

EPOCHS = 50

BATCH_SIZE = 256

RANDOM_SEED = 42


np.random.seed(RANDOM_SEED)

tf.random.set_seed(RANDOM_SEED)


# ============================================================
# Sliding Window
#
# 각 UCI 60초 Record 안에서만 생성
#
# 예)
#
# 0~29초
#     ↓
# 30~39초 10개 Sensor 시점
#
# 1~30초
#     ↓
# 31~40초
#
# ...
#
# 서로 다른 Record는 연결하지 않는다.
# ============================================================


def create_windows(
    records,
    input_window,
    output_steps,
):

    x_list = []

    y_list = []


    for record in records:

        seconds = record.shape[0]

        max_start = (
            seconds
            - input_window
            - output_steps
            + 1
        )


        for start in range(
            max_start
        ):

            input_end = (
                start
                + input_window
            )


            output_end = (
                input_end
                + output_steps
            )


            x = record[
                start:input_end
            ]


            y = record[
                input_end:output_end
            ]


            x_list.append(x)

            # Dense 출력과 맞추기 위해
            # 10 x 17을 170개 값으로 펼친다.
            y_list.append(
                y.reshape(-1)
            )


    x_data = np.asarray(
        x_list,
        dtype=np.float32,
    )


    y_data = np.asarray(
        y_list,
        dtype=np.float32,
    )


    return (
        x_data,
        y_data,
    )


# ============================================================
# Main
# ============================================================


def main():

    print("=" * 70)

    print(
        "HydroTwin Virtual Factory "
        "LSTM Generator V3"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # UCI Raw 기반 1Hz 데이터
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
        f"Input Window  : "
        f"{INPUT_WINDOW} sec"
    )

    print(
        f"Output Steps  : "
        f"{OUTPUT_STEPS} sec"
    )


    # --------------------------------------------------------
    # Record 단위 Train / Validation
    # --------------------------------------------------------

    record_count = (
        data.shape[0]
    )


    train_count = int(
        record_count
        * TRAIN_RATIO
    )


    train_records = (
        data[:train_count]
    )


    val_records = (
        data[train_count:]
    )


    print()

    print(
        f"Train Records : "
        f"{train_records.shape[0]}"
    )

    print(
        f"Val Records   : "
        f"{val_records.shape[0]}"
    )


    # --------------------------------------------------------
    # StandardScaler
    #
    # Train 데이터로만 fit
    # --------------------------------------------------------

    scaler = StandardScaler()


    scaler.fit(
        train_records.reshape(
            -1,
            sensor_count,
        )
    )


    train_scaled = (
        scaler.transform(
            train_records.reshape(
                -1,
                sensor_count,
            )
        )
        .reshape(
            train_records.shape
        )
        .astype(np.float32)
    )


    val_scaled = (
        scaler.transform(
            val_records.reshape(
                -1,
                sensor_count,
            )
        )
        .reshape(
            val_records.shape
        )
        .astype(np.float32)
    )


    # --------------------------------------------------------
    # Sliding Window
    # --------------------------------------------------------

    print()
    print(
        "Creating V3 training windows..."
    )


    x_train, y_train = (
        create_windows(
            train_scaled,
            INPUT_WINDOW,
            OUTPUT_STEPS,
        )
    )


    x_val, y_val = (
        create_windows(
            val_scaled,
            INPUT_WINDOW,
            OUTPUT_STEPS,
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
    # LSTM V3
    #
    # 구조는 V1/V2와 동일
    #
   # 차이:
    # 마지막 Dense 출력만
    # 17 -> 170
    #
    # 10초 x 17 Sensor
    # ========================================================

    output_size = (
        OUTPUT_STEPS
        * sensor_count
    )


    model = Sequential(
        [
            Input(
                shape=(
                    INPUT_WINDOW,
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
                output_size
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
    print("=" * 70)
    print("V3 MODEL")
    print("=" * 70)

    model.summary()


    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # EarlyStopping
    # --------------------------------------------------------

    early_stopping = (
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        )
    )


    # --------------------------------------------------------
    # Best V3 Model
    # --------------------------------------------------------

    checkpoint = (
        ModelCheckpoint(
            MODEL_FILE,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        )
    )


    # --------------------------------------------------------
    # V3 학습
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING V3 START")
    print("=" * 70)


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


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATION V3")
    print("=" * 70)


    val_loss, val_mae = (
        model.evaluate(
            x_val,
            y_val,
            verbose=0,
        )
    )


    print(
        f"Validation MSE : "
        f"{val_loss:.6f}"
    )


    print(
        f"Validation MAE : "
        f"{val_mae:.6f}"
    )


    # --------------------------------------------------------
    # Scaler 저장
    # --------------------------------------------------------

    joblib.dump(
        scaler,
        SCALER_FILE,
    )


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "version": "v3",

        "sensor_names": sensor_names,

        "sensor_count": sensor_count,

        "input_window": INPUT_WINDOW,

        "output_steps": OUTPUT_STEPS,

        "input_shape": [
            INPUT_WINDOW,
            sensor_count,
        ],

        "output_shape": [
            OUTPUT_STEPS,
            sensor_count,
        ],

        "training_records": int(
            train_records.shape[0]
        ),

        "validation_records": int(
            val_records.shape[0]
        ),

        "training_source": (
            "UCI Raw 17 Sensors Only"
        ),

        "generated_data_used_for_training": False,
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


    # --------------------------------------------------------
    # 모델 저장
    # --------------------------------------------------------

    model.save(
        MODEL_FILE
    )


    print(
        f"[SAVED] {MODEL_FILE}"
    )

    print(
        f"[SAVED] {SCALER_FILE}"
    )

    print(
        f"[SAVED] {METADATA_FILE}"
    )


    # ========================================================
    # Sample
    #
    # 실제 다음 10초
    # vs
    # 생성 다음 10초
    # ========================================================

    predicted_scaled_flat = (
        model.predict(
            x_val[:1],
            verbose=0,
        )[0]
    )


    predicted_scaled = (
        predicted_scaled_flat.reshape(
            OUTPUT_STEPS,
            sensor_count,
        )
    )


    actual_scaled = (
        y_val[0].reshape(
            OUTPUT_STEPS,
            sensor_count,
        )
    )


    generated = (
        scaler.inverse_transform(
            predicted_scaled
        )
    )


    actual = (
        scaler.inverse_transform(
            actual_scaled
        )
    )


    print()
    print("=" * 70)
    print("SAMPLE NEXT-10-SECONDS V3")
    print("=" * 70)


    # 첫 번째 미래 1초
    print()
    print("[Future +1 sec]")


    for (
        sensor,
        actual_value,
        generated_value,
    ) in zip(
        sensor_names,
        actual[0],
        generated[0],
    ):

        print(
            f"{sensor:5s} "
            f"actual={actual_value:10.3f} "
            f"generated={generated_value:10.3f}"
        )


    # 마지막 미래 10초
    print()
    print("[Future +10 sec]")


    for (
        sensor,
        actual_value,
        generated_value,
    ) in zip(
        sensor_names,
        actual[-1],
        generated[-1],
    ):

        print(
            f"{sensor:5s} "
            f"actual={actual_value:10.3f} "
            f"generated={generated_value:10.3f}"
        )


    print()
    print("=" * 70)
    print("TRAINING V3 PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
