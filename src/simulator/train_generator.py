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
# HydroTwin Virtual Factory Generator Training
# ============================================================
#
# 목적:
# UCI Raw 17개 Sensor의 시간 흐름을 학습하여
#
#   과거 20초 x 17 Sensor
#              ↓
#             LSTM
#              ↓
#   다음 1초 x 17 Sensor
#
# 를 생성하는 내부 Generator 모델을 만든다.
#
# 주의:
# - profile.txt 사용 안 함
# - 고장 Label 사용 안 함
# - 기존 상태예측 model.pkl 사용 안 함
# - 오직 17개 Raw Sensor만 사용
#
# 이 모델은 내부 Sensor 생성용이다.
# Kafka에는 모델 정보가 아니라 생성된 Sensor 값만 나간다.
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
    / "virtual_factory_generator.keras"
)

SCALER_FILE = (
    MODEL_DIR
    / "sensor_scaler.joblib"
)

METADATA_FILE = (
    MODEL_DIR
    / "generator_metadata.json"
)


# ============================================================
# 학습 설정
# ============================================================

WINDOW_SIZE = 20

TRAIN_RATIO = 0.8

EPOCHS = 50

BATCH_SIZE = 256

RANDOM_SEED = 42


np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# Sliding Window 생성
#
# 각 60초 기록 안에서만 Window 생성
#
# 0~19초 -> 20초
# 1~20초 -> 21초
# ...
#
# 서로 다른 60초 기록 사이를 연결하지 않는다.
# ============================================================

def create_windows(records, window_size):

    x_list = []
    y_list = []

    for record in records:

        seconds = record.shape[0]

        for start in range(
            seconds - window_size
        ):

            end = start + window_size

            x = record[
                start:end
            ]

            y = record[
                end
            ]

            x_list.append(x)
            y_list.append(y)

    x_data = np.asarray(
        x_list,
        dtype=np.float32,
    )

    y_data = np.asarray(
        y_list,
        dtype=np.float32,
    )

    return x_data, y_data


def main():

    print("=" * 70)
    print(
        "HydroTwin Virtual Factory "
        "LSTM Generator Training"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # 데이터 로드
    # --------------------------------------------------------

    dataset_file = np.load(
        DATA_FILE
    )

    data = dataset_file[
        "data"
    ].astype(
        np.float32
    )

    sensor_names = (
        dataset_file[
            "sensor_names"
        ]
        .astype(str)
        .tolist()
    )

    print(
        f"Dataset Shape : {data.shape}"
    )

    print(
        f"Sensor Count  : {len(sensor_names)}"
    )

    print(
        f"Window Size   : {WINDOW_SIZE} sec"
    )

    # --------------------------------------------------------
    # Train / Validation 기록 분리
    # --------------------------------------------------------

    record_count = data.shape[0]

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
    # Train 데이터만으로 fit
    # --------------------------------------------------------

    sensor_count = data.shape[2]

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
        .astype(
            np.float32
        )
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
        .astype(
            np.float32
        )
    )

    # --------------------------------------------------------
    # Sliding Window 생성
    # --------------------------------------------------------

    print()
    print(
        "Creating training windows..."
    )

    x_train, y_train = (
        create_windows(
            train_scaled,
            WINDOW_SIZE,
        )
    )

    x_val, y_val = (
        create_windows(
            val_scaled,
            WINDOW_SIZE,
        )
    )

    print(
        f"X Train : {x_train.shape}"
    )

    print(
        f"Y Train : {y_train.shape}"
    )

    print(
        f"X Val   : {x_val.shape}"
    )

    print(
        f"Y Val   : {y_val.shape}"
    )

    # --------------------------------------------------------
    # LSTM Generator
    # --------------------------------------------------------

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
    print("=" * 70)
    print("MODEL")
    print("=" * 70)

    model.summary()

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Callback
    # --------------------------------------------------------

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1,
    )

    checkpoint = ModelCheckpoint(
        MODEL_FILE,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    # --------------------------------------------------------
    # 학습
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING START")
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
    # Validation 평가
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATION")
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

    print(
        f"[SAVED] {SCALER_FILE}"
    )

    # --------------------------------------------------------
    # Metadata 저장
    # --------------------------------------------------------

    metadata = {
        "sensor_names": sensor_names,
        "sensor_count": sensor_count,
        "window_size": WINDOW_SIZE,
        "input_shape": [
            WINDOW_SIZE,
            sensor_count,
        ],
        "output_size": sensor_count,
        "training_records": int(
            train_records.shape[0]
        ),
        "validation_records": int(
            val_records.shape[0]
        ),
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

    print(
        f"[SAVED] {METADATA_FILE}"
    )

    # --------------------------------------------------------
    # 최종 모델 저장
    # --------------------------------------------------------

    model.save(
        MODEL_FILE
    )

    print(
        f"[SAVED] {MODEL_FILE}"
    )

    # --------------------------------------------------------
    # 샘플 생성 결과 확인
    # --------------------------------------------------------

    sample_input = (
        x_val[:1]
    )

    predicted_scaled = (
        model.predict(
            sample_input,
            verbose=0,
        )
    )

    predicted = (
        scaler.inverse_transform(
            predicted_scaled
        )[0]
    )

    actual = (
        scaler.inverse_transform(
            y_val[:1]
        )[0]
    )

    print()
    print("=" * 70)
    print("SAMPLE NEXT-SECOND")
    print("=" * 70)

    for (
        sensor_name,
        actual_value,
        generated_value,
    ) in zip(
        sensor_names,
        actual,
        predicted,
    ):

        print(
            f"{sensor_name:5s} "
            f"actual={actual_value:10.3f} "
            f"generated={generated_value:10.3f}"
        )

    print()
    print("=" * 70)
    print("TRAINING PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
