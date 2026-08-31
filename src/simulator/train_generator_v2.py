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
# HydroTwin Virtual Factory Generator V2
# ============================================================
#
# V1과의 차이:
#
# V1
#   최근 20초 -> 다음 1초
#
# V2
#   최근 30초 -> 다음 1초
#
# 나머지 LSTM 구조와 학습 조건은 최대한 동일하게 유지하여
# Window Size 증가가 장시간 생성 안정성에
# 도움이 되는지 비교한다.
#
#
# 학습 데이터:
#   UCI Raw 17 Sensor -> 1Hz 데이터만 사용
#
# 절대 사용하지 않음:
#   generated_300s.csv
#   V1 생성 데이터
#   profile.txt
#   고장 Label
#   기존 상태예측 model.pkl
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
    / "virtual_factory_generator_v2.keras"
)


SCALER_FILE = (
    MODEL_DIR
    / "sensor_scaler_v2.joblib"
)


METADATA_FILE = (
    MODEL_DIR
    / "generator_metadata_v2.json"
)


# ============================================================
# V2 학습 설정
# ============================================================

WINDOW_SIZE = 30

TRAIN_RATIO = 0.8

EPOCHS = 50

BATCH_SIZE = 256

RANDOM_SEED = 42


np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# Sliding Window
#
# 각 UCI 60초 Record 내부에서만 생성한다.
#
# 예:
#
# 0~29초 -> 30초
# 1~30초 -> 31초
# 2~31초 -> 32초
#
# ...
#
# 서로 다른 Record의 경계는 절대 연결하지 않는다.
# ============================================================


def create_windows(records, window_size):

    x_list = []
    y_list = []

    for record in records:

        seconds = record.shape[0]

        for start in range(
            seconds - window_size
        ):

            end = (
                start
                + window_size
            )

            x_list.append(
                record[start:end]
            )

            y_list.append(
                record[end]
            )

    x_data = np.asarray(
        x_list,
        dtype=np.float32,
    )

    y_data = np.asarray(
        y_list,
        dtype=np.float32,
    )

    return x_data, y_data


# ============================================================
# Main
# ============================================================


def main():

    print("=" * 70)

    print(
        "HydroTwin Virtual Factory "
        "LSTM Generator V2"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # UCI Raw 기반 1Hz Sensor 데이터 로드
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
        f"Dataset Shape : "
        f"{data.shape}"
    )

    print(
        f"Sensor Count  : "
        f"{len(sensor_names)}"
    )

    print(
        f"Window Size   : "
        f"{WINDOW_SIZE} sec"
    )


    # --------------------------------------------------------
    # Record 단위 Train / Validation 분리
    #
    # 같은 Record가 Train과 Validation에
    # 동시에 들어가지 않도록 한다.
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
    # Train 데이터로만 fit한다.
    # --------------------------------------------------------

    sensor_count = (
        data.shape[2]
    )


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
    # Sliding Window 생성
    # --------------------------------------------------------

    print()

    print(
        "Creating V2 training windows..."
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


    # --------------------------------------------------------
    # LSTM
    #
    # V1과 동일한 구조 유지
    # Window Size만 20 -> 30으로 변경
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
    print("V2 MODEL")
    print("=" * 70)

    model.summary()


    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # EarlyStopping
    # --------------------------------------------------------

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1,
    )


    # --------------------------------------------------------
    # 가장 좋은 V2 모델만 저장
    # --------------------------------------------------------

    checkpoint = ModelCheckpoint(
        MODEL_FILE,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )


    # --------------------------------------------------------
    # V2 학습
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING V2 START")
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
    print("VALIDATION V2")
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
    # V2 Scaler 저장
    # --------------------------------------------------------

    joblib.dump(
        scaler,
        SCALER_FILE,
    )


    print(
        f"[SAVED] {SCALER_FILE}"
    )


    # --------------------------------------------------------
    # V2 Metadata
    # --------------------------------------------------------

    metadata = {

        "version": "v2",

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


    print(
        f"[SAVED] {METADATA_FILE}"
    )


    # --------------------------------------------------------
    # V2 모델 저장
    # --------------------------------------------------------

    model.save(
        MODEL_FILE
    )


    print(
        f"[SAVED] {MODEL_FILE}"
    )


    # --------------------------------------------------------
    # 실제 다음 1초 비교
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


    generated = (
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
    print("SAMPLE NEXT-SECOND V2")
    print("=" * 70)


    for (
        sensor,
        actual_value,
        generated_value,
    ) in zip(
        sensor_names,
        actual,
        generated,
    ):

        print(
            f"{sensor:5s} "
            f"actual={actual_value:10.3f} "
            f"generated={generated_value:10.3f}"
        )


    print()
    print("=" * 70)
    print("TRAINING V2 PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
