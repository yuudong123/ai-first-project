import csv
import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf


# ============================================================
# HydroTwin Virtual Factory
# Continuous Sensor Generation Test - V4
# ============================================================
#
# 목적
# ------------------------------------------------------------
# 학습 완료된 V4 LSTM Generator를 고정한 상태에서
# 새로운 17개 Sensor 데이터를 300초 연속 생성한다.
#
#
# V4 동작 방식
# ------------------------------------------------------------
# 최근 30초 x 17 Sensor
#          ↓
#         LSTM
#          ↓
# 다음 1초의 Sensor Delta 예측
#          ↓
# 현재 Sensor + Delta
#          ↓
# 다음 1초 Sensor 생성
#
#
# 중요
# ------------------------------------------------------------
# - model.fit() 사용하지 않음
# - 재학습하지 않음
# - 생성 데이터를 학습에 사용하지 않음
#
# 오직:
# - load_model()
# - predict()
#
# 만 사용한다.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# UCI Raw 기반 1Hz Dataset
#
# 최초 가상 공장 시동을 위한
# 30초 Seed만 사용한다.
# ============================================================

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "simulator"
    / "uci_1hz_17sensors.npz"
)


# ============================================================
# V4 모델 경로
# ============================================================

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
# V4 생성 결과
# ============================================================

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "simulator"
    / "generated_300s_v4.csv"
)


# ============================================================
# 생성 설정
# ============================================================

GENERATE_SECONDS = 300

TRAIN_RATIO = 0.8


def main():

    print("=" * 80)
    print("HydroTwin Continuous Sensor Generation Test V4")
    print("=" * 80)


    # ========================================================
    # V4 모델 Load
    #
    # compile=False
    # → 학습 없이 predict만 사용
    # ========================================================

    model = tf.keras.models.load_model(
        MODEL_FILE,
        compile=False,
    )


    # ========================================================
    # Scaler Load
    # ========================================================

    input_scaler = joblib.load(
        INPUT_SCALER_FILE
    )


    delta_scaler = joblib.load(
        DELTA_SCALER_FILE
    )


    # ========================================================
    # Metadata Load
    # ========================================================

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(f)


    sensor_names = (
        metadata["sensor_names"]
    )


    sensor_count = int(
        metadata["sensor_count"]
    )


    window_size = int(
        metadata["window_size"]
    )


    print(
        "Generator Version : V4"
    )

    print(
        f"Window Size       : "
        f"{window_size} sec"
    )

    print(
        f"Sensor Count      : "
        f"{sensor_count}"
    )

    print(
        f"Generate Seconds  : "
        f"{GENERATE_SECONDS}"
    )


    # ========================================================
    # UCI Raw 기반 Dataset
    # ========================================================

    dataset = np.load(
        DATA_FILE
    )


    data = (
        dataset["data"]
        .astype(np.float32)
    )


    print(
        f"Raw Dataset Shape : "
        f"{data.shape}"
    )


    # ========================================================
    # Validation 첫 Record를 Seed로 사용
    #
    # 학습에 사용한 Train 영역이 아니라
    # Validation 영역 첫 Record 사용
    #
    # Raw에서 가져오는 것은 최초 30초뿐이다.
    # ========================================================

    seed_record_index = int(
        data.shape[0]
        * TRAIN_RATIO
    )


    seed_raw = (
        data[
            seed_record_index,
            :window_size,
            :
        ]
        .copy()
    )


    expected_seed_shape = (
        window_size,
        sensor_count,
    )


    if seed_raw.shape != expected_seed_shape:

        raise ValueError(
            f"Invalid Seed Shape: "
            f"{seed_raw.shape}, "
            f"expected={expected_seed_shape}"
        )


    print(
        f"Seed Record       : "
        f"{seed_record_index}"
    )

    print(
        f"Seed Shape        : "
        f"{seed_raw.shape}"
    )


    # ========================================================
    # 현재 Window는 Raw 단위 그대로 유지
    # ========================================================

    current_window_raw = (
        seed_raw.copy()
    )


    generated_rows = []


    print()
    print("=" * 80)
    print("V4 GENERATION START")
    print("=" * 80)


    # ========================================================
    # 300초 연속 생성
    #
    # 매 시점:
    #
    # 최근 30초 Sensor
    #       ↓
    # Input Scaling
    #       ↓
    # LSTM
    #       ↓
    # 다음 1초 Delta
    #       ↓
    # Delta 역변환
    #       ↓
    # 현재 Sensor + Delta
    #       ↓
    # 다음 Sensor 생성
    #
    # model.fit() 없음
    # ========================================================

    for second in range(
        1,
        GENERATE_SECONDS + 1,
    ):

        # ----------------------------------------------------
        # 현재 30초 Sensor Scaling
        # ----------------------------------------------------

        current_window_scaled = (
            input_scaler
            .transform(
                current_window_raw
            )
            .astype(np.float32)
        )


        input_batch = (
            current_window_scaled[
                np.newaxis,
                ...
            ]
        )


        # ====================================================
        # 다음 1초 Delta Prediction
        #
        # 재학습 없음
        # ====================================================

        predicted_delta_scaled = (
            model.predict(
                input_batch,
                verbose=0,
            )[0]
        )


        # ====================================================
        # Delta를 실제 Sensor 단위로 복원
        # ====================================================

        predicted_delta = (
            delta_scaler
            .inverse_transform(
                predicted_delta_scaled.reshape(
                    1,
                    -1,
                )
            )[0]
        )


        # ====================================================
        # 현재 마지막 Sensor
        # ====================================================

        current_sensor = (
            current_window_raw[-1]
        )


        # ====================================================
        # 새로운 다음 1초 Sensor 생성
        #
        # current + delta
        # ====================================================

        next_sensor = (
            current_sensor
            + predicted_delta
        )


        # ====================================================
        # NaN / Inf 검사
        # ====================================================

        if not np.isfinite(
            next_sensor
        ).all():

            raise ValueError(
                f"NaN/Inf detected "
                f"at generated second "
                f"{second}"
            )


        # ====================================================
        # 생성 결과 저장
        # ====================================================

        generated_rows.append(
            next_sensor.copy()
        )


        # ====================================================
        # Sliding Window 갱신
        #
        # 가장 오래된 1초 제거
        # +
        # 새로 생성한 1초 추가
        #
        # 재학습이 아니라
        # 다음 predict 입력 갱신
        # ====================================================

        current_window_raw = np.vstack(
            [
                current_window_raw[1:],
                next_sensor,
            ]
        )


        # ====================================================
        # 콘솔 확인
        #
        # 10초마다 주요 Sensor 출력
        # ====================================================

        if (
            second == 1
            or second % 10 == 0
            or second == GENERATE_SECONDS
        ):

            values = dict(
                zip(
                    sensor_names,
                    next_sensor,
                )
            )


            print(
                f"[V4 generated "
                f"{second:3d}s] "
                f"PS1={values['PS1']:.3f} "
                f"PS2={values['PS2']:.3f} "
                f"FS1={values['FS1']:.3f} "
                f"TS1={values['TS1']:.3f} "
                f"TS2={values['TS2']:.3f} "
                f"VS1={values['VS1']:.3f}"
            )


    # ========================================================
    # 생성 결과 배열
    # ========================================================

    generated = np.asarray(
        generated_rows,
        dtype=np.float32,
    )


    print()
    print("=" * 80)
    print("V4 GENERATION VALIDATION")
    print("=" * 80)


    print(
        f"Generated Shape : "
        f"{generated.shape}"
    )


    expected_shape = (
        GENERATE_SECONDS,
        sensor_count,
    )


    if generated.shape != expected_shape:

        raise ValueError(
            f"Unexpected Generated Shape: "
            f"{generated.shape}"
        )


    # ========================================================
    # UCI Raw 전체 통계
    # ========================================================

    reference_flat = (
        data.reshape(
            -1,
            sensor_count,
        )
    )


    reference_mean = (
        reference_flat.mean(
            axis=0
        )
    )


    reference_std = (
        reference_flat.std(
            axis=0
        )
    )


    reference_min = (
        reference_flat.min(
            axis=0
        )
    )


    reference_max = (
        reference_flat.max(
            axis=0
        )
    )


    # ========================================================
    # V4 생성 데이터 통계
    # ========================================================

    generated_mean = (
        generated.mean(
            axis=0
        )
    )


    generated_std = (
        generated.std(
            axis=0
        )
    )


    generated_min = (
        generated.min(
            axis=0
        )
    )


    generated_max = (
        generated.max(
            axis=0
        )
    )


    print()
    print("Sensor Statistics")
    print("-" * 110)


    # ========================================================
    # mean / std / min / max 평가
    # ========================================================

    for index, sensor in enumerate(
        sensor_names
    ):

        mean_diff = (
            generated_mean[index]
            - reference_mean[index]
        )


        if reference_std[index] > 0:

            std_ratio = (
                generated_std[index]
                / reference_std[index]
            )

        else:

            std_ratio = 0.0


        range_ok = (
            generated_min[index]
            >= reference_min[index]
            and
            generated_max[index]
            <= reference_max[index]
        )


        print(
            f"{sensor:5s} "
            f"mean={generated_mean[index]:10.3f} "
            f"mean_diff={mean_diff:10.3f} "
            f"std={generated_std[index]:9.4f} "
            f"std_ratio={std_ratio:7.3f} "
            f"min={generated_min[index]:10.3f} "
            f"max={generated_max[index]:10.3f} "
            f"range="
            f"{'PASS' if range_ok else 'CHECK'}"
        )


    # ========================================================
    # CSV 저장
    #
    # 생성된 Sensor 결과만 저장
    #
    # actual / label / prediction / model 정보 없음
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)


        writer.writerow(
            [
                "generated_second"
            ]
            + sensor_names
        )


        for second, row in enumerate(
            generated,
            start=1,
        ):

            writer.writerow(
                [
                    second
                ]
                + [
                    round(
                        float(value),
                        6,
                    )
                    for value
                    in row
                ]
            )


    print()
    print(
        f"[SAVED] "
        f"{OUTPUT_FILE}"
    )


    print()
    print("=" * 80)
    print("V4 GENERATION TEST PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
