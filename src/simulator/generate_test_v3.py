import csv
import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf


# ============================================================
# HydroTwin Virtual Factory
# Continuous Sensor Generation Test - V3
# ============================================================
#
# 목적
# ------------------------------------------------------------
# 학습 완료된 V3 LSTM Generator를 고정하고
# 새로운 17개 Sensor 데이터를 300초 연속 생성한다.
#
#
# V3
# ------------------------------------------------------------
# 최근 30초 x 17 Sensor
#          ↓
#         LSTM
#          ↓
# 다음 10초 x 17 Sensor
#
#
# 중요
# ------------------------------------------------------------
# model.fit() 사용하지 않음
# 재학습하지 않음
# 생성 데이터를 학습 데이터로 사용하지 않음
#
# 오직:
#
# load_model()
# predict()
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
# V3 Model
# ============================================================

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
# V3 생성 결과
# ============================================================

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "simulator"
    / "generated_300s_v3.csv"
)


GENERATE_SECONDS = 300

TRAIN_RATIO = 0.8


def main():

    print("=" * 80)

    print(
        "HydroTwin Continuous "
        "Sensor Generation Test V3"
    )

    print("=" * 80)


    # ========================================================
    # V3 모델 Load
    #
    # compile=False
    #
    # 학습하지 않고 predict만 사용
    # ========================================================

    model = (
        tf.keras.models.load_model(
            MODEL_FILE,
            compile=False,
        )
    )


    scaler = joblib.load(
        SCALER_FILE
    )


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


    input_window = int(
        metadata["input_window"]
    )


    output_steps = int(
        metadata["output_steps"]
    )


    print(
        f"Generator Version : V3"
    )

    print(
        f"Input Window      : "
        f"{input_window} sec"
    )

    print(
        f"Output Steps      : "
        f"{output_steps} sec"
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
    # UCI Raw 기반 1Hz Dataset
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
    # 학습에 사용한 Train Record가 아니라
    # Validation 영역에서 시작
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
            :input_window,
            :
        ]
        .copy()
    )


    expected_seed_shape = (
        input_window,
        sensor_count,
    )


    if seed_raw.shape != expected_seed_shape:

        raise ValueError(
            f"Invalid Seed Shape: "
            f"{seed_raw.shape}"
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
    # Scaling
    # ========================================================

    current_window = (
        scaler.transform(
            seed_raw
        )
        .astype(np.float32)
    )


    generated_rows = []


    print()

    print("=" * 80)
    print("V3 GENERATION START")
    print("=" * 80)


    # ========================================================
    # 300초 생성
    #
    # V3는 10초를 한 번에 생성한다.
    #
    # 따라서
    #
    # 300초 / 10초
    # = 30번 predict
    #
    # V1/V2의 300번 predict보다
    # 자기 생성값 재입력 횟수를 줄인다.
    # ========================================================

    generated_seconds = 0


    while generated_seconds < GENERATE_SECONDS:

        # ----------------------------------------------------
        # Shape
        #
        # (30, 17)
        #       ↓
        # (1, 30, 17)
        # ----------------------------------------------------

        input_batch = (
            current_window[
                np.newaxis,
                ...
            ]
        )


        # ====================================================
        # V3 predict
        #
        # 결과:
        #
        # 10초 x 17 Sensor
        # = 170개
        #
        # 재학습 없음
        # ====================================================

        predicted_scaled_flat = (
            model.predict(
                input_batch,
                verbose=0,
            )[0]
        )


        predicted_scaled = (
            predicted_scaled_flat.reshape(
                output_steps,
                sensor_count,
            )
        )


        # ====================================================
        # 실제 Sensor 단위로 역변환
        # ====================================================

        predicted_raw = (
            scaler.inverse_transform(
                predicted_scaled
            )
        )


        # ====================================================
        # NaN / Inf 검사
        # ====================================================

        if not np.isfinite(
            predicted_raw
        ).all():

            raise ValueError(
                "NaN or Inf detected "
                "during V3 generation"
            )


        # ====================================================
        # 생성된 10초 저장
        # ====================================================

        for row in predicted_raw:

            if (
                generated_seconds
                >= GENERATE_SECONDS
            ):
                break


            generated_rows.append(
                row.copy()
            )


            generated_seconds += 1


            # 10초 단위 주요 값 출력
            if (
                generated_seconds == 1
                or generated_seconds % 10 == 0
                or generated_seconds == GENERATE_SECONDS
            ):

                values = dict(
                    zip(
                        sensor_names,
                        row,
                    )
                )


                print(
                    f"[V3 generated "
                    f"{generated_seconds:3d}s] "
                    f"PS1={values['PS1']:.3f} "
                    f"PS2={values['PS2']:.3f} "
                    f"FS1={values['FS1']:.3f} "
                    f"TS1={values['TS1']:.3f} "
                    f"TS2={values['TS2']:.3f} "
                    f"VS1={values['VS1']:.3f}"
                )


        # ====================================================
        # Sliding Window 업데이트
        #
        # 기존 최근 30초
        #
        # +
        #
        # 새로 생성한 10초
        #
        # →
        #
        # 가장 최근 30초만 다시 입력
        #
        # 이것은 predict 입력 갱신이지
        # 재학습이 아니다.
        # ====================================================

        combined_window = np.vstack(
            [
                current_window,
                predicted_scaled,
            ]
        )


        current_window = (
            combined_window[
                -input_window:
            ]
        )


    # ========================================================
    # 생성 결과
    # ========================================================

    generated = np.asarray(
        generated_rows,
        dtype=np.float32,
    )


    print()

    print("=" * 80)
    print("V3 GENERATION VALIDATION")
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
            f"Unexpected Shape: "
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
    # V3 생성 통계
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

    print(
        "Sensor Statistics"
    )

    print("-" * 110)


    # ========================================================
    # 수업에서 배운
    #
    # mean / std / min / max
    #
    # 기준 평가
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
    # 생성된 17개 Sensor 값만 저장
    #
    # 실제값 / Label / 모델정보 등 없음
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
    print("V3 GENERATION TEST PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
