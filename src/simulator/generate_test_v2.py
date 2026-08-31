import csv
import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf


# ============================================================
# HydroTwin Virtual Factory
# Continuous Sensor Generation Test - V2
# ============================================================
#
# 목적
# ------------------------------------------------------------
# 이미 학습 완료된 V2 LSTM Generator를 고정하여
# 새로운 17개 Sensor 데이터를 300초 연속 생성한다.
#
# V2 특징
# ------------------------------------------------------------
# 최근 30초 x 17 Sensor
#          ↓
#         LSTM
#          ↓
# 다음 1초 x 17 Sensor
#
#
# 절대 하지 않는 것
# ------------------------------------------------------------
# - model.fit()
# - 재학습
# - 생성 데이터를 다시 학습에 사용
# - profile.txt 사용
# - 고장 Label 사용
#
#
# 사용:
# - load_model()
# - predict()
#
# 생성된 데이터는 다음 predict의 입력으로만 사용한다.
# 이는 재학습이 아니다.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# 원본 UCI 기반 1Hz Dataset
#
# 최초 가상 공장 시동용 Seed만 가져온다.
# 이후 생성값은 UCI에서 가져오지 않는다.
# ============================================================

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "simulator"
    / "uci_1hz_17sensors.npz"
)


# ============================================================
# V2 모델 경로
# ============================================================

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
# V2 생성 결과
# ============================================================

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "simulator"
    / "generated_300s_v2.csv"
)


# ============================================================
# 생성 설정
# ============================================================

GENERATE_SECONDS = 300


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("HydroTwin Continuous Sensor Generation Test V2")
    print("=" * 70)


    # --------------------------------------------------------
    # V2 모델 로드
    #
    # compile=False:
    # 학습용 optimizer 등이 필요하지 않다.
    # predict만 사용한다.
    # --------------------------------------------------------

    model = tf.keras.models.load_model(
        MODEL_FILE,
        compile=False,
    )


    # --------------------------------------------------------
    # V2 Scaler 로드
    # --------------------------------------------------------

    scaler = joblib.load(
        SCALER_FILE
    )


    # --------------------------------------------------------
    # V2 Metadata 로드
    # --------------------------------------------------------

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(f)


    sensor_names = (
        metadata["sensor_names"]
    )


    window_size = int(
        metadata["window_size"]
    )


    sensor_count = int(
        metadata["sensor_count"]
    )


    print(
        f"Generator Version : V2"
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
    # 원본 Raw 기반 Dataset 로드
    #
    # 주의:
    # 재학습용이 아니다.
    #
    # 최초 30초 Seed를 가져오기 위한 용도만 사용한다.
    # ========================================================

    dataset_file = np.load(
        DATA_FILE
    )


    data = (
        dataset_file["data"]
        .astype(np.float32)
    )


    print(
        f"Raw Dataset Shape : "
        f"{data.shape}"
    )


    # ========================================================
    # Validation 영역의 첫 Record 사용
    #
    # Train Ratio = 0.8
    #
    # 2205 * 0.8
    # ≈ 1764
    #
    # V2는 Window Size 30초이므로
    # 최초 30초만 Seed로 사용한다.
    # ========================================================

    seed_record_index = int(
        data.shape[0] * 0.8
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
            f"Invalid seed shape: "
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
    # Seed Scaling
    # ========================================================

    current_window = (
        scaler.transform(
            seed_raw
        )
        .astype(np.float32)
    )


    generated_rows = []


    print()
    print("=" * 70)
    print("V2 GENERATION START")
    print("=" * 70)


    # ========================================================
    # 300초 연속 생성
    #
    # 최초 Seed 이후에는
    # UCI에서 다음 Sensor 값을 가져오지 않는다.
    #
    # LSTM이 생성한 Sensor 값을
    # 다음 Window에 추가하여 계속 predict한다.
    #
    # model.fit()은 절대 호출하지 않는다.
    # ========================================================

    for second in range(
        1,
        GENERATE_SECONDS + 1,
    ):

        # ----------------------------------------------------
        # 현재 Window
        #
        # Shape:
        #
        # (30, 17)
        #
        # →
        #
        # (1, 30, 17)
        # ----------------------------------------------------

        input_batch = (
            current_window[
                np.newaxis,
                ...,
            ]
        )


        # ----------------------------------------------------
        # V2 LSTM Prediction
        #
        # 재학습 없음
        # ----------------------------------------------------

        next_scaled = (
            model.predict(
                input_batch,
                verbose=0,
            )[0]
        )


        # ----------------------------------------------------
        # StandardScaler 역변환
        #
        # 실제 Sensor 단위로 복원
        # ----------------------------------------------------

        next_raw = (
            scaler.inverse_transform(
                next_scaled.reshape(
                    1,
                    -1,
                )
            )[0]
        )


        # ----------------------------------------------------
        # NaN / Inf 검증
        # ----------------------------------------------------

        if not np.isfinite(
            next_raw
        ).all():

            raise ValueError(
                f"NaN/Inf detected "
                f"at generated second "
                f"{second}"
            )


        generated_rows.append(
            next_raw.copy()
        )


        # ====================================================
        # Sliding Window 이동
        #
        # 가장 오래된 1초 제거
        #
        # +
        #
        # 방금 생성한 다음 1초 추가
        #
        # 이것은 재학습이 아니라
        # 다음 predict를 위한 입력 갱신이다.
        # ====================================================

        current_window = np.vstack(
            [
                current_window[1:],
                next_scaled,
            ]
        )


        # ----------------------------------------------------
        # 콘솔 확인
        #
        # 10초마다 주요 Sensor만 표시
        # ----------------------------------------------------

        if (
            second == 1
            or second % 10 == 0
            or second == GENERATE_SECONDS
        ):

            values = dict(
                zip(
                    sensor_names,
                    next_raw,
                )
            )


            print(
                f"[V2 generated "
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
    #
    # Shape:
    #
    # (300, 17)
    # ========================================================

    generated = np.asarray(
        generated_rows,
        dtype=np.float32,
    )


    # ========================================================
    # 생성 결과 검증
    # ========================================================

    print()
    print("=" * 70)
    print("V2 GENERATION VALIDATION")
    print("=" * 70)


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
            f"Unexpected generated shape: "
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


    # ========================================================
    # V2 생성 데이터 통계
    # ========================================================

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


    print()
    print("Sensor Statistics")
    print("-" * 100)


    # ========================================================
    # 수업에서 배운
    #
    # mean
    # std
    # min
    # max
    #
    # 기준으로 원본과 생성 데이터 비교
    # ========================================================

    for index, sensor in enumerate(
        sensor_names
    ):

        mean_diff = (
            generated_mean[index]
            - reference_mean[index]
        )


        std_ratio = (
            generated_std[index]
            / reference_std[index]
            if reference_std[index] > 0
            else 0.0
        )


        range_ok = (
            generated_min[index]
            >= reference_min[index]
            and generated_max[index]
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
    # V2 생성 데이터 CSV 저장
    #
    # 저장되는 것은
    # 생성된 Sensor 데이터뿐이다.
    #
    # actual
    # label
    # prediction
    # model 정보
    #
    # 등을 넣지 않는다.
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
        f"[SAVED] {OUTPUT_FILE}"
    )


    print()
    print("=" * 70)
    print("V2 GENERATION TEST PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
