import csv
import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf


# ============================================================
# HydroTwin Virtual Factory
# Continuous Sensor Generation Test
# ============================================================
#
# 목적
# ------------------------------------------------------------
# 이미 학습 완료된 LSTM Generator를 고정하여
# 새로운 17개 Sensor 데이터를 연속 생성한다.
#
# 절대 하지 않는 것
# ------------------------------------------------------------
# - model.fit()
# - 재학습
# - 생성 데이터로 가중치 변경
# - profile.txt 사용
# - 고장 Label 사용
#
# 오직:
#
#   load_model()
#       ↓
#   predict()
#       ↓
#   predict()
#       ↓
#   predict()
#
# 만 반복한다.
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


OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "simulator"
    / "generated_300s.csv"
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
    print("HydroTwin Continuous Sensor Generation Test")
    print("=" * 70)


    # --------------------------------------------------------
    # 저장된 학습 결과 로드
    #
    # 중요:
    # compile=False
    #
    # 학습용 optimizer 등은 필요하지 않으며
    # predict만 사용한다.
    # --------------------------------------------------------

    model = tf.keras.models.load_model(
        MODEL_FILE,
        compile=False,
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

    window_size = int(
        metadata["window_size"]
    )

    sensor_count = int(
        metadata["sensor_count"]
    )


    print(
        f"Window Size      : "
        f"{window_size} sec"
    )

    print(
        f"Sensor Count     : "
        f"{sensor_count}"
    )

    print(
        f"Generate Seconds : "
        f"{GENERATE_SECONDS}"
    )


    # --------------------------------------------------------
    # Raw Dataset 로드
    #
    # 이것은 재학습용이 아니다.
    #
    # 가상 공장을 처음 시동할 때 필요한
    # 최초 20초 Seed만 가져오기 위해 사용한다.
    # --------------------------------------------------------

    dataset_file = np.load(
        DATA_FILE
    )

    data = (
        dataset_file["data"]
        .astype(np.float32)
    )


    # --------------------------------------------------------
    # 학습에 사용하지 않았던 Validation 첫 Record를
    # 시작 Seed로 사용한다.
    #
    # 2205 x 0.8 = 1764
    # --------------------------------------------------------

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


    if seed_raw.shape != (
        window_size,
        sensor_count,
    ):

        raise ValueError(
            f"Invalid seed shape: "
            f"{seed_raw.shape}"
        )


    print(
        f"Seed Record      : "
        f"{seed_record_index}"
    )

    print(
        f"Seed Shape       : "
        f"{seed_raw.shape}"
    )


    # --------------------------------------------------------
    # Seed Scaling
    # --------------------------------------------------------

    current_window = (
        scaler.transform(
            seed_raw
        )
        .astype(np.float32)
    )


    generated_rows = []


    print()
    print("=" * 70)
    print("GENERATION START")
    print("=" * 70)


    # ========================================================
    # 연속 생성
    #
    # 최초 20초 이후부터는
    # UCI TXT에서 다음 값을 가져오지 않는다.
    #
    # LSTM이 생성한 다음 1초를
    # 다시 Window에 넣고 계속 predict한다.
    #
    # model.fit() 없음.
    # ========================================================

    for second in range(
        1,
        GENERATE_SECONDS + 1,
    ):

        input_batch = (
            current_window[
                np.newaxis,
                ...,
            ]
        )


        # ----------------------------------------------------
        # predict만 수행
        # ----------------------------------------------------

        next_scaled = (
            model.predict(
                input_batch,
                verbose=0,
            )[0]
        )


        # 원래 Sensor 단위로 복원
        next_raw = (
            scaler.inverse_transform(
                next_scaled.reshape(
                    1,
                    -1,
                )
            )[0]
        )


        # NaN / Inf 안전 검사
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


        # ----------------------------------------------------
        # 가장 오래된 1초 제거
        # +
        # 방금 생성한 1초 추가
        #
        # 이것은 재학습이 아니라
        # 다음 예측의 입력 Window 갱신이다.
        # ----------------------------------------------------

        current_window = np.vstack(
            [
                current_window[1:],
                next_scaled,
            ]
        )


        # 너무 많은 출력은 피하고
        # 10초마다 상태 표시
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
                f"[generated {second:3d}s] "
                f"PS1={values['PS1']:.3f} "
                f"FS1={values['FS1']:.3f} "
                f"TS1={values['TS1']:.3f} "
                f"TS2={values['TS2']:.3f} "
                f"VS1={values['VS1']:.3f}"
            )


    generated = np.asarray(
        generated_rows,
        dtype=np.float32,
    )


    # ========================================================
    # 생성 결과 검증
    # ========================================================

    print()
    print("=" * 70)
    print("GENERATION VALIDATION")
    print("=" * 70)


    print(
        f"Generated Shape : "
        f"{generated.shape}"
    )


    if generated.shape != (
        GENERATE_SECONDS,
        sensor_count,
    ):

        raise ValueError(
            f"Unexpected generated shape: "
            f"{generated.shape}"
        )


    # 원본 전체 Sensor의 실제 범위
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

    reference_std = (
        reference_flat.std(
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

    generated_std = (
        generated.std(
            axis=0
        )
    )


    print()
    print(
        "Sensor Statistics"
    )

    print("-" * 70)


    for index, sensor in enumerate(
        sensor_names
    ):

        std_ratio = (
            generated_std[index]
            / max(
                reference_std[index],
                1e-8,
            )
        )

        print(
            f"{sensor:5s} "
            f"gen_min={generated_min[index]:10.3f} "
            f"gen_max={generated_max[index]:10.3f} "
            f"gen_std={generated_std[index]:9.4f} "
            f"std_ratio={std_ratio:7.3f}"
        )


    # ========================================================
    # CSV 저장
    #
    # 이 CSV에는 생성된 300초 데이터만 저장된다.
    # actual 값이나 학습 Label은 저장하지 않는다.
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
            ["generated_second"]
            + sensor_names
        )


        for second, row in enumerate(
            generated,
            start=1,
        ):

            writer.writerow(
                [second]
                + [
                    round(
                        float(value),
                        6,
                    )
                    for value in row
                ]
            )


    print()
    print(
        f"[SAVED] {OUTPUT_FILE}"
    )


    print()
    print("=" * 70)
    print("GENERATION TEST PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
