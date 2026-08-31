from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# HydroTwin Virtual Factory
# Generated Sensor Data Evaluation
# ============================================================
#
# 목적
# ------------------------------------------------------------
# UCI Raw 기반 1Hz 데이터와
# LSTM이 새로 생성한 300초 데이터를 비교한다.
#
# 사용하는 평가:
# - mean
# - std
# - min
# - max
#
# 새로운 학습은 하지 않는다.
# model.fit() 없음.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]


REFERENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "simulator"
    / "uci_1hz_17sensors.npz"
)


GENERATED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "simulator"
    / "generated_300s.csv"
)


def main():

    print("=" * 100)
    print("HydroTwin Generated Sensor Evaluation")
    print("=" * 100)


    # --------------------------------------------------------
    # UCI Raw 기반 Reference 데이터
    # --------------------------------------------------------

    reference_npz = np.load(
        REFERENCE_FILE
    )

    reference = (
        reference_npz["data"]
        .astype(np.float32)
    )

    sensor_names = (
        reference_npz["sensor_names"]
        .astype(str)
        .tolist()
    )


    # (2205, 60, 17)
    # -> 모든 1초 Sensor 데이터를 하나로 펼친다.
    reference_flat = reference.reshape(
        -1,
        len(sensor_names),
    )


    # --------------------------------------------------------
    # LSTM 생성 데이터
    # --------------------------------------------------------

    generated_df = pd.read_csv(
        GENERATED_FILE
    )

    generated = (
        generated_df[
            sensor_names
        ]
        .to_numpy(
            dtype=np.float32
        )
    )


    print(
        f"Reference Shape : {reference_flat.shape}"
    )

    print(
        f"Generated Shape : {generated.shape}"
    )

    print()


    # --------------------------------------------------------
    # 센서별 통계 비교
    # --------------------------------------------------------

    print("=" * 100)
    print(
        "SENSOR STATISTICS"
    )
    print("=" * 100)


    for index, sensor in enumerate(
        sensor_names
    ):

        ref_values = (
            reference_flat[:, index]
        )

        gen_values = (
            generated[:, index]
        )


        ref_mean = np.mean(
            ref_values
        )

        ref_std = np.std(
            ref_values
        )

        ref_min = np.min(
            ref_values
        )

        ref_max = np.max(
            ref_values
        )


        gen_mean = np.mean(
            gen_values
        )

        gen_std = np.std(
            gen_values
        )

        gen_min = np.min(
            gen_values
        )

        gen_max = np.max(
            gen_values
        )


        mean_diff = (
            gen_mean - ref_mean
        )


        std_ratio = (
            gen_std / ref_std
            if ref_std > 0
            else 0
        )


        # 생성값이 원본 전체 최소/최대 범위 안에 있는지
        range_ok = (
            gen_min >= ref_min
            and gen_max <= ref_max
        )


        print(
            f"\n[{sensor}]"
        )

        print(
            f"  Reference "
            f"mean={ref_mean:10.3f} "
            f"std={ref_std:10.3f} "
            f"min={ref_min:10.3f} "
            f"max={ref_max:10.3f}"
        )

        print(
            f"  Generated "
            f"mean={gen_mean:10.3f} "
            f"std={gen_std:10.3f} "
            f"min={gen_min:10.3f} "
            f"max={gen_max:10.3f}"
        )

        print(
            f"  Mean Diff = "
            f"{mean_diff:10.3f}"
        )

        print(
            f"  Std Ratio = "
            f"{std_ratio:10.3f}"
        )

        print(
            f"  Raw Range = "
            f"{'PASS' if range_ok else 'CHECK'}"
        )


    print()
    print("=" * 100)
    print("EVALUATION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
