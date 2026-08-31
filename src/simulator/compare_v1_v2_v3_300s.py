from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# HydroTwin
# Raw vs V1 vs V2 vs V3 - TS1 300 sec Comparison
# ============================================================
#
# 사용 기술
# - NumPy
# - pandas
# - matplotlib
#
# Raw:
# UCI Validation 영역의 60초 Record 5개를 연결하여
# 300초 참고 데이터로 사용한다.
#
# 주의:
# 각 UCI Record가 실제 시간적으로 연속되었다는 뜻은 아니다.
# 따라서 60초마다 Record 경계를 그래프에 표시한다.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "simulator"
)


GRAPH_DIR = (
    DATA_DIR
    / "graphs"
)


RAW_FILE = (
    DATA_DIR
    / "uci_1hz_17sensors.npz"
)


V1_FILE = (
    DATA_DIR
    / "generated_300s_v1.csv"
)


V2_FILE = (
    DATA_DIR
    / "generated_300s_v2.csv"
)


V3_FILE = (
    DATA_DIR
    / "generated_300s_v3.csv"
)


OUTPUT_FILE = (
    GRAPH_DIR
    / "raw_v1_v2_v3_ts1_300s.png"
)


TRAIN_RATIO = 0.8
RAW_RECORD_COUNT = 5


def main():

    print("=" * 80)
    print("HydroTwin Raw vs V1 vs V2 vs V3 - TS1 300s")
    print("=" * 80)


    # ========================================================
    # Raw UCI 데이터
    # ========================================================

    raw_npz = np.load(
        RAW_FILE
    )


    raw_data = (
        raw_npz["data"]
        .astype(np.float32)
    )


    sensor_names = (
        raw_npz["sensor_names"]
        .astype(str)
        .tolist()
    )


    ts1_index = (
        sensor_names.index("TS1")
    )


    # --------------------------------------------------------
    # Validation 영역 첫 번째 Record부터 5개 사용
    #
    # 5 Records x 60 sec
    # = 300 sec
    # --------------------------------------------------------

    validation_start = int(
        raw_data.shape[0]
        * TRAIN_RATIO
    )


    raw_records = (
        raw_data[
            validation_start:
            validation_start + RAW_RECORD_COUNT,
            :,
            ts1_index
        ]
    )


    # (5, 60)
    # -> (300,)
    raw_ts1 = (
        raw_records
        .reshape(-1)
    )


    print(
        f"Raw Records     : "
        f"{validation_start} ~ "
        f"{validation_start + RAW_RECORD_COUNT - 1}"
    )

    print(
        f"Raw TS1 Shape   : "
        f"{raw_ts1.shape}"
    )


    # ========================================================
    # V1 / V2 / V3
    # ========================================================

    v1 = pd.read_csv(
        V1_FILE
    )

    v2 = pd.read_csv(
        V2_FILE
    )

    v3 = pd.read_csv(
        V3_FILE
    )


    print(
        f"V1 Shape        : {v1.shape}"
    )

    print(
        f"V2 Shape        : {v2.shape}"
    )

    print(
        f"V3 Shape        : {v3.shape}"
    )


    # ========================================================
    # 시간축
    # ========================================================

    seconds = np.arange(
        1,
        301
    )


    # ========================================================
    # Line Plot
    # ========================================================

    plt.figure(
        figsize=(16, 7)
    )


    plt.plot(
        seconds,
        raw_ts1,
        label="Raw UCI TS1",
    )


    plt.plot(
        v1["generated_second"],
        v1["TS1"],
        label="V1 Generated",
    )


    plt.plot(
        v2["generated_second"],
        v2["TS1"],
        label="V2 Generated",
    )


    plt.plot(
        v3["generated_second"],
        v3["TS1"],
        label="V3 Generated",
    )


    # ========================================================
    # Raw Record 경계 표시
    #
    # 60 / 120 / 180 / 240초
    # ========================================================

    for boundary in [
        60,
        120,
        180,
        240,
    ]:

        plt.axvline(
            x=boundary,
            linestyle="--",
            alpha=0.5,
        )


    plt.title(
        "HydroTwin TS1 - Raw vs V1 vs V2 vs V3 (300 sec)"
    )


    plt.xlabel(
        "Time (sec)"
    )


    plt.ylabel(
        "TS1"
    )


    plt.grid(
        True
    )


    plt.legend()


    plt.tight_layout()


    GRAPH_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    plt.savefig(
        OUTPUT_FILE,
        dpi=150,
    )


    plt.close()


    # ========================================================
    # 간단한 평균 / 표준편차도 출력
    # ========================================================

    print()
    print("=" * 80)
    print("TS1 STATISTICS")
    print("=" * 80)


    datasets = {
        "RAW": raw_ts1,
        "V1": v1["TS1"].to_numpy(),
        "V2": v2["TS1"].to_numpy(),
        "V3": v3["TS1"].to_numpy(),
    }


    for name, values in datasets.items():

        print(
            f"{name:3s} "
            f"Mean={np.mean(values):8.3f} "
            f"Std={np.std(values):8.3f} "
            f"Min={np.min(values):8.3f} "
            f"Max={np.max(values):8.3f}"
        )


    print()
    print(
        f"[SAVED] {OUTPUT_FILE}"
    )


    print()
    print("=" * 80)
    print("300s COMPARISON PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
