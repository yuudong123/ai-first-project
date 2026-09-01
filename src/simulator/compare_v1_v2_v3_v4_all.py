from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# HydroTwin Generator Comparison
#
# Raw vs V1 vs V2 vs V3 vs V4
#
# 사용 기술
# ------------------------------------------------------------
# - NumPy
# - pandas
# - matplotlib
# - mean
# - std
# - min
# - max
#
# 생성 결과
# ------------------------------------------------------------
# 1. TS1 300초 Line Plot
# 2. V1~V4 Mean / Std Error Bar Plot
# 3. TS1 Histogram
# 4. TS1 Box Plot
# 5. 통계 Summary CSV
#
# 주의
# ------------------------------------------------------------
# Raw 300초는 Validation 영역의
# 60초 Record 5개를 참고용으로 이어 붙인 것이다.
#
# 5개의 Record가 실제 시간적으로
# 연속된 300초라는 의미는 아니다.
# ============================================================


# ============================================================
# 프로젝트 경로
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


# ============================================================
# 데이터 파일
# ============================================================

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


V4_FILE = (
    DATA_DIR
    / "generated_300s_v4.csv"
)


MODEL_FILES = {
    "V1": V1_FILE,
    "V2": V2_FILE,
    "V3": V3_FILE,
    "V4": V4_FILE,
}


# ============================================================
# 출력 파일
# ============================================================

LINE_FILE = (
    GRAPH_DIR
    / "raw_v1_v2_v3_v4_ts1_300s.png"
)


BAR_FILE = (
    GRAPH_DIR
    / "v1_v2_v3_v4_mean_std_error.png"
)


HIST_FILE = (
    GRAPH_DIR
    / "raw_v1_v2_v3_v4_ts1_histogram.png"
)


BOX_FILE = (
    GRAPH_DIR
    / "raw_v1_v2_v3_v4_ts1_boxplot.png"
)


SUMMARY_FILE = (
    DATA_DIR
    / "generator_v1_v2_v3_v4_summary.csv"
)


# ============================================================
# 설정
# ============================================================

TRAIN_RATIO = 0.8

RAW_RECORD_COUNT = 5

GENERATED_SECONDS = 300


# ============================================================
# 파일 존재 확인
# ============================================================

def check_files():

    required_files = [
        RAW_FILE,
        V1_FILE,
        V2_FILE,
        V3_FILE,
        V4_FILE,
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"File not found: {file_path}"
            )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 100)

    print(
        "HydroTwin "
        "Raw vs V1 vs V2 vs V3 vs V4"
    )

    print("=" * 100)


    # --------------------------------------------------------
    # 필요한 파일 확인
    # --------------------------------------------------------

    check_files()


    GRAPH_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # Raw UCI 데이터 로드
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


    sensor_count = len(
        sensor_names
    )


    ts1_index = (
        sensor_names.index(
            "TS1"
        )
    )


    print(
        f"Raw Dataset       : "
        f"{raw_data.shape}"
    )


    # ========================================================
    # Raw 300초 참고 데이터
    #
    # Validation 영역 첫 5개 Record
    #
    # 60초 x 5
    # = 300초
    #
    # 실제 연속 300초라는 의미는 아니다.
    # ========================================================

    validation_start = int(
        raw_data.shape[0]
        * TRAIN_RATIO
    )


    raw_reference = (
        raw_data[
            validation_start:
            validation_start
            + RAW_RECORD_COUNT
        ]
        .reshape(
            -1,
            sensor_count,
        )
    )


    if raw_reference.shape[0] != GENERATED_SECONDS:

        raise ValueError(
            f"Raw reference seconds error: "
            f"{raw_reference.shape}"
        )


    raw_ts1_300 = (
        raw_reference[
            :,
            ts1_index
        ]
    )


    print(
        f"Raw 300s Reference: "
        f"{raw_reference.shape}"
    )


    print(
        f"Raw Records       : "
        f"{validation_start} ~ "
        f"{validation_start + RAW_RECORD_COUNT - 1}"
    )


    # ========================================================
    # 전체 Raw 통계용 데이터
    # ========================================================

    raw_all = (
        raw_data.reshape(
            -1,
            sensor_count,
        )
    )


    raw_mean = (
        raw_all.mean(
            axis=0
        )
    )


    raw_std = (
        raw_all.std(
            axis=0
        )
    )


    raw_min = (
        raw_all.min(
            axis=0
        )
    )


    raw_max = (
        raw_all.max(
            axis=0
        )
    )


    # ========================================================
    # V1 ~ V4 생성 데이터 로드
    # ========================================================

    generated = {}


    for (
        model_name,
        file_path,
    ) in MODEL_FILES.items():

        dataframe = pd.read_csv(
            file_path
        )


        missing_sensors = [
            sensor
            for sensor in sensor_names
            if sensor not in dataframe.columns
        ]


        if missing_sensors:

            raise ValueError(
                f"{model_name}: "
                f"missing sensors = "
                f"{missing_sensors}"
            )


        if len(dataframe) != GENERATED_SECONDS:

            raise ValueError(
                f"{model_name}: "
                f"expected 300 rows, "
                f"actual={len(dataframe)}"
            )


        generated[
            model_name
        ] = dataframe


        print(
            f"{model_name} Generated      : "
            f"{dataframe.shape}"
        )


    # ========================================================
    # 1.
    # TS1 300초 LINE PLOT
    #
    # 장시간 움직임 / 자체 Drift 확인
    # ========================================================

    seconds = np.arange(
        1,
        GENERATED_SECONDS + 1,
    )


    plt.figure(
        figsize=(16, 7)
    )


    plt.plot(
        seconds,
        raw_ts1_300,
        label="Raw UCI Reference",
    )


    for (
        model_name,
        dataframe,
    ) in generated.items():

        plt.plot(
            dataframe[
                "generated_second"
            ],
            dataframe[
                "TS1"
            ],
            label=f"{model_name} Generated",
        )


    # --------------------------------------------------------
    # Raw Record 경계 표시
    #
    # 60초 단위로 서로 다른 UCI Record임을 표시
    # --------------------------------------------------------

    for boundary in [
        60,
        120,
        180,
        240,
    ]:

        plt.axvline(
            x=boundary,
            linestyle="--",
            alpha=0.3,
        )


    plt.title(
        "HydroTwin TS1 - "
        "Raw vs V1 vs V2 vs V3 vs V4 "
        "(300 sec)"
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


    plt.savefig(
        LINE_FILE,
        dpi=150,
    )


    plt.close()


    # ========================================================
    # 2.
    # Mean / Std Error 계산
    #
    # Raw 전체 132,300초 통계와
    # 각 생성 모델 300초 통계 비교
    #
    # 값이 낮을수록 Raw 전체 통계에 가까움
    # ========================================================

    summary_rows = []


    for (
        model_name,
        dataframe,
    ) in generated.items():

        generated_values = (
            dataframe[
                sensor_names
            ]
            .to_numpy(
                dtype=np.float32
            )
        )


        gen_mean = (
            generated_values.mean(
                axis=0
            )
        )


        gen_std = (
            generated_values.std(
                axis=0
            )
        )


        gen_min = (
            generated_values.min(
                axis=0
            )
        )


        gen_max = (
            generated_values.max(
                axis=0
            )
        )


        mean_error_pct = (
            np.abs(
                gen_mean
                - raw_mean
            )
            /
            np.maximum(
                np.abs(raw_mean),
                1e-8,
            )
            * 100
        )


        std_error_pct = (
            np.abs(
                gen_std
                - raw_std
            )
            /
            np.maximum(
                raw_std,
                1e-8,
            )
            * 100
        )


        range_pass_count = int(
            np.sum(
                (
                    gen_min
                    >= raw_min
                )
                &
                (
                    gen_max
                    <= raw_max
                )
            )
        )


        summary_rows.append(
            {
                "model":
                    model_name,

                "mean_error_pct":
                    float(
                        mean_error_pct.mean()
                    ),

                "std_error_pct":
                    float(
                        std_error_pct.mean()
                    ),

                "range_pass_count":
                    range_pass_count,
            }
        )


    summary_df = pd.DataFrame(
        summary_rows
    )


    print()
    print("=" * 100)

    print(
        "OVERALL STATISTICS"
    )

    print("=" * 100)


    print(
        summary_df.to_string(
            index=False
        )
    )


    # ========================================================
    # Mean / Std Error BAR PLOT
    # ========================================================

    x = np.arange(
        len(summary_df)
    )


    width = 0.35


    plt.figure(
        figsize=(10, 6)
    )


    plt.bar(
        x - width / 2,
        summary_df[
            "mean_error_pct"
        ],
        width,
        label="Mean Error %",
    )


    plt.bar(
        x + width / 2,
        summary_df[
            "std_error_pct"
        ],
        width,
        label="Std Error %",
    )


    plt.xticks(
        x,
        summary_df[
            "model"
        ],
    )


    plt.ylabel(
        "Error (%)"
    )


    plt.title(
        "HydroTwin Generator "
        "Mean / Standard Deviation Error"
    )


    plt.legend()


    plt.grid(
        True,
        axis="y",
    )


    plt.tight_layout()


    plt.savefig(
        BAR_FILE,
        dpi=150,
    )


    plt.close()


    # ========================================================
    # 3.
    # TS1 HISTOGRAM
    #
    # 값이 어느 범위에 많이 분포하는지 비교
    # ========================================================

    plt.figure(
        figsize=(13, 7)
    )


    plt.hist(
        raw_ts1_300,
        bins=20,
        alpha=0.5,
        label="Raw",
    )


    for (
        model_name,
        dataframe,
    ) in generated.items():

        plt.hist(
            dataframe[
                "TS1"
            ],
            bins=20,
            alpha=0.4,
            label=model_name,
        )


    plt.title(
        "HydroTwin TS1 Distribution "
        "- Raw vs V1 vs V2 vs V3 vs V4"
    )


    plt.xlabel(
        "TS1"
    )


    plt.ylabel(
        "Frequency"
    )


    plt.legend()


    plt.grid(
        True,
        axis="y",
    )


    plt.tight_layout()


    plt.savefig(
        HIST_FILE,
        dpi=150,
    )


    plt.close()


    # ========================================================
    # 4.
    # TS1 BOX PLOT
    #
    # 중앙값 / 분포 / 범위 비교
    # ========================================================

    box_data = [
        raw_ts1_300
    ]


    box_labels = [
        "Raw"
    ]


    for (
        model_name,
        dataframe,
    ) in generated.items():

        box_data.append(
            dataframe[
                "TS1"
            ].to_numpy()
        )


        box_labels.append(
            model_name
        )


    plt.figure(
        figsize=(10, 7)
    )


    # --------------------------------------------------------
    # 최신 matplotlib에서는
    # labels가 아니라 tick_labels 사용
    # --------------------------------------------------------

    plt.boxplot(
        box_data,
        tick_labels=box_labels,
    )


    plt.title(
        "HydroTwin TS1 "
        "- Raw vs Generated Models"
    )


    plt.xlabel(
        "Dataset"
    )


    plt.ylabel(
        "TS1"
    )


    plt.grid(
        True,
        axis="y",
    )


    plt.tight_layout()


    plt.savefig(
        BOX_FILE,
        dpi=150,
    )


    plt.close()


    # ========================================================
    # TS1 숫자 통계도 출력
    # ========================================================

    print()
    print("=" * 100)

    print(
        "TS1 300s STATISTICS"
    )

    print("=" * 100)


    ts1_datasets = {
        "RAW":
            raw_ts1_300,
    }


    for (
        model_name,
        dataframe,
    ) in generated.items():

        ts1_datasets[
            model_name
        ] = (
            dataframe[
                "TS1"
            ]
            .to_numpy(
                dtype=np.float32
            )
        )


    for (
        name,
        values,
    ) in ts1_datasets.items():

        print(
            f"{name:3s} "
            f"Mean={np.mean(values):9.3f} "
            f"Std={np.std(values):9.3f} "
            f"Min={np.min(values):9.3f} "
            f"Max={np.max(values):9.3f}"
        )


    # ========================================================
    # Summary CSV
    # ========================================================

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )


    # ========================================================
    # 결과
    # ========================================================

    print()
    print("=" * 100)

    print(
        "FILES SAVED"
    )

    print("=" * 100)


    output_files = [
        LINE_FILE,
        BAR_FILE,
        HIST_FILE,
        BOX_FILE,
        SUMMARY_FILE,
    ]


    for file_path in output_files:

        print(
            f"[SAVED] "
            f"{file_path}"
        )


    print()
    print("=" * 100)

    print(
        "ALL COMPARISON PASS"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
