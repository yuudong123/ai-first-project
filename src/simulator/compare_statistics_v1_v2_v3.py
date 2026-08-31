from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# HydroTwin
# Raw vs V1 vs V2 vs V3
# Mean / Standard Deviation Comparison
# ============================================================
#
# 사용:
# - mean
# - std
#
# 목적:
# 생성 데이터가 UCI Raw Sensor의 통계 특성과
# 얼마나 비슷한지 숫자로 비교한다.
#
# 재학습 없음.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]


RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "simulator"
    / "uci_1hz_17sensors.npz"
)


V1_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "simulator"
    / "generated_300s_v1.csv"
)


V2_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "simulator"
    / "generated_300s_v2.csv"
)


V3_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "simulator"
    / "generated_300s_v3.csv"
)


OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "simulator"
    / "statistics_v1_v2_v3.csv"
)


# ============================================================
# Raw 데이터
# ============================================================

raw_npz = np.load(RAW_FILE)

raw_data = (
    raw_npz["data"]
    .astype(np.float32)
)

sensor_names = (
    raw_npz["sensor_names"]
    .astype(str)
    .tolist()
)


raw_flat = raw_data.reshape(
    -1,
    len(sensor_names),
)


# ============================================================
# 생성 데이터
# ============================================================

v1 = pd.read_csv(V1_FILE)
v2 = pd.read_csv(V2_FILE)
v3 = pd.read_csv(V3_FILE)


models = {
    "V1": v1,
    "V2": v2,
    "V3": v3,
}


# ============================================================
# 비교
# ============================================================

results = []


print("=" * 120)
print("HydroTwin Raw vs V1 vs V2 vs V3")
print("Mean / Standard Deviation Comparison")
print("=" * 120)


for index, sensor in enumerate(sensor_names):

    raw_values = raw_flat[:, index]

    raw_mean = float(
        np.mean(raw_values)
    )

    raw_std = float(
        np.std(raw_values)
    )


    model_stats = {}


    for model_name, dataframe in models.items():

        values = (
            dataframe[sensor]
            .to_numpy(
                dtype=np.float32
            )
        )


        generated_mean = float(
            np.mean(values)
        )

        generated_std = float(
            np.std(values)
        )


        # ----------------------------------------------------
        # 평균 차이
        # ----------------------------------------------------

        mean_diff = abs(
            generated_mean
            - raw_mean
        )


        # Raw 평균 대비 %
        if abs(raw_mean) > 1e-8:

            mean_error_percent = (
                mean_diff
                / abs(raw_mean)
                * 100
            )

        else:

            mean_error_percent = 0.0


        # ----------------------------------------------------
        # 표준편차 차이
        # ----------------------------------------------------

        std_diff = abs(
            generated_std
            - raw_std
        )


        if raw_std > 1e-8:

            std_error_percent = (
                std_diff
                / raw_std
                * 100
            )

        else:

            std_error_percent = 0.0


        model_stats[
            model_name
        ] = {

            "mean": generated_mean,

            "std": generated_std,

            "mean_error_percent":
                mean_error_percent,

            "std_error_percent":
                std_error_percent,
        }


    # ========================================================
    # 센서별 가장 가까운 모델
    # ========================================================

    best_mean_model = min(
        model_stats,
        key=lambda name:
        model_stats[name][
            "mean_error_percent"
        ],
    )


    best_std_model = min(
        model_stats,
        key=lambda name:
        model_stats[name][
            "std_error_percent"
        ],
    )


    print()
    print(
        f"[{sensor}] "
        f"Raw Mean={raw_mean:.4f} "
        f"Raw Std={raw_std:.4f}"
    )


    for model_name in [
        "V1",
        "V2",
        "V3",
    ]:

        stats = model_stats[
            model_name
        ]


        print(
            f"  {model_name} "
            f"Mean={stats['mean']:10.4f} "
            f"Mean Error={stats['mean_error_percent']:7.2f}% "
            f"Std={stats['std']:10.4f} "
            f"Std Error={stats['std_error_percent']:7.2f}%"
        )


    print(
        f"  BEST Mean : "
        f"{best_mean_model}"
    )

    print(
        f"  BEST Std  : "
        f"{best_std_model}"
    )


    row = {
        "sensor": sensor,
        "raw_mean": raw_mean,
        "raw_std": raw_std,
        "best_mean": best_mean_model,
        "best_std": best_std_model,
    }


    for model_name in [
        "V1",
        "V2",
        "V3",
    ]:

        stats = model_stats[
            model_name
        ]


        row[
            f"{model_name}_mean"
        ] = stats["mean"]


        row[
            f"{model_name}_std"
        ] = stats["std"]


        row[
            f"{model_name}_mean_error_pct"
        ] = stats[
            "mean_error_percent"
        ]


        row[
            f"{model_name}_std_error_pct"
        ] = stats[
            "std_error_percent"
        ]


    results.append(row)


# ============================================================
# 전체 결과 DataFrame
# ============================================================

result_df = pd.DataFrame(
    results
)


# ============================================================
# 모델별 전체 평균 오차
#
# 17개 Sensor의 %
# 평균값으로 단순 비교한다.
#
# 별도의 새로운 AI 평가기법이 아니라
# 위에서 계산한 평균/표준편차 오차의 요약이다.
# ============================================================

print()
print()
print("=" * 120)
print("OVERALL RESULT")
print("=" * 120)


for model_name in [
    "V1",
    "V2",
    "V3",
]:

    mean_error = (
        result_df[
            f"{model_name}_mean_error_pct"
        ]
        .mean()
    )


    std_error = (
        result_df[
            f"{model_name}_std_error_pct"
        ]
        .mean()
    )


    print(
        f"{model_name} "
        f"Average Mean Error = "
        f"{mean_error:7.2f}% | "
        f"Average Std Error = "
        f"{std_error:7.2f}%"
    )


# ============================================================
# 몇 개 Sensor에서 각각 승리했는지
# ============================================================

print()
print("=" * 120)
print("BEST MODEL COUNTS")
print("=" * 120)


for model_name in [
    "V1",
    "V2",
    "V3",
]:

    mean_wins = int(
        (
            result_df["best_mean"]
            == model_name
        ).sum()
    )


    std_wins = int(
        (
            result_df["best_std"]
            == model_name
        ).sum()
    )


    print(
        f"{model_name} "
        f"Mean Best = "
        f"{mean_wins:2d}/17 | "
        f"Std Best = "
        f"{std_wins:2d}/17"
    )


# ============================================================
# CSV 저장
# ============================================================

result_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)


print()
print(
    f"[SAVED] {OUTPUT_FILE}"
)

print()
print("=" * 120)
print("STATISTICS COMPARISON PASS")
print("=" * 120)
