from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# HydroTwin
# V1 vs V2 Generator Comparison
#
# 사용 기술:
# - pandas
# - matplotlib
#
# 비교 대상:
# - V1 300초 생성 데이터
# - V2 300초 생성 데이터
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "simulator"
    / "graphs"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 데이터 로드
# ============================================================

v1 = pd.read_csv(V1_FILE)
v2 = pd.read_csv(V2_FILE)


print("=" * 70)
print("HydroTwin V1 vs V2 Comparison")
print("=" * 70)

print(
    f"V1 Shape : {v1.shape}"
)

print(
    f"V2 Shape : {v2.shape}"
)


# ============================================================
# 1. TS1 300초 시계열 비교
#
# V1:
# 장시간 생성 중 값이 한쪽으로 이동하는지 확인
#
# V2:
# NORMAL 상태에서 안정적으로 유지되는지 확인
# ============================================================

plt.figure(
    figsize=(14, 6)
)


plt.plot(
    v1["generated_second"],
    v1["TS1"],
    label="V1 TS1",
)


plt.plot(
    v2["generated_second"],
    v2["TS1"],
    label="V2 TS1",
)


plt.title(
    "HydroTwin V1 vs V2 - TS1 300s"
)

plt.xlabel(
    "Generated Time (sec)"
)

plt.ylabel(
    "TS1 Temperature"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


ts1_output = (
    OUTPUT_DIR
    / "v1_v2_ts1_300s.png"
)


plt.savefig(
    ts1_output,
    dpi=150,
)


plt.show()


print(
    f"[SAVED] {ts1_output}"
)


# ============================================================
# 2. 전체 17 Sensor Std Ratio 비교
#
# Raw UCI 대비 생성 데이터의 변동성이
# 어느 정도 유지되는지 비교한다.
#
# 1.0에 가까울수록
# Raw 데이터와 비슷한 변동폭
# ============================================================


SENSORS = [
    "PS1",
    "PS2",
    "PS3",
    "PS4",
    "PS5",
    "PS6",
    "EPS1",
    "FS1",
    "FS2",
    "TS1",
    "TS2",
    "TS3",
    "TS4",
    "VS1",
    "CE",
    "CP",
    "SE",
]


RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "simulator"
    / "uci_1hz_17sensors.npz"
)


import numpy as np


raw_data = np.load(
    RAW_FILE
)["data"]


raw_flat = raw_data.reshape(
    -1,
    len(SENSORS),
)


v1_std_ratio = []

v2_std_ratio = []


for index, sensor in enumerate(
    SENSORS
):

    raw_std = raw_flat[
        :,
        index,
    ].std()


    v1_std = (
        v1[sensor]
        .std()
    )


    v2_std = (
        v2[sensor]
        .std()
    )


    v1_std_ratio.append(
        v1_std / raw_std
    )


    v2_std_ratio.append(
        v2_std / raw_std
    )


plt.figure(
    figsize=(15, 6)
)


plt.plot(
    SENSORS,
    v1_std_ratio,
    marker="o",
    label="V1",
)


plt.plot(
    SENSORS,
    v2_std_ratio,
    marker="o",
    label="V2",
)


plt.axhline(
    y=1.0,
    linestyle="--",
    label="Raw UCI = 1.0",
)


plt.title(
    "HydroTwin V1 vs V2 - Sensor Variability"
)

plt.xlabel(
    "Sensor"
)

plt.ylabel(
    "Std Ratio (Generated / Raw)"
)

plt.xticks(
    rotation=45
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


std_output = (
    OUTPUT_DIR
    / "v1_v2_std_ratio.png"
)


plt.savefig(
    std_output,
    dpi=150,
)


plt.show()


print(
    f"[SAVED] {std_output}"
)


print()
print("=" * 70)
print("COMPARISON COMPLETE")
print("=" * 70)
