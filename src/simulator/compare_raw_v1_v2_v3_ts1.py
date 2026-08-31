from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 경로 설정
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[2]

RAW_NPZ_PATH = BASE_DIR / "data" / "processed" / "simulator" / "uci_1hz_17sensors.npz"
V1_PATH = BASE_DIR / "data" / "processed" / "simulator" / "generated_300s.csv"
V2_PATH = BASE_DIR / "data" / "processed" / "simulator" / "generated_300s_v2.csv"
V3_PATH = BASE_DIR / "data" / "processed" / "simulator" / "generated_300s_v3.csv"

GRAPH_DIR = BASE_DIR / "data" / "processed" / "simulator" / "graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = GRAPH_DIR / "raw_v1_v2_v3_ts1_31_60s.png"

# ============================================================
# 센서 순서
# ============================================================
SENSOR_NAMES = [
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6",
    "EPS1", "FS1", "FS2",
    "TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE"
]

# ============================================================
# Raw 데이터 로드
# ============================================================
npz = np.load(RAW_NPZ_PATH, allow_pickle=True)

if "data" in npz.files:
    raw_data = npz["data"]
elif "arr_0" in npz.files:
    raw_data = npz["arr_0"]
else:
    raise ValueError(f"NPZ 파일 구조를 알 수 없습니다. keys={npz.files}")

ts1_idx = SENSOR_NAMES.index("TS1")

# 첫 번째 cycle의 31~60초 구간 사용
raw_ts1 = raw_data[0, 30:60, ts1_idx]

# ============================================================
# 생성 데이터 로드
# ============================================================
df_v1 = pd.read_csv(V1_PATH)
df_v2 = pd.read_csv(V2_PATH)
df_v3 = pd.read_csv(V3_PATH)

# TS1 컬럼 확인
for name, df in [("V1", df_v1), ("V2", df_v2), ("V3", df_v3)]:
    if "TS1" not in df.columns:
        raise ValueError(f"{name} CSV에 TS1 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

v1_ts1 = df_v1["TS1"].iloc[30:60].reset_index(drop=True)
v2_ts1 = df_v2["TS1"].iloc[30:60].reset_index(drop=True)
v3_ts1 = df_v3["TS1"].iloc[30:60].reset_index(drop=True)

# ============================================================
# 그래프 그리기
# ============================================================
time_axis = list(range(31, 61))

plt.figure(figsize=(14, 6))
plt.plot(time_axis, raw_ts1, marker="o", label="Raw TS1")
plt.plot(time_axis, v1_ts1, marker="o", label="V1 Generated")
plt.plot(time_axis, v2_ts1, marker="o", label="V2 Generated")
plt.plot(time_axis, v3_ts1, marker="o", label="V3 Generated")

plt.title("HydroTwin TS1 - Raw vs V1 vs V2 vs V3")
plt.xlabel("Time (sec)")
plt.ylabel("TS1")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(OUT_PATH, dpi=150)
plt.show()

print("=" * 70)
print("HydroTwin TS1 Comparison")
print("=" * 70)
print(f"[SAVED] {OUT_PATH}")
print("=" * 70)
