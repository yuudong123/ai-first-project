import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATOR_SOURCE = PROJECT_ROOT / "src" / "simulator"
sys.path.insert(0, str(SIMULATOR_SOURCE))

from v5_multi_station_utils import (  # noqa: E402
    EQUIPMENT_IDS,
    V5StationRuntime,
    MixedSeedController,
    assert_station_values_differ,
    clip_for_six_decimal_raw_range,
    create_multi_raw_message,
    validate_multi_raw_message,
)


class IdentityScaler:
    def transform(self, values):
        return values

    def inverse_transform(self, values):
        return values


class ZeroOffsetModel:
    def predict(self, values, verbose=0):
        return np.zeros((values.shape[0], 17), dtype=np.float32)


def make_runtime(equipment_id, anchor, model=None):
    seed = np.tile(np.asarray(anchor, dtype=np.float32), (30, 1))
    return V5StationRuntime(
        equipment_id=equipment_id,
        seed_record=1790,
        model=model or ZeroOffsetModel(),
        input_scaler=IdentityScaler(),
        offset_scaler=IdentityScaler(),
        sensor_min=np.full(17, -1000.0),
        sensor_max=np.full(17, 1000.0),
        seed_window=seed,
        ps4_index=3,
    )


def test_station_runtime_state_is_independent():
    shared_model = ZeroOffsetModel()
    first = make_runtime("station-01", np.arange(17), shared_model)
    second = make_runtime("station-02", np.arange(17) + 100, shared_model)
    second_window_before = second.sensor_window.copy()

    first_values = first.predict_next()

    assert np.array_equal(first_values, np.arange(17))
    assert np.array_equal(second.sensor_window, second_window_before)
    assert first.sensor_window is not second.sensor_window
    assert first.anchor is not second.anchor
    assert first.model is second.model


def test_multi_raw_contract_has_only_allowed_fields():
    sensors = [f"S{index}" for index in range(17)]
    message = create_multi_raw_message(
        "station-01", "2026-09-01T12:00:00+09:00", sensors, np.arange(17)
    )
    validate_multi_raw_message(message, sensors)
    assert set(message) == {"equipment_id", "timestamp", "sensors"}
    assert len(message["sensors"]) == 17

    message["prediction"] = "NORMAL"
    with pytest.raises(ValueError):
        validate_multi_raw_message(message, sensors)


def test_station_difference_check_rejects_identical_values():
    values = {
        equipment_id: np.arange(17, dtype=np.float64) + index
        for index, equipment_id in enumerate(EQUIPMENT_IDS)
    }
    assert_station_values_differ(values)
    values["station-03"] = values["station-01"].copy()
    with pytest.raises(ValueError):
        assert_station_values_differ(values)


def test_six_decimal_values_stay_strictly_inside_raw_bounds():
    raw_min = np.asarray([134.84780883789062] * 17)
    raw_max = np.asarray([191.6153106689453] * 17)
    values = clip_for_six_decimal_raw_range(
        np.asarray([200.0] * 17), raw_min, raw_max
    )
    assert (values >= raw_min).all()
    assert (values <= raw_max).all()
    assert values[0] == 191.615310


def mixed_controller(seed=42):
    profiles=np.tile([100,100,0,130,0],(12,1))
    profiles[3:,4]=1
    raw=np.asarray([np.full((60,17),index,dtype=np.float32) for index in range(12)])
    model=ZeroOffsetModel()
    runtimes={equipment:make_runtime(equipment,np.full(17,index),model)
              for index,equipment in enumerate(EQUIPMENT_IDS)}
    for index,runtime in enumerate(runtimes.values()):
        runtime.seed_record=index
    return MixedSeedController(raw,profiles,runtimes,seed=seed)


def test_mixed_seeds_follow_120_initial_then_120_stable_60_unstable():
    controller=mixed_controller()
    counts={equipment:{0:0,1:0} for equipment in EQUIPMENT_IDS}
    shared_model=controller.runtimes['station-01'].model
    stable_runtime=controller.runtimes['station-01']
    for second in range(480):
        controller.advance(second)
        expected=1 if second>=120 and (second-120)//60%3==2 else 0
        records=[]
        for equipment,choice in controller.choices.items():
            assert choice['seed_stable_flag']==expected
            assert controller.runtimes[equipment].model is shared_model
            counts[equipment][expected]+=int(second>=120)
            values=controller.runtimes[equipment].predict_next()
            assert np.all(values==choice['seed_record'])
            records.append(choice['seed_record'])
        assert len(set(records))==3
        if second==239:
            assert controller.runtimes['station-01'] is stable_runtime
    assert all(count=={0:240,1:120} for count in counts.values())


def test_mixed_transition_resets_phase_and_reuses_reference_on_return():
    controller=mixed_controller()
    controller.advance(0)
    controller.runtimes['station-01'].predict_next()
    controller.advance(240)
    assert all(choice['segment_id']==1 for choice in controller.choices.values())
    assert all(runtime.cycle_position==29 for runtime in controller.runtimes.values())
    controller.advance(300)
    assert [runtime.seed_record for runtime in controller.runtimes.values()]==[0,1,2]
    assert all(choice['segment_id']==2 for choice in controller.choices.values())


def test_mixed_seed_selection_is_reproducible():
    first,second=mixed_controller(),mixed_controller()
    for elapsed in (0,240,300,420,480,600):
        first.advance(elapsed);second.advance(elapsed)
        assert first.choices==second.choices


def test_multi_message_accepts_boundary_without_label_leakage():
    sensors=[f'S{index}' for index in range(17)]
    message=create_multi_raw_message('station-01','2026-09-01T12:00:00+09:00',sensors,np.arange(17),
                                     run_id='run-a',event_id=241,segment_id=1,reference_context=False)
    validate_multi_raw_message(message,sensors)
    assert set(message)=={'equipment_id','timestamp','sensors','run_id','event_id','segment_id','reference_context'}
    with pytest.raises(ValueError):
        validate_multi_raw_message({**message,'seed_stable_flag':1},sensors)
    with pytest.raises(ValueError):
        validate_multi_raw_message({**message,'segment_id':None},sensors)
