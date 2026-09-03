"""서버 배포 시 필수 모델과 Unity 자산 검증."""
import pytest
from src.runtime import server_entry
from src import hydrotwin_pipeline as p


def test_inference_requires_ten_seconds(monkeypatch):
    monkeypatch.setattr(p,'load_model_bundle',lambda:{'window_sec':60})
    with pytest.raises(ValueError,match='10초'):
        server_entry.validate_inference()


def test_inference_requires_all_five_targets(monkeypatch):
    bundle={'window_sec':10,'feature_names':p.MEAN_FEATURE_COLUMNS,'models':{'pump':object()}}
    monkeypatch.setattr(p,'load_model_bundle',lambda:bundle)
    with pytest.raises(ValueError,match='stable_flag'):
        server_entry.validate_inference()


def test_inference_validates_prediction_and_writable_state(tmp_path,monkeypatch):
    from src.runtime import inference
    bundle={'window_sec':10,'feature_names':p.MEAN_FEATURE_COLUMNS,
            'models':dict.fromkeys(p.TARGET_ORDER,object())}
    calls=[]
    monkeypatch.setattr(p,'load_model_bundle',lambda:bundle)
    monkeypatch.setattr(inference,'diagnose',lambda saved,rows:calls.append((saved,rows)))
    monkeypatch.setattr(server_entry,'STATE_DIR',tmp_path/'runtime')
    server_entry.validate_inference()
    assert calls[0][0] is bundle and len(calls[0][1])==10
    assert list((tmp_path/'runtime').iterdir())==[]


def test_api_requires_all_unity_files(tmp_path,monkeypatch):
    from src.monitoring import sensor_bands
    monkeypatch.setattr(sensor_bands,'load_sensor_bands',lambda:{})
    monkeypatch.setenv('UNITY_WEBGL_PATH',str(tmp_path))
    build=tmp_path/'Build';build.mkdir()
    for suffix in ('loader.js','data.gz','framework.js.gz'):
        (build/f'pro-build.{suffix}').write_bytes(b'asset')
    with pytest.raises(FileNotFoundError,match='wasm'):
        server_entry.validate_api()
    (build/'pro-build.wasm.gz').write_bytes(b'asset')
    server_entry.validate_api()


def test_api_does_not_hide_missing_reference_data(monkeypatch):
    from src.monitoring import sensor_bands
    def missing():
        raise FileNotFoundError('기준 데이터 누락')
    monkeypatch.setattr(sensor_bands,'load_sensor_bands',missing)
    with pytest.raises(FileNotFoundError,match='기준 데이터'):
        server_entry.validate_api()


@pytest.mark.parametrize('module',['src.runtime.multi_inference','src.runtime.inference'])
def test_same_entry_supports_remote_and_local_inference(monkeypatch,module):
    calls=[]
    monkeypatch.setattr(server_entry,'validate_inference',lambda:None)
    monkeypatch.setenv('HYDROTWIN_INFERENCE_MODULE',module)
    monkeypatch.setattr(server_entry.sys,'argv',['entry','inference'])
    monkeypatch.setattr(server_entry.os,'execvp',lambda executable,args:calls.append(args))
    server_entry.main()
    assert calls[0][-1]==module
