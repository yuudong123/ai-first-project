"""서버의 필수 자산을 검사한 뒤 지정 서비스를 실행한다. Kafka를 생성·수정하지 않는다."""
import os
import sys
from pathlib import Path
from tempfile import TemporaryFile

from src.runtime.common import STATE_DIR


def validate_inference():
    from src import hydrotwin_pipeline as p
    from src.runtime.inference import diagnose
    bundle = p.load_model_bundle()
    if bundle['window_sec'] != 10 or list(bundle['feature_names']) != p.MEAN_FEATURE_COLUMNS:
        raise ValueError('센서 순서가 일치하는 10초 평균 모델이 필요합니다.')
    if set(bundle['models']) != set(p.TARGET_ORDER):
        raise ValueError('부품 4개와 stable_flag 모델이 모두 필요합니다.')
    # 모델과 설치된 라이브러리의 추론·설명 호환성을 시작 시 확인한다.
    diagnose(bundle, [[0.0]*17 for _ in range(10)])
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryFile(dir=STATE_DIR) as probe:
        probe.write(b'check')


def validate_api():
    from src.monitoring.sensor_bands import load_sensor_bands
    load_sensor_bands()
    root = Path(os.environ['UNITY_WEBGL_PATH']) / 'Build'
    for suffix in ('loader.js', 'data.gz', 'framework.js.gz', 'wasm.gz'):
        path = root / f'pro-build.{suffix}'
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f'Unity 빌드 파일 누락: {path}')


def main():
    mode = sys.argv[1] if len(sys.argv)>1 else ''
    if mode == 'inference':
        validate_inference()
        command = [sys.executable, str(Path(__file__).resolve().parents[2]/'kafka/consumer.py')]
    elif mode == 'api':
        validate_api()
        command = [sys.executable, '-m', 'uvicorn', 'api.main:app', '--host', '0.0.0.0', '--port', '8000']
    else:
        raise SystemExit('실행 대상은 inference 또는 api여야 합니다.')
    os.execvp(command[0], command)


if __name__ == '__main__':
    main()
