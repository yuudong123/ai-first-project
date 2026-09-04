"""Jenkins에서 설비 3대가 확인한 드리프트의 공통 모델을 안전하게 재학습한다."""
import argparse
import json
import os
import shutil
import time
from pathlib import Path
from src.runtime.common import STATE_DIR, read_state, write_state, now
from src.model.retrain import run_retraining, RetrainConfig, promote_candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mark-failed',action='store_true')
    args = parser.parse_args()
    status = read_state('retraining.json')
    if args.mark_failed:
        if status.get('status') in ('queued','running'):
            write_state('retraining.json',{**status,'status':'failed','message':'Jenkins 작업 실패. 콘솔 로그를 확인하세요.','updated_at':now()})
        return
    # 컨테이너의 파일 잠금으로 Jenkins 외부의 중복 수동 실행도 막는다.
    import fcntl
    STATE_DIR.mkdir(parents=True,exist_ok=True)
    with (STATE_DIR/'training.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        request = read_state('retrain_request.json')
        if not request or status.get('request_id')!=request['request_id'] or status.get('status')!='queued':
            print('실행할 새 재학습 요청이 없습니다.',flush=True)
            return
        expected_equipment = {'station-01','station-02','station-03'}
        if set(request.get('equipment_ids',[])) != expected_equipment or request.get('confirmation')!='all_three_equipment':
            raise RuntimeError('설비 3대가 함께 확인한 드리프트 요청만 재학습할 수 있습니다.')
        status = {**status,'status':'running','updated_at':now()}
        write_state('retraining.json',status)
        try:
            config = RetrainConfig(auto_promote=False,final_window_sec=10)
            report = run_retraining(config=config,drift_context=request)
            outcome = 'rejected'
            current = read_state('monitor.json')
            reference = read_state('reference.json').get('reference',{}).get('sensors',{})
            changed = (current.get('run_id') != request['run_id'] or
                not set(request['sensor_offsets']).issubset(current.get('estimated_offsets',{}))) or any(
                abs(current.get('estimated_offsets',{}).get(sensor,offset)-offset) >
                (0.5 if sensor.startswith('TS') else max(abs(reference.get(sensor,{}).get('mean',0))*0.02,0.02))
                for sensor,offset in request['sensor_offsets'].items()
            )
            if changed:
                outcome = 'superseded'
                report['rejection_reasons'].append('학습 중 계절 분포가 다시 바뀌어 현재 후보를 배포하지 않습니다.')
            if report['accepted'] and not changed:
                # 후보 파일이 실제 추론 계약을 만족하는지 먼저 확인한다.
                from src import hydrotwin_pipeline as p
                from src.runtime.inference import diagnose
                sample = read_state('latest.json')
                if not sample or sample.get('run_id')!=request['run_id']:
                    raise RuntimeError('학습 중 생성기 실행이 바뀌어 배포를 중단했습니다.')
                equipment_states = sample.get('equipment_states',[])
                if {item.get('equipment_id') for item in equipment_states} != expected_equipment:
                    raise RuntimeError('후보 모델 검사에 필요한 설비 3대의 최신값이 없습니다.')
                candidate_bundle = p.load_model_bundle(report['candidate_path'])
                for item in equipment_states:
                    rows = [[item['sensors'][sensor] for sensor in p.SENSOR_NAMES]]*10
                    diagnose(candidate_bundle,rows)
                backup = promote_candidate(Path(report['candidate_path']),config.production_model_path)
                expected = str(config.production_model_path.stat().st_mtime_ns)
                deadline = time.monotonic()+30
                while time.monotonic()<deadline:
                    current_state = read_state('latest.json')
                    versions = {item.get('model_version') for item in current_state.get('equipment_states',[])}
                    if current_state.get('model_version')==expected and versions=={expected}:
                        outcome = 'promoted'
                        break
                    time.sleep(1)
                if outcome!='promoted':
                    if backup:
                        temporary = config.production_model_path.with_suffix('.rollback.tmp')
                        shutil.copy2(backup,temporary)
                        os.replace(temporary,config.production_model_path)
                    raise RuntimeError('새 모델 반영 확인 실패. 기존 모델로 복구했습니다.')
                report['promoted'] = True
                report['backup_path'] = str(backup) if backup else None
            write_state('retraining.json',{**status,'status':outcome,'updated_at':now(),'report':report})
            Path(report['report_path']).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps({'status':outcome,'candidate':report['candidate_metrics_by_environment'],
                              'reasons':report['rejection_reasons']},ensure_ascii=False),flush=True)
        except Exception as error:
            write_state('retraining.json',{**status,'status':'failed','updated_at':now(),'message':str(error)})
            raise


if __name__ == '__main__':
    main()
