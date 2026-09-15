"""실시간 서비스에서 공유하는 파일 저장과 Kafka 연결 함수."""
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = ROOT / 'artifacts' / 'runtime'
SENSORS = ('PS1','PS2','PS3','PS4','PS5','PS6','EPS1','FS1','FS2','TS1','TS2','TS3','TS4','VS1','CE','CP','SE')
BROKER = os.getenv('KAFKA_BROKER', 'kafka:29092')
TOPIC = os.getenv('KAFKA_TOPIC', 'hydraulic.sensor.multi.raw')


def now():
    return datetime.now(timezone.utc).isoformat()


def read_state(name, default=None):
    try:
        return json.loads((STATE_DIR / name).read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return {} if default is None else default


def write_state(name, value):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / name
    temp = path.with_name(f'{path.name}.{uuid.uuid4().hex}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def consumer(group):
    from kafka import KafkaConsumer
    return KafkaConsumer(TOPIC, bootstrap_servers=BROKER, group_id=group,
                         auto_offset_reset='latest', enable_auto_commit=True,
                         value_deserializer=lambda b: json.loads(b.decode('utf-8')))


def age(timestamp):
    return (datetime.now(timezone.utc) - datetime.fromisoformat(timestamp)).total_seconds()
