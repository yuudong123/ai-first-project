"""60~1,200초 무작위 간격으로 절대 offset을 교체하는 계절 시나리오."""
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioConfig:
    minimum_interval: int = 60
    maximum_interval: int = 1200
    temperature_min: float = -4.0
    temperature_max: float = 4.0
    pressure_percent: float = 10.0
    ramp_seconds: int = 30
    initial_normal_seconds: int = 120

    def __post_init__(self):
        if not 60 <= self.minimum_interval <= self.maximum_interval <= 1200:
            raise ValueError('드리프트 시작 간격은 60~1,200초 범위여야 합니다.')
        if self.temperature_min > self.temperature_max or self.pressure_percent < 0:
            raise ValueError('offset 범위가 올바르지 않습니다.')
        if not 0 < self.ramp_seconds <= self.minimum_interval:
            raise ValueError('이동 시간은 양수이고 최소 발생 간격 이하여야 합니다.')
        if self.initial_normal_seconds < 120:
            raise ValueError('초기 기준 통계를 위해 정상 구간 120초가 필요합니다.')


class RandomSeason:
    def __init__(self, config, seed=None):
        self.config = config
        self.random = random.Random(seed)
        self.next_start = config.initial_normal_seconds
        self.start = 0
        self.source = (0.0, 0.0)
        self.target = (0.0, 0.0)
        self.event_id = 0

    def update(self, elapsed):
        # 절대 offset으로 교체하므로 변화가 반복되어도 상·하한 밖으로 누적되지 않는다.
        current = self.value(elapsed)
        if elapsed >= self.next_start:
            self.source = current
            self.target = (
                self.random.uniform(self.config.temperature_min, self.config.temperature_max),
                self.random.uniform(-self.config.pressure_percent, self.config.pressure_percent),
            )
            self.start = elapsed
            self.next_start = elapsed + self.random.randint(self.config.minimum_interval, self.config.maximum_interval)
            self.event_id += 1
        return self.value(elapsed)

    def value(self, elapsed):
        alpha = min(1.0, max(0.0, (elapsed - self.start) / self.config.ramp_seconds))
        return tuple(a + (b-a)*alpha for a,b in zip(self.source, self.target))
