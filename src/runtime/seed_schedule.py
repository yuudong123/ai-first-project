"""설비마다 독립적인 간격으로 안정·불안정 운전을 전환하는 시나리오."""
import random
import numpy as np


class SeedSchedule:
    def __init__(
        self,
        profile,
        initial_seconds=120,
        minimum_interval_seconds=60,
        maximum_interval_seconds=600,
        seed=None,
        reference_seed=None,
    ):
        profile = np.asarray(profile)
        if (
            profile.ndim != 2
            or profile.shape[1] != 5
            or initial_seconds < 120
            or not 60 <= minimum_interval_seconds <= maximum_interval_seconds <= 600
        ):
            raise ValueError(
                '5개 프로필 라벨, 120초 이상 기준 구간, '
                '60~600초 범위의 운전 전환 간격이 필요합니다.'
            )
        stable = np.flatnonzero(np.all(profile == [100,100,0,130,0],axis=1))
        unstable = np.flatnonzero(profile[:,4] == 1)
        if len(stable)==0 or len(unstable)==0:
            raise ValueError('안정 기준 시계열과 불안정 초기 시계열이 모두 필요합니다.')
        if reference_seed is not None and reference_seed not in stable:
            raise ValueError('기준 초기값은 부품 4개 정상 및 안정 상태여야 합니다.')
        self.reference_seed = int(stable[0] if reference_seed is None else reference_seed)
        self.unstable_pool = [int(i) for i in unstable]
        self.random = random.Random(seed)
        self.initial_seconds = initial_seconds
        self.minimum_interval_seconds = minimum_interval_seconds
        self.maximum_interval_seconds = maximum_interval_seconds
        self.current_seed = self.reference_seed
        self.reference = True
        self.segment_id = 0
        self.next_transition = initial_seconds + self.random.randint(
            minimum_interval_seconds,
            maximum_interval_seconds,
        )
        self.previous_elapsed = -1

    def select(self, elapsed):
        if elapsed < self.previous_elapsed:
            raise ValueError('운전 시나리오 시간은 이전 호출보다 작아질 수 없습니다.')
        while elapsed >= self.next_transition:
            self.reference = not self.reference
            self.current_seed = (
                self.reference_seed
                if self.reference
                else self.random.choice(self.unstable_pool)
            )
            self.segment_id += 1
            self.next_transition += self.random.randint(
                self.minimum_interval_seconds,
                self.maximum_interval_seconds,
            )
        self.previous_elapsed = elapsed
        return self.current_seed, self.segment_id, self.reference


def window_discontinuity(previous_run, previous_event, previous_segment, data):
    """초기 시계열을 바꾼 경계의 서로 다른 운전 데이터를 한 평균에 섞지 않는다."""
    return (data['run_id'] != previous_run or
            (previous_event is not None and data['event_id'] != previous_event+1) or
            data.get('segment_id',0) != previous_segment)
