"""안정 기준 운전과 불안정 초기 시계열을 분리하는 재현 가능한 시나리오."""
import random
import numpy as np


class SeedSchedule:
    def __init__(self, profile, initial_seconds=120, segment_seconds=60, seed=None, reference_seed=None):
        profile = np.asarray(profile)
        if profile.ndim != 2 or profile.shape[1] != 5 or segment_seconds < 10 or initial_seconds < 120:
            raise ValueError('5개 프로필 라벨, 120초 이상 기준 구간, 10초 이상 시나리오 구간이 필요합니다.')
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
        self.segment_seconds = segment_seconds
        self.last_block = None
        self.current_seed = self.reference_seed
        self.segment_id = 0

    def select(self, elapsed):
        block = -1 if elapsed < self.initial_seconds else (elapsed-self.initial_seconds)//self.segment_seconds
        reference = block < 0 or block % 3 != 2
        if block != self.last_block:
            previous = self.current_seed
            self.current_seed = self.reference_seed if reference else self.random.choice(self.unstable_pool)
            if self.current_seed != previous:
                self.segment_id += 1
            self.last_block = block
        return self.current_seed, self.segment_id, reference


def window_discontinuity(previous_run, previous_event, previous_segment, data):
    """초기 시계열을 바꾼 경계의 서로 다른 운전 데이터를 한 평균에 섞지 않는다."""
    return (data['run_id'] != previous_run or
            (previous_event is not None and data['event_id'] != previous_event+1) or
            data.get('segment_id',0) != previous_segment)
