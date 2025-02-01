# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple, Callable  # 리스트, 튜플, 함수 타입 힌트를 위해 임포트
import heapq  # 최소/최대 힙(우선순위 큐) 연산을 위한 모듈

import torch  # 딥러닝 라이브러리, 텐서 연산을 위해 사용
from torch import distributed as dist  # 분산 학습 관련 기능 사용

from tensordict import TensorDict  # TensorDict: 텐서를 딕셔너리 형태로 관리하기 위한 라이브러리 (추가 기능)
import copy  # 객체 복사를 위한 모듈


def karmarkar_karp(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    """
    # see: https://en.wikipedia.org/wiki/Largest_differencing_method
    Karmarkar-Karp 알고리즘을 변형하여 주어진 시퀀스 길이 목록(seqlen_list)을
    k_partitions개의 그룹으로 나누되, 각 그룹의 합이 균형을 이루도록 분할합니다.
    
    인자:
        seqlen_list: 각 항목의 시퀀스 길이를 담은 리스트
        k_partitions: 나눌 그룹의 수
        equal_size: True이면 각 그룹에 들어가는 항목의 개수도 동일해야 함
    반환:
        partitions: 각 그룹에 해당하는 항목 인덱스의 리스트 목록
    """
    # 내부에서 사용할 Set 클래스를 정의 (각 그룹을 표현)
    class Set:
        def __init__(self) -> None:
            self.sum = 0      # 현재 그룹에 있는 항목들의 합
            self.items = []   # 현재 그룹에 포함된 (인덱스, 값) 튜플들의 리스트

        def add(self, idx: int, val: int):
            # 그룹에 새 항목을 추가하면서 합을 업데이트
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other):
            # 다른 Set의 모든 항목을 현재 Set에 합침
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other):
            # 힙 정렬을 위한 비교 연산자 구현
            # 먼저 합(sum)을 기준으로 비교하고, 같다면 항목의 개수, 그 다음 항목 리스트 자체로 비교
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    # State 클래스는 현재 분할 상태(여러 Set의 모음)를 표현합니다.
    class State:
        def __init__(self, items: List[Tuple[int, int]], k: int) -> None:
            self.k = k  # 전체 그룹 수
            # k 개의 빈 Set을 생성합니다.
            self.sets = [Set() for _ in range(k)]
            # items의 길이가 1 또는 k여야 함 (초기 상태에서는 하나의 그룹 또는 k개의 항목을 각각의 그룹에 할당)
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            # 초기 items를 순서대로 각 Set에 추가합니다.
            for i, (idx, seqlen) in enumerate(items):
                self.sets[i].add(idx=idx, val=seqlen)
            # 각 그룹(세트)을 내림차순(큰 값부터)으로 정렬합니다.
            self.sets = sorted(self.sets, reverse=True)

        def spread(self):
            # 가장 큰 그룹과 가장 작은 그룹의 합 차이를 반환 (균형 정도를 나타냄)
            return self.sets[0].sum - self.sets[-1].sum

        def get_partitions(self):
            # 현재 상태에서 각 그룹의 항목 인덱스만 추출하여 partitions 리스트로 반환
            partitions = []
            for i in range(len(self.sets)):
                cur_partition = []
                for idx, _ in self.sets[i].items:
                    cur_partition.append(idx)
                partitions.append(cur_partition)
            return partitions

        def merge(self, other):
            # 다른 State와의 병합: 양쪽 State의 그룹들을 순서대로 합쳐 새 State를 만듦
            for i in range(self.k):
                # i번째 그룹과 다른 State의 반대쪽 그룹을 합침
                self.sets[i].merge(other.sets[self.k - 1 - i])
            # 병합 후, 다시 내림차순 정렬
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> int:
            # 속성으로도 spread 값을 사용할 수 있게 함
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other):
            # 힙에서 State 객체들을 비교할 때 사용하는 연산자
            # 가장 큰 spread(불균형)를 가진 State가 우선 순위가 높아야 하므로, 비교 반대로 함
            if self.spread != other.spread:
                return self.spread > other.spread
            # spread가 같으면, 가장 큰 그룹을 비교하여 결정
            return self.sets[0] > other.sets[0]

        def __repr__(self) -> str:
            # State 객체를 문자열로 표현 (디버깅용)
            repr_str = "["
            for i in range(self.k):
                if i > 0:
                    repr_str += ","
                repr_str += "{"
                for j, (_, seqlen) in enumerate(self.sets[i].items):
                    if j > 0:
                        repr_str += ","
                    repr_str += str(seqlen)
                repr_str += "}"
            repr_str += "]"
            return repr_str

    # seqlen_list의 각 항목을 (시퀀스 길이, 인덱스) 튜플로 만들고 오름차순 정렬합니다.
    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    # 상태(State) 객체들을 저장할 우선순위 큐(힙)를 초기화합니다.
    states_pq = []
    if equal_size:
        # 각 파티션에 들어갈 항목의 수가 동일해야 하는 경우,
        # 전체 항목 수가 k_partitions의 배수여야 합니다.
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"
        # k_partitions씩 묶어서 초기 State 객체들을 생성합니다.
        for offset in range(0, len(sorted_seqlen_list), k_partitions):
            items = []
            for i in range(k_partitions):
                seqlen, idx = sorted_seqlen_list[offset + i]
                items.append((idx, seqlen))
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        # 각 그룹의 항목 개수가 균등하지 않아도 되는 경우,
        # 각 항목을 개별 State 객체로 생성하여 힙에 추가합니다.
        for seqlen, idx in sorted_seqlen_list:
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    # 힙에 남은 State 객체들을 병합해 하나의 최종 State를 만듭니다.
    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)  # 가장 우선순위가 높은 State
        state1 = heapq.heappop(states_pq)  # 다음 우선순위 State
        # 두 State를 병합
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    final_state = states_pq[0]
    # 최종 State에서 각 그룹의 인덱스 목록을 추출
    partitions = final_state.get_partitions()
    if equal_size:
        # equal_size 옵션이 True인 경우, 각 파티션의 항목 수가 동일한지 검증
        for i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(seqlen_list), f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
    return partitions


def greedy_partition(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    """
    단순한 그리디 알고리즘을 사용하여 시퀀스 길이 목록을 k_partitions개의 그룹으로 분할합니다.
    
    인자:
        seqlen_list: 각 항목의 시퀀스 길이를 담은 리스트
        k_partitions: 분할할 그룹 수
        equal_size: True이면 각 그룹에 들어가는 항목의 수도 동일해야 함
    반환:
        partitions: 각 그룹에 속하는 항목의 인덱스 리스트
    """
    # equal_size가 True이면, bias 값을 주어 각 항목에 동일한 오프셋을 더합니다.
    bias = sum(seqlen_list) + 1 if equal_size else 0
    # 각 항목에 bias를 더한 후 (값, 인덱스) 튜플 리스트 생성
    sorted_seqlen = [(seqlen + bias, i) for i, seqlen in enumerate(seqlen_list)]
    # 각 그룹별 인덱스 리스트 초기화
    partitions = [[] for _ in range(k_partitions)]
    # 각 그룹의 현재 합을 저장하는 리스트 초기화
    partition_sums = [0 for _ in range(k_partitions)]
    # 정렬된 항목들을 순회하며 가장 합이 작은 그룹에 할당
    for seqlen, i in sorted_seqlen:
        min_idx = None
        for j in range(k_partitions):
            if min_idx is None or partition_sums[j] < partition_sums[min_idx]:
                min_idx = j
        partitions[min_idx].append(i)
        partition_sums[min_idx] += seqlen
    if equal_size:
        # equal_size 옵션이 True이면 각 그룹의 크기를 검증
        for i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(seqlen_list), f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
    return partitions


def get_seqlen_balanced_partitions(seqlen_list: List[int], k_partitions: int, equal_size: bool):
    """
    시퀀스 길이의 합이 균형을 이루도록 k_partitions 개의 그룹으로 분할합니다.
    이 함수는 데이터 병렬 학습(dp)이나 마이크로 배치 구성 시 각 그룹의 작업량을 균등하게 하기 위해 사용됩니다.
    
    인자:
        seqlen_list: 각 항목의 시퀀스 길이 리스트
        k_partitions: 생성할 파티션 수
        equal_size: True이면 각 파티션의 항목 개수가 같아야 함, False이면 합만 균형을 맞춤
    반환:
        partitions: 각 파티션에 해당하는 항목 인덱스들의 리스트
    """
    # 전체 항목 수가 k_partitions보다 작으면 안됩니다.
    assert len(seqlen_list) >= k_partitions, f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]"

    def _check_and_sort_partitions(partitions):
        # 파티션의 개수가 k_partitions와 동일한지 확인
        assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"
        seen_idx = set()
        sorted_partitions = [None] * k_partitions
        for i, partition in enumerate(partitions):
            # 각 파티션이 비어있으면 안됩니다.
            assert len(partition) > 0, f"the {i}-th partition is empty"
            for idx in partition:
                seen_idx.add(idx)
            # 각 파티션 내 인덱스를 정렬합니다.
            sorted_partitions[i] = sorted(partition)
        # 모든 항목이 모두 포함되어 있는지 확인
        assert seen_idx == set(range(len(seqlen_list)))
        return sorted_partitions

    # Karmarkar-Karp 알고리즘을 사용하여 초기 분할을 생성
    partitions = karmarkar_karp(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    # 분할 결과를 검증 및 정렬하여 반환
    return _check_and_sort_partitions(partitions)


def log_seqlen_unbalance(seqlen_list: List[int], partitions: List[List[int]], prefix):
    """
    데이터 병렬 학습 시 각 배치(rank)별로 시퀀스 길이 합의 불균형 정도를 로그로 남깁니다.
    
    인자:
        seqlen_list: 원본 시퀀스 길이 리스트
        partitions: 각 파티션(예: dp rank)에 해당하는 인덱스들의 리스트
        prefix: 로그 키 이름의 접두사
    반환:
        각 로그 항목(최소, 최대, 차이, 균형 잡힌 최소/최대, 평균)의 딕셔너리
    """
    k_partition = len(partitions)
    # 배치 크기: 전체 항목 수를 k_partition로 나눈 값 (각 배치당 항목 수)
    batch_size = len(seqlen_list) // k_partition
    min_sum_seqlen = None
    max_sum_seqlen = None
    total_sum_seqlen = 0
    # 전체 시퀀스 길이 합을 배치 단위로 계산
    for offset in range(0, len(seqlen_list), batch_size):
        cur_sum_seqlen = sum(seqlen_list[offset:offset + batch_size])
        if min_sum_seqlen is None or cur_sum_seqlen < min_sum_seqlen:
            min_sum_seqlen = cur_sum_seqlen
        if max_sum_seqlen is None or cur_sum_seqlen > max_sum_seqlen:
            max_sum_seqlen = cur_sum_seqlen
        total_sum_seqlen += cur_sum_seqlen

    # 각 파티션별 균형 잡힌 시퀀스 길이 합 계산
    balanced_sum_seqlen_list = []
    for partition in partitions:
        cur_sum_seqlen_balanced = sum([seqlen_list[i] for i in partition])
        balanced_sum_seqlen_list.append(cur_sum_seqlen_balanced)
    min_sum_seqlen_balanced = min(balanced_sum_seqlen_list)
    max_sum_seqlen_balanced = max(balanced_sum_seqlen_list)

    return {
        f'{prefix}/min': min_sum_seqlen,
        f'{prefix}/max': max_sum_seqlen,
        f'{prefix}/minmax_diff': max_sum_seqlen - min_sum_seqlen,
        f'{prefix}/balanced_min': min_sum_seqlen_balanced,
        f'{prefix}/balanced_max': max_sum_seqlen_balanced,
        f'{prefix}/mean': total_sum_seqlen / len(partitions)
    }


def ceildiv(a, b):
    """
    a를 b로 나눈 후 올림한 값을 반환합니다.
    예: ceildiv(5, 2) -> 3
    """
    return -(a // -b)


def rearrange_micro_batches(batch: TensorDict, max_token_len, dp_group=None):
    """
    주어진 배치(batch)를 micro-batch 단위로 재구성합니다.
    각 micro-batch 내의 총 토큰 수가 max_token_len 이하가 되도록 하며,
    토큰 수가 균형을 이루도록 분할합니다.
    
    인자:
        batch: TensorDict 형태의 배치 데이터 (예: attention_mask, input_ids 등 포함)
        max_token_len: 각 micro-batch에서 허용하는 최대 토큰 수
        dp_group: 분산 학습에서 사용하는 데이터 병렬 그룹 (없으면 None)
    반환:
        micro_batches: 재구성된 micro-batch의 리스트
        micro_bsz_idx: 각 micro-batch에 해당하는 인덱스들의 분할 정보
    """
    # 배치 내 최대 시퀀스 길이 확인
    max_seq_len = batch['attention_mask'].shape[-1]
    # max_token_len은 시퀀스 길이보다 크거나 같아야 합니다.
    assert max_token_len >= max_seq_len, \
        f'max_token_len must be greater than the sequence length. Got {max_token_len=} and {max_seq_len=}'

    # attention_mask에서 각 샘플의 유효 토큰 수를 계산 (1의 합)
    seq_len_effective: torch.Tensor = batch['attention_mask'].sum(dim=1)
    # 전체 유효 토큰 수 계산
    total_seqlen = seq_len_effective.sum().item()
    # 전체 토큰 수를 max_token_len으로 나눈 후 올림한 값이 micro-batch의 개수
    num_micro_batches = ceildiv(total_seqlen, max_token_len)
    if dist.is_initialized():
        # 분산 학습이 초기화된 경우, 모든 프로세스 간 최대 micro-batch 수를 동기화
        num_micro_batches = torch.tensor([num_micro_batches], device='cuda')
        dist.all_reduce(num_micro_batches, op=dist.ReduceOp.MAX, group=dp_group)
        num_micro_batches = num_micro_batches.cpu().item()

    # 유효 토큰 수를 리스트로 변환
    seq_len_effective = seq_len_effective.tolist()
    # micro-batch 수는 전체 샘플 수보다 작거나 같아야 합니다.
    assert num_micro_batches <= len(seq_len_effective)

    # get_seqlen_balanced_partitions 함수를 사용하여 각 micro-batch에 할당할 샘플 인덱스 분할 생성
    micro_bsz_idx = get_seqlen_balanced_partitions(seq_len_effective, num_micro_batches, equal_size=False)

    micro_batches = []

    # 생성된 인덱스 분할에 따라 micro-batch를 구성
    for partition in micro_bsz_idx:
        curr_micro_batch = []
        for idx in partition:
            # 배치에서 해당 인덱스에 해당하는 항목을 선택 (슬라이스 형태로 만듦)
            curr_micro_batch.append(batch[idx:idx + 1])
        # 선택된 샘플들을 하나의 텐서로 결합
        curr_micro_batch = torch.cat(curr_micro_batch)

        micro_batches.append(curr_micro_batch)

    return micro_batches, micro_bsz_idx


def get_reverse_idx(idx_map):
    """
    주어진 인덱스 매핑(idx_map)에 대해, key와 value의 위치를 바꾼 역매핑(reverse index map)을 생성합니다.
    
    인자:
        idx_map: 원래 인덱스 매핑 (예: {0: 2, 1: 0, 2: 1})
    반환:
        reverse_idx_map: key와 value가 뒤바뀐 매핑 (예: {2: 0, 0: 1, 1: 2})
    """
    reverse_idx_map = copy.deepcopy(idx_map)

    for i, idx in enumerate(idx_map):
        reverse_idx_map[idx] = i

    return reverse_idx_map
