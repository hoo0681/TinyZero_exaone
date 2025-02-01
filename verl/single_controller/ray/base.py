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

import time  # 시간 관련 함수들을 사용하기 위해 time 모듈을 가져옴
from typing import Dict, List, Any, Tuple  # 타입 힌트를 위해 Dict, List, Any, Tuple을 가져옴

import ray  # Ray 분산 처리 프레임워크를 사용하기 위한 모듈
from ray.util import list_named_actors  # 현재 생성된 이름 있는 actor 목록을 가져오는 함수
from ray.util.placement_group import placement_group, PlacementGroup  # 자원 할당을 위한 placement group 관련 함수와 클래스
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy, NodeAffinitySchedulingStrategy
# PlacementGroupSchedulingStrategy: placement group 내에서 스케줄링 전략을 정의
# NodeAffinitySchedulingStrategy: 특정 노드에 affinity(친화도)를 부여하는 스케줄링 전략을 정의
from ray.experimental.state.api import get_actor  # actor의 상태 정보를 가져오는 함수

from verl.single_controller.base import WorkerGroup, ResourcePool, ClassWithInitArgs, Worker
# WorkerGroup, ResourcePool, ClassWithInitArgs, Worker 등은 분산 워커 관리를 위한 베이스 클래스들임

__all__ = ['Worker']  # 이 모듈에서 외부에 공개할 객체 목록

# ---------------------------------------------------------------------------
# 유틸리티 함수: 지정된 길이의 랜덤 문자열 생성
# ---------------------------------------------------------------------------
def get_random_string(length: int) -> str:
    """
    지정된 길이(length)의 임의의 영문자(대소문자)와 숫자로 구성된 문자열을 생성하는 함수입니다.
    이 문자열은 주로 고유한 이름 접두사로 사용됩니다.
    """
    import random
    import string
    # 영문 대소문자와 숫자를 모두 포함한 문자열 생성
    letters_digits = string.ascii_letters + string.digits
    # length 길이만큼 무작위로 선택하여 문자열로 합침
    return ''.join(random.choice(letters_digits) for _ in range(length))


# ---------------------------------------------------------------------------
# 원격 함수 호출을 위한 래퍼 함수 생성기
# ---------------------------------------------------------------------------
def func_generator(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking):
    """
    주어진 인자와 옵션들을 기반으로, 원격 함수 호출을 위한 래퍼 함수를 생성합니다.
    
    매개변수:
      - self: 클래스 인스턴스 (메소드가 바인딩될 대상)
      - method_name: 호출할 메소드의 이름 (문자열)
      - dispatch_fn: 호출 전, 인자들을 변환하거나 전처리하는 함수
      - collect_fn: 원격 실행 후, 결과를 후처리하는 함수
      - execute_fn: 실제 원격 메소드를 호출하는 함수 (예: ray.remote 호출)
      - blocking: True인 경우, 원격 호출의 결과를 기다림 (동기식 실행)
    
    반환:
      - 내부에서 정의된 func 함수 (래퍼 함수)
    """
    def func(*args, **kwargs):
        # dispatch_fn을 통해 입력 인자들을 변환 또는 전처리
        args, kwargs = dispatch_fn(self, *args, **kwargs)
        # execute_fn을 호출하여 원격 메소드를 실행하고 결과를 받아옴
        output = execute_fn(method_name, *args, **kwargs)
        # blocking 옵션이 True이면 ray.get을 통해 결과를 동기적으로 받아옴
        if blocking:
            output = ray.get(output)
        # collect_fn을 통해 결과를 후처리하여 반환
        output = collect_fn(self, output)
        return output

    return func


# ---------------------------------------------------------------------------
# Ray 기반의 자원 풀(ResourcePool) 구현 클래스
# ---------------------------------------------------------------------------
class RayResourcePool(ResourcePool):
    """
    RayResourcePool은 Ray를 이용하여 분산 자원을 관리하는 클래스입니다.
    ResourcePool(부모 클래스)에서 정의된 노드별 프로세스 수(_store 등)를 바탕으로,
    각 노드에 할당할 자원(bundle)을 생성하고 placement group을 관리합니다.
    """

    def __init__(self,
                 process_on_nodes: List[int] = None,  # 각 노드별로 할당할 프로세스 수 리스트 (예: [2, 3, 1]이면 0번 노드에 2개, 1번 노드에 3개, 2번 노드에 1개)
                 use_gpu: bool = True,  # GPU 사용 여부 (True이면 GPU 할당)
                 name_prefix: str = "",  # placement group이나 actor 이름의 접두사로 사용
                 max_colocate_count: int = 5,  # 한 노드에 동시에 colocated(같은 자원 내에) 실행 가능한 프로세스 수
                 detached=False  # detached 모드인 경우, 생성된 actor의 lifetime이 detached로 설정됨
                 ) -> None:
        # 부모 클래스 ResourcePool의 초기화 (여기서 self._store 등 설정됨)
        super().__init__(process_on_nodes, max_colocate_count)
        self.use_gpu = use_gpu
        self.name_prefix = name_prefix  # 디버깅이나 관리용 이름 접두사
        self.pgs = None  # 생성된 placement group들을 저장할 변수 (초기에는 None)
        self.detached = detached

    def get_placement_groups(self, strategy="STRICT_PACK", name=None):
        """
        현재 resource pool에 해당하는 placement group들을 생성하여 반환합니다.
        
        매개변수:
          - strategy: placement group 생성 시 사용할 스케줄링 전략 (기본은 "STRICT_PACK")
          - name: placement group 이름 접두사 (지정하지 않으면 기본값 사용)
        
        동작:
          1. 이미 생성된 placement group(self.pgs)이 있으면 바로 반환.
          2. 그렇지 않으면, self._store(노드별 프로세스 수 리스트)에 따라 bundle 구성을 생성.
          3. 각 노드에 대해 bundle(자원 묶음)을 생성하고, placement group 객체를 생성.
          4. 모든 placement group이 준비될 때까지 대기.
          5. 생성된 placement group 리스트를 self.pgs에 저장 후 반환.
        """
        # 이미 placement group이 생성되어 있다면 재사용
        if self.pgs is not None:
            return self.pgs

        # placement group의 이름 접두사 설정:
        # name 매개변수가 주어지면 이를 사용, 아니면 self.name_prefix와 self._store의 값을 조합하여 생성
        pg_name_prefix = name if name else \
            f"{self.name_prefix}verl_group_{'_'.join([str(count) for count in self._store])}:"
        # self._store는 부모 ResourcePool에서 설정된 각 노드별 프로세스 수 리스트입니다.

        # pg_scheme: 각 노드별로 생성할 bundle(자원 묶음) 구성을 리스트 내포로 생성
        # 각 노드에서 process_count 만큼의 bundle을 생성하며, 각 bundle은 GPU 사용 여부에 따라 자원 사양이 달라짐
        pg_scheme = [[{
            "CPU": self.max_collocate_count,
            "GPU": 1
        } if self.use_gpu else {
            "CPU": self.max_collocate_count
        } for _ in range(process_count)] for process_count in self._store]

        # detached 모드이면 lifetime을 'detached'로 설정, 아니면 None
        lifetime = 'detached' if self.detached else None

        # 각 노드별로 placement group 생성: enumerate로 인덱스와 bundle 리스트를 가져와서 생성
        pgs = [
            placement_group(bundles=bundles, strategy=strategy, name=pg_name_prefix + str(idx), lifetime=lifetime)
            for idx, bundles in enumerate(pg_scheme)
        ]

        # 생성된 모든 placement group이 준비될 때까지 ray.get을 통해 동기적으로 대기
        ray.get([pg.ready() for pg in pgs])

        # 생성된 placement group들을 인스턴스 변수에 저장 후 반환
        self.pgs = pgs
        return pgs


# ---------------------------------------------------------------------------
# 기존 resource pool들에서 특정 역할(src_role_names)에 해당하는 placement group 추출 함수
# ---------------------------------------------------------------------------
def extract_pg_from_exist(resource_pools: Dict[str, RayResourcePool], src_role_names: List[str],
                          resource_pool: RayResourcePool) -> List:
    """
    여러 resource_pools(dict: role_name -> RayResourcePool) 중에서,
    src_role_names에 해당하는 placement group들을 추출하여, 
    resource_pool의 노드별 요청(process 수)에 맞게 정렬 후 반환합니다.
    
    동작:
      1. resource_pools 내에서 role_name이 src_role_names에 포함된 모든 placement group들을 수집.
      2. bundle_count (placement group 내 bundle 수)를 기준으로 내림차순 정렬.
      3. resource_pool.store (각 노드별 요청 프로세스 수, 인덱스 포함)를 내림차순 정렬.
      4. 각 노드 요청에 대해 충분한 bundle 수를 갖는 placement group을 할당.
      5. 원래의 노드 순서에 맞게 정렬하여 반환.
    """
    # src_role_names에 해당하는 placement group들을 추출
    src_pgs = [
        pg for role_name, resource_pool in resource_pools.items() for pg in resource_pool.get_placement_groups()
        if role_name in src_role_names
    ]

    # bundle 수 기준으로 내림차순 정렬
    sorted_src_pgs = sorted(src_pgs, key=lambda pg: pg.bundle_count, reverse=True)
    # resource_pool의 요청 프로세스 수와 인덱스를 내림차순 정렬 (예: (process_count, index))
    sorted_process_on_nodes = sorted([(val, idx) for idx, val in enumerate(resource_pool.store)], reverse=True)

    unsorted_pgs: List[Tuple[int, PlacementGroup]] = []
    searching_idx = 0
    # 각 노드의 요청(process 수)에 대해 placement group 할당
    for request_process, original_idx in sorted_process_on_nodes:
        # 할당 가능한 placement group이 충분히 있는지 확인
        assert searching_idx < len(sorted_src_pgs), f"no enough nodes for request: searching {searching_idx} th node"
        # 요청한 process 수가 placement group의 bundle 수를 초과하지 않아야 함
        assert request_process <= sorted_src_pgs[searching_idx].bundle_count, \
            f"requesting {request_process} processes, bundle count cannot satisfy"
        # (원래 인덱스, 할당된 placement group) 튜플을 저장
        unsorted_pgs.append((original_idx, sorted_src_pgs[searching_idx]))
        searching_idx += 1

    # 원래 노드 순서대로 정렬하여 placement group 리스트 반환
    return [pg for _, pg in sorted(unsorted_pgs)]


# ---------------------------------------------------------------------------
# 두 RayResourcePool을 병합하는 함수
# ---------------------------------------------------------------------------
def merge_resource_pool(rp1: RayResourcePool, rp2: RayResourcePool) -> RayResourcePool:
    """
    두 개의 RayResourcePool을 병합하여 하나의 새로운 resource pool을 생성합니다.
    병합 전에는 두 resource pool의 사용 조건(예: use_gpu, max_collocate_count, n_gpus_per_node, detached)이 동일해야 합니다.
    
    동작:
      1. 각종 조건(use_gpu, max_collocate_count, n_gpus_per_node, detached)이 일치하는지 assert로 확인.
      2. 두 resource pool의 store(각 노드별 프로세스 수 리스트)를 합침.
      3. 새로운 RayResourcePool 객체를 생성하고, 기존 placement group들을 합침.
      4. 병합된 resource pool 반환.
    """
    assert rp1.use_gpu == rp2.use_gpu, 'Both RayResourcePool must either use_gpu or not'
    assert rp1.max_collocate_count == rp2.max_collocate_count, 'Both RayResourcePool must has the same max_collocate_count'
    assert rp1.n_gpus_per_node == rp2.n_gpus_per_node, 'Both RayResourcePool must has the same n_gpus_per_node'
    assert rp1.detached == rp2.detached, 'Detached ResourcePool cannot be merged with non-detached ResourcePool'

    # 두 resource pool의 노드별 프로세스 수 리스트를 합침
    new_store = rp1.store + rp2.store

    # 새로운 resource pool 생성: 이름 접두사는 두 pool의 접두사를 합침
    merged = RayResourcePool(new_store, rp1.use_gpu, f"{rp1.name_prefix}_{rp2.name_prefix}")
    # 기존 placement group들을 합쳐서 할당
    merged.pgs = rp1.get_placement_groups() + rp2.get_placement_groups()

    return merged


# ---------------------------------------------------------------------------
# Ray에서 사용할 클래스 초기화 인자 및 옵션 관리 클래스
# ---------------------------------------------------------------------------
class RayClassWithInitArgs(ClassWithInitArgs):
    """
    RayClassWithInitArgs는 주어진 클래스(cls)를 초기화 인자(args, kwargs)와 함께 감싸서,
    Ray Actor 생성 시 추가 옵션(예: scheduling_strategy, runtime_env 등)을 동적으로 설정할 수 있게 합니다.
    """

    def __init__(self, cls, *args, **kwargs) -> None:
        # 부모 클래스(ClassWithInitArgs)의 초기화를 수행합니다.
        super().__init__(cls, *args, **kwargs)
        # 추가 옵션 및 자원 정보를 저장할 딕셔너리 초기화
        self._options = {}
        self._additional_resource = {}

    def set_additional_resource(self, additional_resource):
        """
        추가 자원 정보를 설정합니다.
        
        매개변수:
          - additional_resource: 추가적으로 필요한 자원 정보를 담은 딕셔너리
        """
        self._additional_resource = additional_resource

    def update_options(self, options: Dict):
        """
        기존 옵션 딕셔너리에 새로운 옵션들을 업데이트합니다.
        
        매개변수:
          - options: 업데이트할 옵션 딕셔너리
        """
        self._options.update(options)

    def __call__(self,
                 placement_group,
                 placement_group_bundle_idx,
                 use_gpu: bool = True,
                 num_gpus=1,
                 sharing_with=None) -> Any:
        """
        Ray Actor를 생성하기 위한 호출 메소드입니다.
        
        매개변수:
          - placement_group: 해당 Actor가 속할 placement group
          - placement_group_bundle_idx: placement group 내에서 사용할 bundle의 인덱스
          - use_gpu: GPU 사용 여부
          - num_gpus: 사용하려는 GPU 수 (use_gpu가 True일 때 적용)
          - sharing_with: 다른 Actor와 자원을 공유할 경우 해당 Actor를 지정 (NodeAffinitySchedulingStrategy 사용)
        
        동작:
          1. sharing_with가 주어지면, 해당 Actor의 노드 정보를 가져와 affinity 옵션을 설정.
          2. 그렇지 않으면 PlacementGroupSchedulingStrategy를 사용하여 scheduling 옵션 설정.
          3. self._options와 self._additional_resource의 옵션들을 병합.
          4. 최종 옵션을 적용하여 Ray Actor를 remote 방식으로 생성 후 반환.
        """
        if sharing_with is not None:
            # 다른 Actor와 자원을 공유하는 경우, 대상 Actor의 노드 정보와 CUDA 디바이스 정보를 가져옴
            target_node_id = ray.get(sharing_with.get_node_id.remote())
            cuda_visible_devices = ray.get(sharing_with.get_cuda_visible_devices.remote())
            options = {"scheduling_strategy": NodeAffinitySchedulingStrategy(node_id=target_node_id, soft=False)}
            return self.cls.options(**options).remote(*self.args,
                                                      cuda_visible_devices=cuda_visible_devices,
                                                      **self.kwargs)

        # 기본 scheduling 전략: PlacementGroupSchedulingStrategy를 사용하여 placement group과 bundle 인덱스 지정
        options = {
            "scheduling_strategy":
                PlacementGroupSchedulingStrategy(placement_group=placement_group,
                                                 placement_group_bundle_index=placement_group_bundle_idx)
        }
        # 미리 설정된 옵션들을 추가
        options.update(self._options)

        if use_gpu:
            options["num_gpus"] = num_gpus

        # 추가 자원 옵션이 2개 이상인 경우 모두 옵션에 추가
        if len(self._additional_resource) > 1:
            for k, v in self._additional_resource.items():
                options[k] = v

        # 최종 옵션을 적용하여 Ray Actor를 생성하고 remote 객체를 반환
        return self.cls.options(**options).remote(*self.args, **self.kwargs)


# ---------------------------------------------------------------------------
# Ray를 이용한 Worker Group 관리 클래스
# ---------------------------------------------------------------------------
class RayWorkerGroup(WorkerGroup):
    """
    RayWorkerGroup은 Ray를 이용하여 분산 환경에서 워커(Actor)들을 관리하는 클래스입니다.
    WorkerGroup(부모 클래스)에서 정의된 기능을 확장하여, resource pool과 placement group을 이용한
    워커 생성, 배포, 실행 및 동기/비동기 호출 기능을 제공합니다.
    """

    def __init__(self,
                 resource_pool: RayResourcePool = None,  # RayResourcePool 객체 (자원 할당 정보를 담고 있음)
                 ray_cls_with_init: RayClassWithInitArgs = None,  # Ray Actor 생성을 위한 클래스 감싸기 객체
                 bin_pack: bool = True,  # bin packing 전략 사용 여부 (자원 최적화를 위해)
                 name_prefix: str = None,  # 워커 이름 접두사 (없으면 랜덤 문자열 생성)
                 detached=False,  # detached 모드 여부
                 worker_names=None,  # 이미 생성된 detached worker들의 이름 목록 (선택 사항)
                 **kwargs) -> None:
        # 부모 WorkerGroup 초기화 (여기서 resource_pool 등 기본 속성 설정)
        super().__init__(resource_pool=resource_pool, **kwargs)
        self.ray_cls_with_init = ray_cls_with_init
        # name_prefix가 없으면 get_random_string 함수를 이용하여 랜덤 문자열 생성
        self.name_prefix = get_random_string(length=6) if name_prefix is None else name_prefix

        if worker_names is not None:
            # detached worker로 초기화할 때, worker_names가 주어지면 해당 리스트를 사용
            # _is_init_with_detached_workers는 WorkerGroup에서 정의된 속성으로, detached 모드 여부를 나타냄
            assert self._is_init_with_detached_workers
            self._worker_names = worker_names

        if self._is_init_with_detached_workers:
            # detached 모드인 경우, 기존에 생성된 detached worker들을 이용하여 초기화
            self._init_with_detached_workers(worker_names=worker_names)
        else:
            # resource_pool 정보를 이용하여 새롭게 워커들을 생성하여 초기화
            self._init_with_resource_pool(resource_pool=resource_pool,
                                          ray_cls_with_init=ray_cls_with_init,
                                          bin_pack=bin_pack,
                                          detached=detached)

        if ray_cls_with_init is not None:
            # 워커들의 메소드를 동적으로 바인딩하기 위한 함수(func_generator)를 사용하여 메소드 연결
            self._bind_worker_method(self.ray_cls_with_init.cls, func_generator)

    def _is_worker_alive(self, worker: ray.actor.ActorHandle):
        """
        주어진 worker(Actor)가 현재 ALIVE 상태인지 확인하는 함수입니다.
        actor의 상태 정보를 가져와서 "ALIVE"인지 여부를 판단합니다.
        """
        worker_state_dict = get_actor(worker._actor_id.hex())
        return worker_state_dict.get("state", "undefined") == "ALIVE" if worker_state_dict is not None else False

    def _init_with_detached_workers(self, worker_names):
        """
        detached 모드로 이미 생성된 worker들의 이름 목록을 이용하여 워커들을 초기화합니다.
        ray.get_actor(name)을 이용하여 각 worker의 ActorHandle을 가져와 self._workers에 저장합니다.
        """
        workers = [ray.get_actor(name=name) for name in worker_names]
        self._workers = workers
        self._world_size = len(worker_names)

    def _init_with_resource_pool(self, resource_pool, ray_cls_with_init, bin_pack, detached):
        """
        resource_pool 정보를 이용하여 새롭게 worker들을 생성하는 초기화 함수입니다.
        
        동작:
          1. resource_pool으로부터 placement group들을 얻어옴 (bin_pack 여부에 따라 scheduling 전략 결정).
          2. 각 노드별로 할당된 process 수(self._store의 값)를 기준으로 반복문을 수행.
          3. 각 process에 대해 환경변수(예: WORLD_SIZE, RANK, MASTER 정보 등)를 설정.
          4. ray_cls_with_init 객체의 옵션을 업데이트하고, placement group의 bundle 인덱스를 지정하여 worker 생성.
          5. 첫 번째(worker rank 0) worker의 경우, master 주소와 포트를 register_center_actor에서 획득.
        """
        use_gpu = resource_pool.use_gpu

        # bin_pack이 True이면 "STRICT_PACK", 아니면 "PACK" 전략 사용
        strategy = "PACK"
        if bin_pack:
            strategy = "STRICT_PACK"
        # resource_pool에서 placement group들을 가져옴
        pgs = resource_pool.get_placement_groups(strategy=strategy)
        world_size = resource_pool.world_size  # 전체 워커 수 (노드별 process 수의 합)
        self._world_size = world_size
        # 각 워커가 사용할 GPU 수: 한 노드의 최대 동시 실행 개수에 따른 분할 (예: 1/5)
        num_gpus = 1 / resource_pool.max_collocate_count

        rank = -1  # 전역 rank 초기값
        # resource_pool.store는 각 노드에 할당된 프로세스 수 리스트
        for pg_idx, local_world_size in enumerate(resource_pool.store):
            pg = pgs[pg_idx]
            # 현재 노드의 process 수가 placement group의 bundle 수를 초과하면 안됨
            assert local_world_size <= pg.bundle_count, \
                f"when generating for {self.name_prefix}, for the "
            for local_rank in range(local_world_size):
                rank += 1

                # 각 worker에 전달할 환경변수 설정:
                # WORLD_SIZE: 전체 워커 수, RANK: worker의 전역 순번,
                # WG_PREFIX: 워커 그룹 이름 접두사, WG_BACKEND: 사용 중인 분산 백엔드 (여기서는 'ray'),
                # RAY_LOCAL_WORLD_SIZE: 현재 노드의 worker 수, RAY_LOCAL_RANK: 노드 내 worker 순번
                env_vars = {
                    'WORLD_SIZE': str(world_size),
                    'RANK': str(rank),
                    'WG_PREFIX': self.name_prefix,
                    'WG_BACKEND': 'ray',
                    'RAY_LOCAL_WORLD_SIZE': str(local_world_size),
                    'RAY_LOCAL_RANK': str(local_rank),
                }
                if rank != 0:
                    # rank 0가 아닌 경우, master의 주소와 포트를 환경변수에 추가
                    env_vars['MASTER_ADDR'] = self._master_addr
                    env_vars['MASTER_PORT'] = self._master_port

                import re
                # ray_cls_with_init.cls의 이름에서 실제 클래스 이름을 추출 (예: "ActorClass(Worker)"에서 "Worker" 추출)
                cia_name = type(ray_cls_with_init.cls).__name__
                match = re.search(r"ActorClass\(([^)]+)\)", cia_name)
                cia_name = match.group(1) if match else cia_name
                # worker의 고유 이름 생성: 예) WG_PREFIX + 클래스명 + 노드 인덱스 + ':' + 로컬 rank
                name = f"{self.name_prefix}{cia_name}_{pg_idx}:{local_rank}"

                # ray_cls_with_init의 옵션을 업데이트:
                # runtime_env에 env_vars를 포함하고, 이름 옵션(name)도 설정
                ray_cls_with_init.update_options({'runtime_env': {'env_vars': env_vars}, 'name': name})

                if detached:
                    # detached 모드이면 lifetime 옵션을 'detached'로 설정
                    ray_cls_with_init.update_options({'lifetime': 'detached'})

                # 실제로 worker(Actor)를 생성: placement group, bundle 인덱스, GPU 사용 정보 등을 전달
                worker = ray_cls_with_init(placement_group=pg,
                                           placement_group_bundle_idx=local_rank,
                                           use_gpu=use_gpu,
                                           num_gpus=num_gpus)
                # 생성된 worker와 이름을 각각 저장
                self._workers.append(worker)
                self._worker_names.append(name)

                if rank == 0:
                    # 첫 번째 worker(rank 0)는 register_center_actor를 통해 master의 주소와 포트를 획득
                    register_center_actor = None
                    # 최대 120초 동안 register_center_actor가 생성될 때까지 대기
                    for _ in range(120):
                        if f"{self.name_prefix}_register_center" not in list_named_actors():
                            time.sleep(1)
                        else:
                            register_center_actor = ray.get_actor(f"{self.name_prefix}_register_center")
                            break
                    # register_center_actor를 얻지 못하면 에러 발생
                    assert register_center_actor is not None, f"failed to get register_center_actor: {self.name_prefix}_register_center in {list_named_actors(all_namespaces=True)}"
                    # rank 0 worker에서 master 정보 획득
                    rank_zero_info = ray.get(register_center_actor.get_rank_zero_info.remote())
                    self._master_addr, self._master_port = rank_zero_info['MASTER_ADDR'], rank_zero_info['MASTER_PORT']
                    # master 정보는 이후 다른 worker들에게도 전달됨

    @property
    def worker_names(self):
        """
        현재 워커 그룹에 속한 모든 worker들의 이름 목록을 반환합니다.
        """
        return self._worker_names

    @classmethod
    def from_detached(cls, worker_names=None, ray_cls_with_init=None):
        """
        detached 모드로 초기화된 워커 그룹을 생성하는 클래스 메소드입니다.
        
        매개변수:
          - worker_names: 이미 생성된 detached worker들의 이름 목록
          - ray_cls_with_init: Ray Actor 생성을 위한 클래스 감싸기 객체
          
        반환:
          - detached 모드로 초기화된 RayWorkerGroup 객체
        """
        worker_group = cls(resource_pool=None,
                           ray_cls_with_init=ray_cls_with_init,
                           name_prefix=None,
                           worker_names=worker_names)
        return worker_group

    def spawn(self, prefix_set):
        """
        주어진 prefix_set에 따라 여러 워커 그룹을 생성하여, 각 그룹이 특정 prefix가 붙은 메소드를 갖도록 합니다.
        
        동작:
          1. 내부 함수 _rebind_actor_methods를 사용하여, 각 actor의 메소드 이름에서 prefix를 제거하고 원래 이름으로 바인딩.
          2. prefix_set에 있는 각 prefix에 대해, 새로운 detached 워커 그룹을 생성.
          3. 생성된 워커 그룹들을 딕셔너리 형태로 반환 (key: prefix, value: 해당 워커 그룹).
        """
        def _rebind_actor_methods(worker_group, actor_name):
            """
            주어진 actor_name(prefix)를 가진 메소드들을 원래 이름으로 재바인딩하는 함수입니다.
            
            예: "worker_method"가 "prefix_worker_method"로 바인딩되어 있다면,
                prefix를 제거하여 "worker_method"로 다시 설정.
            """
            prefix: str = actor_name + '_'
            for method_name in dir(worker_group):
                if method_name.startswith(prefix):
                    # Python 3.9 이상에서 removeprefix를 사용하여 접두사를 제거
                    original_method_name = method_name.removeprefix(prefix)
                    method = getattr(worker_group, method_name)
                    setattr(worker_group, original_method_name, method)

        new_worker_group_dict = {}
        # prefix_set에 포함된 각 prefix에 대해 새로운 워커 그룹 생성
        for prefix in prefix_set:
            new_worker_group = self.from_detached(worker_names=self._worker_names,
                                                  ray_cls_with_init=self.ray_cls_with_init)
            # 생성된 워커 그룹에 대해 메소드 이름 재바인딩 수행
            _rebind_actor_methods(new_worker_group, prefix)
            new_worker_group_dict[prefix] = new_worker_group
        return new_worker_group_dict

    def execute_rank_zero_sync(self, method_name: str, *args, **kwargs):
        """
        rank 0 (첫 번째) 워커의 메소드를 동기식으로 실행하고, 결과를 반환합니다.
        내부적으로 execute_all_async를 호출한 후, ray.get을 통해 결과를 기다립니다.
        """
        return ray.get(self.execute_all_async(method_name, **args, **kwargs))

    def execute_rank_zero_async(self, method_name: str, *args, **kwargs):
        """
        rank 0 (첫 번째) 워커의 메소드를 비동기식으로 실행합니다.
        반환값은 원격 호출의 결과로, ray.get을 호출하여 결과를 얻을 수 있습니다.
        """
        remote_call = getattr(self._workers[0], method_name)
        return remote_call.remote(*args, **kwargs)

    def execute_rank_zero(self, method_name: str, *args, **kwargs):
        """
        rank 0 워커의 메소드를 실행합니다.
        기본적으로 execute_rank_zero_async와 동일하게 동작합니다.
        """
        return self.execute_rank_zero_async(method_name, *args, **kwargs)

    def execute_all(self, method_name: str, *args, **kwargs):
        """
        모든 워커의 메소드를 실행하는 함수입니다.
        내부적으로 execute_all_async를 호출합니다.
        """
        return self.execute_all_async(method_name, *args, **kwargs)

    def execute_all_sync(self, method_name: str, *args, **kwargs):
        """
        모든 워커의 메소드를 동기식으로 실행하여 결과를 반환합니다.
        """
        return ray.get(self.execute_all_async(method_name, *args, **kwargs))

    def execute_all_async(self, method_name: str, *args, **kwargs):
        """
        모든 워커의 메소드를 비동기식으로 실행합니다.
        
        동작:
          1. 모든 인자들이 리스트이고, 각 리스트의 길이가 전체 워커 수와 일치하면,
             각 워커에 대해 해당 인자들만 분할하여 전달 (샤딩)합니다.
          2. 그렇지 않으면, 모든 워커에 동일한 인자들을 전달하여 메소드를 호출합니다.
          
        반환:
          - 각 워커에 대한 원격 호출 결과의 리스트
        """
        length = len(self._workers)
        # 모든 positional 인자와 keyword 인자가 리스트인지 확인
        if all(isinstance(arg, list) for arg in args) and all(isinstance(kwarg, list) for kwarg in kwargs.values()):
            # 각 리스트의 길이가 워커 수와 동일한지 확인
            if all(len(arg) == length for arg in args) and all(len(kwarg) == length for kwarg in kwargs.values()):
                # 각 워커에 대해 인자들을 분할하여 전달
                result = []
                for i in range(length):
                    sliced_args = tuple(arg[i] for arg in args)
                    sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
                    remote_call = getattr(self._workers[i], method_name)
                    result.append(remote_call.remote(*sliced_args, **sliced_kwargs))
                return result

        # 기본적으로 모든 워커에 동일한 인자들을 전달하여 메소드 호출
        return [getattr(worker, method_name).remote(*args, **kwargs) for worker in self._workers]

    @property
    def master_address(self):
        """
        master (rank 0 워커)의 주소를 반환합니다.
        """
        return self._master_addr

    @property
    def master_port(self):
        """
        master (rank 0 워커)의 포트 번호를 반환합니다.
        """
        return self._master_port

    @property
    def workers(self):
        """
        현재 워커 그룹에 속한 모든 워커(Actor)의 리스트를 반환합니다.
        """
        return self._workers

    @property
    def world_size(self):
        """
        전체 워커 수(전역 world size)를 반환합니다.
        """
        return self._world_size


# ---------------------------------------------------------------------------
# 여러 ray.Actor 내에 독립적인 워커를 생성할 수 있도록 하는 유틸리티 함수들
# ---------------------------------------------------------------------------
from unittest.mock import patch  # 특정 상황에서 환경 변수를 임시로 변경하기 위한 모듈
from verl.single_controller.base.decorator import MAGIC_ATTR
# MAGIC_ATTR: 데코레이터를 통해 특정 메소드에 부여된 특별한 속성 (public 메소드임을 나타냄)
import os  # 환경 변수 접근을 위한 모듈


def _bind_workers_method_to_parent(cls, key, user_defined_cls):
    """
    주어진 user_defined_cls(사용자 정의 클래스)의 public 메소드들(MAGIC_ATTR이 적용된 메소드)을,
    WorkerDict(부모 클래스)에 바인딩합니다.
    
    매개변수:
      - cls: 메소드를 바인딩할 대상 클래스 (WorkerDict)
      - key: worker_dict 내에서 해당 worker를 식별하기 위한 key (문자열)
      - user_defined_cls: 실제로 메소드가 정의된 클래스
    """
    for method_name in dir(user_defined_cls):
        try:
            method = getattr(user_defined_cls, method_name)
            # 해당 속성이 callable(함수)인지 확인, 아니라면 에러 발생
            assert callable(method), f"{method_name} in {user_defined_cls} is not callable"
        except Exception as e:
            # 만약 프로퍼티 등 callable하지 않은 경우, 그냥 패스
            continue

        # MAGIC_ATTR 속성이 있는 경우에만 바인딩 진행 (즉, 데코레이터로 등록된 public 메소드)
        if hasattr(method, MAGIC_ATTR):

            def generate_function(name):
                """
                주어진 메소드 이름(name)을 사용하여, WorkerDict 내부에서 실제 worker의 메소드로 디스패치하는 함수를 생성합니다.
                """
                def func(self, *args, **kwargs):
                    # self.worker_dict[key]에 해당하는 worker의 메소드를 호출하여 결과 반환
                    return getattr(self.worker_dict[key], name)(*args, **kwargs)
                return func

            func = generate_function(method_name)
            # 생성된 함수에도 MAGIC_ATTR 속성을 부여하여, 외부에서 이를 인식할 수 있도록 함
            setattr(func, MAGIC_ATTR, getattr(method, MAGIC_ATTR))
            try:
                # key와 언더바(_)를 조합하여 메소드 이름을 생성 (예: "key_methodName")
                method_name_with_prefix = key + '_' + method_name
                setattr(cls, method_name_with_prefix, func)
                # 바인딩 성공 시, 디버깅 목적으로 메시지 출력 가능 (주석 처리됨)
                # print(f'Binding {method_name_with_prefix}')
            except Exception as e:
                raise ValueError(f'Fail to set method_name {method_name}')


def _unwrap_ray_remote(cls):
    """
    만약 주어진 클래스가 Ray의 remote 래퍼 클래스(__ray_actor_class__ 속성을 가짐)라면,
    실제 원래의 클래스를 반환합니다.
    """
    if hasattr(cls, '__ray_actor_class__'):
        cls = cls.__ray_actor_class__
    return cls


def create_colocated_worker_cls(class_dict: dict[str, RayClassWithInitArgs]):
    """
    동일 프로세스 내에서 여러 워커를 동시에 생성할 수 있도록 하는 클래스를 동적으로 생성합니다.
    
    매개변수:
      - class_dict: key가 문자열, value가 RayClassWithInitArgs 객체인 딕셔너리
        각 key는 worker를 식별하기 위한 이름이며, value는 해당 worker 클래스와 초기화 인자를 포함.
    
    동작:
      1. class_dict를 순회하면서, 각 worker의 실제 클래스와 초기화 인자들을 추출하여
         cls_dict와 init_args_dict에 저장.
      2. 모든 worker가 동일한 기본 클래스를 상속받는지 확인.
      3. WorkerDict라는 새로운 클래스를 정의하여, 내부에 worker_dict라는 딕셔너리를 생성.
         각 key에 대해 해당 worker 인스턴스를 생성 (환경 변수 'DISABLE_WORKER_INIT'을 설정하여 초기화 방지).
      4. _bind_workers_method_to_parent를 사용하여, 각 내부 worker 클래스의 public 메소드를
         WorkerDict에 바인딩 (메소드 이름에 key 접두사가 붙음).
      5. WorkerDict를 ray.remote로 래핑한 후, RayClassWithInitArgs로 감싸서 반환.
    """
    cls_dict = {}
    init_args_dict = {}
    worker_cls = None
    for key, cls in class_dict.items():
        if worker_cls is None:
            # 첫 번째 worker의 base class를 기준으로 설정
            worker_cls = cls.cls.__ray_actor_class__.__base__
        else:
            # 이후 worker들도 동일한 base class를 사용하는지 검증
            assert worker_cls == cls.cls.__ray_actor_class__.__base__, \
                'the worker class should be the same when share the same process'
        cls_dict[key] = cls.cls
        init_args_dict[key] = {'args': cls.args, 'kwargs': cls.kwargs}

    # 두 딕셔너리의 키들이 동일한지 확인
    assert cls_dict.keys() == init_args_dict.keys()

    # WorkerDict 클래스 정의 (동적으로 생성)
    class WorkerDict(worker_cls):
        """
        WorkerDict는 동일 프로세스 내에 여러 worker 인스턴스를 생성하고,
        각 worker의 메소드를 통합하여 호출할 수 있도록 합니다.
        """
        def __init__(self):
            # 부모 클래스 초기화
            super().__init__()
            self.worker_dict = {}
            # 각 key에 대해, 환경 변수 'DISABLE_WORKER_INIT'를 설정하여 worker 인스턴스 생성
            for key, user_defined_cls in cls_dict.items():
                user_defined_cls = _unwrap_ray_remote(user_defined_cls)
                with patch.dict(os.environ, {'DISABLE_WORKER_INIT': '1'}):
                    self.worker_dict[key] = user_defined_cls(*init_args_dict[key].get('args', ()),
                                                             **init_args_dict[key].get('kwargs', {}))

    # 각 user_defined_cls의 public 메소드들을 WorkerDict에 바인딩
    for key, user_defined_cls in cls_dict.items():
        user_defined_cls = _unwrap_ray_remote(user_defined_cls)
        _bind_workers_method_to_parent(WorkerDict, key, user_defined_cls)

    # WorkerDict를 Ray Actor로 생성할 수 있도록 remote로 래핑 후, RayClassWithInitArgs로 감싸서 반환
    remote_cls = ray.remote(WorkerDict)
    remote_cls = RayClassWithInitArgs(cls=remote_cls)
    return remote_cls
