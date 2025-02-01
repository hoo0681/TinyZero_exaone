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
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math, multiply, countdown
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _select_rm_score_fn(data_source):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score # gsm8k 데이터셋에 대한 보상 점수 계산 함수
    elif data_source == 'lighteval/MATH':
        return math.compute_score # math 데이터셋에 대한 보상 점수 계산 함수
    elif "multiply" in data_source or "arithmetic" in data_source:
        return multiply.compute_score # multiply 데이터셋에 대한 보상 점수 계산 함수    
    elif "countdown" in data_source:
        return countdown.compute_score # countdown 데이터셋에 대한 보상 점수 계산 함수
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys(): 
            return data.batch['rm_scores'] # 이미 계산된 보상 점수 있으면 반환

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts'] # 프롬프트 토큰 아이디

            prompt_length = prompt_ids.shape[-1] # 프롬프트 토큰 아이디 길이

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] # 응답 토큰 아이디
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum() # 응답 토큰 아이디 길이
            valid_response_ids = response_ids[:valid_response_length] # 필요한 만큼만 가져오기

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids)) # 프롬프트와 응답 토큰 아이디 연결
            sequences_str = self.tokenizer.decode(sequences) # 토큰 아이디를 디코딩하여 문자열로 변환

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth'] # 정답

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source'] # 데이터 소스
            compute_score_fn = _select_rm_score_fn(data_source) # 데이터 소스에 따라 보상 점수 계산 함수 선택

            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth) # 보상 점수 계산
            reward_tensor[i, valid_response_length - 1] = score # 보상 점수 저장

            if data_source not in already_print_data_sources: # 이미 출력한 데이터 소스인지 확인
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine: # 출력할 데이터 소스 개수 확인
                already_print_data_sources[data_source] += 1
                print(sequences_str) # 출력 

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),# 비동기로 호출할 수 있게 함
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes, # GPU개수 * 노드 개수
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable: # 활성화 되지 않는게 기본값
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0) # 보상 점수 관리 정책 준비

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1) # 보상 점수 관리 정책 준비 중간에 1개만 출력

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping) # 리소스 풀 관리자 준비 => 역할과 리소스를 리소스 이름을 키로 하여 매핑; 역할을 주면 리소스

    trainer = RayPPOTrainer(config=config, # 설정
                            tokenizer=tokenizer, # 토크나이저
                            role_worker_mapping=role_worker_mapping, # 역할과 비동기로 준비된 작업자 dict
                            resource_pool_manager=resource_pool_manager, # 역할을 주면 리소스를 반환하는 객체
                            ray_worker_group_cls=ray_worker_group_cls, # 레이 작업자 그룹 클래스
                            reward_fn=reward_fn, # 보상 함수
                            val_reward_fn=val_reward_fn) # 검증 보상 함수
    trainer.init_workers() # 작업자 초기화
    trainer.fit() # 트레이닝 시작


if __name__ == '__main__':
    main()
