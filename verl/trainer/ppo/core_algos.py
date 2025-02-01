# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
PPO (Proximal Policy Optimization) 알고리즘을 구현하기 위한 핵심 함수들을 모아놓은 파일입니다.
이 함수들은 다양한 분산 전략을 사용하는 트레이너에서 PPO 알고리즘을 구현할 때 사용됩니다.
"""

import numpy as np  # 수치 계산 라이브러리
import torch  # 딥러닝 프레임워크 (텐서 연산 등)
from collections import defaultdict  # 기본값이 있는 딕셔너리를 생성할 때 사용

# PPO 관련 함수(손실 계산, 클리핑 등)를 포함한 모듈을 verl_F라는 이름으로 가져옴
import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL Controller는 정책 업데이트 중 KL divergence를 동적으로 조절하기 위한 클래스입니다.
    논문(https://arxiv.org/pdf/1909.08593.pdf)에 소개된 방법을 사용합니다.
    self.horizon은 학습 과정에서 얼마나 오랫동안 KLD의 오차를 누적하여 반영할지 결정하는 값입니다.
    논문 https://ar5iv.labs.arxiv.org/html/1909.08593에서는 이러한 adaptive KL 컨트롤러가 
    정책이 원래의 사전 학습 모델(reference model)에서 크게 벗어나지 않도록 KL divergence를 조절하는 데 사용되며,
    horizon은 정책 업데이트가 안정적으로 이루어질 수 있는 시간 척도로서 반드시 0보다 큰 값이어야 한다고 설명하고 있습니다.
    즉, horizon은 KL 계수를 너무 급격하게 또는 너무 느리게 조정하지 않고 적절한 속도로 업데이트하기 위한 필수적인 요소입니다.
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        # 초기 KL 계수, 목표 KL 값, 조정에 사용할 horizon 값을 저장합니다.
        self.value = init_kl_coef  # 현재 KL 제약 계수
        self.target = target_kl      # 목표 KL divergence
        self.horizon = horizon       # 조정에 사용하는 스텝 수(또는 시간)

    def update(self, current_kl, n_steps):
        """
        현재 KL divergence와 진행된 스텝 수(n_steps)를 기반으로 KL 계수를 업데이트합니다.
        계산 과정에서 비율 오차를 일정 범위(-0.2 ~ 0.2)로 제한하여 급격한 변화를 방지합니다.
        """
        target = self.target
        # 현재 KL와 목표 KL의 비율에서 1을 뺀 값을 -0.2와 0.2 사이로 클리핑
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        # 업데이트 배수(mult)는 현재 스텝 수와 horizon에 비례하여 조정됨
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult  # KL 계수를 업데이트


class FixedKLController:
    """KL 계수를 고정하여 사용하는 간단한 컨트롤러입니다."""

    def __init__(self, kl_coef):
        # 초기 KL 계수를 설정합니다.
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        # 고정 컨트롤러는 업데이트하지 않으므로 아무런 동작도 하지 않습니다.
        pass


def get_kl_controller(config):
    """
    설정(config)에 따라 적절한 KL Controller(Fixed 또는 Adaptive)를 생성하여 반환합니다.
    
    config.critic.kl_ctrl.type 값에 따라 고정형 또는 적응형 컨트롤러를 선택합니다.
    """
    if config.critic.kl_ctrl.type == 'fixed':
        # 고정형 컨트롤러를 생성
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        # adaptive 타입일 경우 horizon 값이 0보다 커야 함을 확인
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        # 적응형 컨트롤러를 생성
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """
    Generalized Advantage Estimation (GAE)을 사용하여 advantage와 return을 계산합니다.
    
    Args:
        token_level_rewards: 각 토큰(또는 시간 단계)마다의 보상 (크기: [배치크기, 응답길이])
        values: 모델이 예측한 가치들 (크기: [배치크기, 응답길이])
        eos_mask: [EOS] 토큰을 구분하기 위한 마스크 (EOS 이후는 무시)
        gamma: 할인율 (미래 보상의 현재 가치 반영)
        lam: GAE 계산 시 사용하는 lambda 값 (bias-variance trade-off 조절)
    
    Returns:
        advantages: 계산된 advantage (크기: [배치크기, 응답길이])
        returns: advantage에 value를 더한 값 (즉, 실제 return)
    """
    with torch.no_grad():  # 기울기 계산 없이 수행 (역전파에 영향을 주지 않음)
        lastgaelam = 0  # 마지막 단계의 advantage 초기값
        advantages_reversed = []  # 역순으로 advantage를 저장할 리스트
        gen_len = token_level_rewards.shape[-1]  # 응답 길이

        # 마지막 토큰부터 첫 토큰까지 역순으로 순회
        for t in reversed(range(gen_len)):
            # 다음 단계의 value; 마지막 단계일 경우 0.0으로 설정
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            # 현재 보상, 할인된 다음 단계 가치, 현재 value를 사용해 델타(오차)를 계산
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            # GAE 계산: 현재 델타와 이전 단계의 advantage를 할인하여 더함
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        # 역순으로 저장된 advantage를 원래 순서로 복원하여 텐서로 만듦
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        # Return은 advantage와 예측 value를 더한 값
        returns = advantages + values
        # EOS 마스크를 사용하여 필요없는 부분은 제거한 후 advantage를 정규화(whiten)
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    GRPO (Generalized Reward Policy Optimization) 방식에서, Outcome 보상(각 응답당 하나의 스칼라 보상)
    을 사용하여 advantage를 계산합니다.
    
    Args:
        token_level_rewards: 각 토큰별 보상 (크기: [배치크기, 응답길이])
        eos_mask: EOS 토큰을 구분하기 위한 마스크 (크기: [배치크기, 응답길이])
        index: 각 응답에 대한 인덱스 (같은 프롬프트끼리 그룹화하기 위함)
        epsilon: 분모가 0이 되는 것을 방지하기 위한 아주 작은 값
    
    Returns:
        scores: 표준화된 보상 값 (advantage로 사용됨, 크기: [배치크기, 응답길이])
        scores: 동일한 값 반환 (때에 따라 리턴 형태가 달라질 수 있음)
    """
    response_length = token_level_rewards.shape[-1]  # 응답의 길이
    # 보상이 0이 아닌 위치를 찾기 위한 마스크 (0이 아니면 True)
    non_zero_mask = (token_level_rewards != 0)
    # 각 응답의 총 보상 계산 (마스크를 적용하여 0이 아닌 값만 합산)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    # 같은 index끼리 보상을 모으기 위해 defaultdict 사용
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]  # 배치 크기
        for i in range(bsz):
            # index에 따라 보상 값을 리스트에 추가
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            # 그룹 내에 보상이 하나만 있으면 평균 0, 표준편차 1로 설정
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            # 그룹 내에 보상이 여러 개 있으면 평균과 표준편차 계산
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            # 각 응답의 보상을 해당 그룹의 평균과 표준편차로 표준화
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        # 각 보상 값을 응답 길이에 맞게 확장(반복)하고 EOS 마스크를 곱함
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    # 여기서는 advantage와 return 모두 scores로 사용됨
    return scores, scores


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """
    토큰 단위의 스코어에서 KL penalty를 빼서 최종 보상을 계산합니다.
    
    Args:
        token_level_scores: 모델이 계산한 토큰별 스코어
        old_log_prob: 이전 정책의 로그 확률
        ref_log_prob: 기준(Reference) 정책의 로그 확률
        kl_ratio: KL penalty에 곱할 비율
    
    Returns:
        최종 보상값 (토큰 단위)
    """
    # KL divergence: 이전 로그 확률과 기준 로그 확률의 차이
    kl = old_log_prob - ref_log_prob
    # 보상에서 KL penalty를 빼줌
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """
    PPO 알고리즘에서 정책(Policy) 업데이트 시 사용되는 손실 함수(Policy Loss)를 계산합니다.
    
    Args:
        old_log_prob: 이전 정책의 로그 확률 (크기: [배치크기, 응답길이])
        log_prob: 현재 정책의 로그 확률 (크기: [배치크기, 응답길이])
        advantages: 계산된 advantage 값 (크기: [배치크기, 응답길이])
        eos_mask: EOS 토큰을 구분하기 위한 마스크 (크기: [배치크기, 응답길이])
        cliprange: PPO 클리핑 범위 (예: 0.2)
    
    Returns:
        pg_loss: 최종 정책 손실 (스칼라 텐서)
        pg_clipfrac: 클리핑이 발생한 비율 (실수)
        ppo_kl: KL divergence의 평균 (마스크 적용)
    """
    # 이전과 현재 로그 확률의 차이 (음수 KL의 근사값)
    negative_approx_kl = log_prob - old_log_prob
    # 정책 업데이트 비율 (비율이 1에서 크게 벗어나지 않도록)
    ratio = torch.exp(negative_approx_kl)
    # 마스크를 적용한 평균 KL divergence 계산
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    # 클리핑 전 손실: advantage에 비율을 곱함
    pg_losses = -advantages * ratio
    # 클리핑 후 손실: ratio를 cliprange 범위 내로 제한 후 계산
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    # 최종 정책 손실은 두 손실 중 더 큰 값을 선택해 평균내어 계산
    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    # 클리핑된 비율을 계산 (손실이 클리핑된 경우의 비율)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """
    모델의 출력(logits)을 기반으로 엔트로피를 계산하여,
    정책이 너무 결정적으로 변하지 않도록(탐색을 유지) 보조 손실로 사용합니다.
    
    Args:
        logits: 모델의 출력 (크기: [배치크기, 응답길이, 단어사전크기])
        eos_mask: EOS 토큰을 구분하기 위한 마스크 (크기: [배치크기, 응답길이])
    
    Returns:
        엔트로피 손실 (스칼라 텐서)
    """
    # logits로부터 엔트로피 계산 (배치별, 응답 길이별 엔트로피)
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    # EOS 마스크를 적용하여 평균 엔트로피 계산
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """
    가치 함수(value function) 손실을 계산합니다.
    예측한 가치와 실제 return 사이의 차이를 클리핑하여 안정적인 학습을 도모합니다.
    
    Args:
        vpreds: 현재 모델이 예측한 가치 (크기: [배치크기, 응답길이])
        returns: 실제 return 값 (크기: [배치크기, 응답길이])
        values: 이전에 저장한 가치 (크기: [배치크기, 응답길이])
        eos_mask: EOS 토큰 마스크 (크기: [배치크기, 응답길이])
        cliprange_value: 클리핑 범위 값
    
    Returns:
        vf_loss: 최종 가치 손실 (스칼라 텐서)
        vf_clipfrac: 클리핑이 발생한 비율 (실수)
    """
    # 예측값을 값의 범위 내로 클리핑
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    # 원래 예측과 실제 return의 제곱 오차
    vf_losses1 = (vpreds - returns)**2
    # 클리핑된 예측값과 실제 return의 제곱 오차
    vf_losses2 = (vpredclipped - returns)**2
    # 최종 가치 손실은 두 오차 중 큰 값을 선택하여 평균내고 0.5를 곱함
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    # 클리핑된 비율 계산
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """
    로그 확률(logprob)과 기준 로그 확률(ref_logprob)을 사용하여 KL divergence에 대한 패널티(벌점)를 계산합니다.
    
    kl_penalty 인자에 따라 다음 방식 중 하나를 선택하여 계산합니다:
      - "kl": 단순 로그 확률 차이
      - "abs": 차이의 절대값
      - "mse": 차이의 제곱의 0.5배 (평균제곱오차)
      - "low_var_kl": 변동성이 낮은 KL divergence로 근사 (결과값을 클리핑하여 사용)
      - "full": (현재 구현되지 않음)
    
    만약 정의되지 않은 방식이 전달되면 에러를 발생시킵니다.
    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # 변동성이 낮은 KL divergence 근사 계산
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # 모든 토큰에 대한 logits를 포함한 경우 처리 (현재 구현되지 않음)
        raise NotImplementedError

    raise NotImplementedError
