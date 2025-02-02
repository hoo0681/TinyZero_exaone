export N_GPUS=1
export CUDA_VISIBLE_DEVICES=0
# ray stop --force && ray start --head --include-dashboard=True
export BASE_MODEL=Qwen/Qwen2.5-0.5B
export DATA_DIR="dataset"
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=countdown-qwen2.5-0.5b-grpo-lora
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero_a6000_grpo.sh