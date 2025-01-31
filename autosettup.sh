#####################################################
#conda 설치되어 있는지 확인
if ! command -v conda &> /dev/null
then
    echo "conda가 설치되어 있지 않습니다. 설치를 진행합니다."
    return 1
    # conda 설치 명령어 (예시)
    # wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    # bash Miniconda3-latest-Linux-x86_64.sh
fi

# conda 환경 생성 (이름은 zero) python 3.9 버전
if ! conda info --envs | grep -q "zero"
then
    echo "conda 환경 'zero'를 생성합니다."
    conda create -n zero python=3.9 -y
fi

# conda 환경 활성화
eval "$(conda shell.bash hook)"
conda activate zero
# source activate zero

# nvcc 설치되어 있는지 확인
if ! command -v nvcc &> /dev/null
then
    echo "nvcc가 설치되어 있지 않습니다. cuda를 conda로 설치합니다."
    conda install nvidia/label/cuda-12.1.1::cuda -y
fi

# pip install 필요한 패키지
pip install vllm==0.6.3
pip install ray
pip install -e .
pip install flash-attn --no-build-isolation
pip install wandb IPython matplotlib

# 데이터 준비
# path_to_your_dataset 경로 설정 항상 ./dataset임
python ./examples/data_preprocess/countdown.py --local_dir dataset
#######################################################


