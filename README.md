# Multisession-Chatbot
한국어 멀티세션 챗봇

### **파일 구조**

```bash
.
├── data                        데이터셋 저장 경로
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv               
│
├── preprocessing               데이터 라벨링 및 전처리 
│   ├── ...
│   ├── build_dataset.py        데이터셋 구축을 위한 실행 코드
│   └── ...                 
│
├── result/                     모델 테스트 결과 저장 경로
├── utils/
├── ...
├── main.py                     모델 학습 및 테스트를 위한 실행 코드
├── READMD.md
└── ...
```

<br>

# **Docker**

## **use docker image url**

nvcr.io/nvidia/tensorflow:20.03-tf2-py3

### 1. Nvidia Container Toolkit 설치
WSL 환경에서 CUDA를 사용하기 위해 WSL에서 UBUNTU에서 아래 명령어 실행
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```

### 2. NVIDIA runtime package 설치
```bash
sudo apt-get update
sudo apt-get install -y nvidia-docker2
```

### 3. docker restart
```bash
sudo service docker stop
sudo service docker start
```

### 4. dock image 다운 및 컨테이너 실행
bot 이라는 이름으로 실행
```bash
docker run --gpus all -it --name bot --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/tensorflow:20.03-tf2-py3
```

### 5. 기본 update 및 필수 패키지 다운
```bash
apt-get update && apt-get -y install sudo
apt-get upgrade
```

## **install requirements**

```bash
cd Multisession-Chatbot
pip install -r requirements.txt
```

if HOROVOD 오류일시
```bash
HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir --upgrade --force-reinstall horovod && ldconfig
```

## **Building Dataset** 


```bash
cd preprocessing/
```

### 1. Build Training, Validation, Test dataset
```bash
python build_dataset.py --preprocessing --split --data_dir ../data --result_dir ../result
```

<br>

---

## **Training/Testing Dialogue Model** 

<br>

- `model_type`: 모델 유형      
    - `gpt2` : Pretrained KoGPT2 (`skt/kogpt2-base-v2`)
    - `bart` : Pretrained KoBART (`gogamza/kobart-base-v2`)

### 1. Training

```bash
python main.py --train --max_epochs 10 --data_dir data/ --model_type gpt2 --model_name gpt2_chat --max_len 64 --gpuid 0
```

<br>

### 2. Testing

*하나의 GPU만 사용*  

#### (1) `<data_dir>`/test.csv에 대한 성능 테스트

```bash
python main.py --data_dir data/ --model_type gpt2 --model_name gpt2_chat --save_dir result --max_len 64 --gpuid 0 --model_pt <model checkpoint path>
```

#### (2) 사용자 입력에 대한 성능 테스트

```bash
python main.py --chat --data_dir data/ --model_type gpt2 --max_len 64 --gpuid 0 --model_pt <model checkpoint path>
```

<br>


