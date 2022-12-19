# Multisession-Chatbot
한국어 멀티세션 챗봇

### **파일 구조**

```bash
.            
│
├── preprocessing               데이터 전처리 
│   ├── main.py                 데이터셋 구축을 위한 실행 코드
│   └── ...                 
│
├── lightning_logs/             훈련 로그 저장 경로
├── model_ckpt/                 모델 저장 경로
├── ...
├── train.py                     모델 학습을 위한 실행 코드
├── generate_chat.py             인퍼런스 실행 코드
└── READMD.md
```


<br>

# **Docker**

## **use docker image url**

pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

https://hub.docker.com/layers/pytorch/pytorch/1.6.0-cuda10.1-cudnn7-devel/images/sha256-ccebb46f954b1d32a4700aaeae0e24bd68653f92c6f276a608bf592b660b63d7?context=explore

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
docker run --gpus all -it --name bot pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
```
만약 학습 시 메모리 오류가 생긴다면  --shm-size, --ulimit memlock=, --ulimit stack 등 명령어로 메모리를 정해준다.

### 5. 기본 update 및 필수 패키지 다운
```bash
apt-get update && apt-get -y install sudo
apt-get upgrade
```
if GPG error 
```bash
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
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

if gpu = [] 오류일시
pytorch와 cuda의 버전 차이때문이므로 cuda와 pytorch의 버전을 맞춰줌 

<br>

----

## **Building Dataset** 

### 1. install package

```bash
sudo apt-get install g++ openjdk-8-jdk python3-dev python3-pip curl         : Install Java 1.8 or up
python3 -m pip install konlpy
sudo apt-get install curl git                                               : Mecab 설치 필요시 시행
bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```

### 2. Build Training, Validation dataset

```bash
cd preprocessing/
```

```bash
python main.py --input_folder data/ --output_folder result/
```

전처리할 `input_folder` 폴더 구조

```bash
input_folder
├── session_2/                
│   ├── train/
│   └── validation/
│       ├── format_train/
│       │     ├── K4-02421-CL21636-CP32259-14-02-S4.json
│       │     └── ...
│       └── format_test/
│
├── session_3/                
│   ├── train/
│   └── validation/
│       ├── format_train/
│       │     ├── K4-02421-CL21636-CP32259-14-02-S4.json
│       │     └── ...
│       └── format_test/
│
└── session_4/                
    ├── train/
    └── validation/
        ├── format_train/
        │     ├── K4-02421-CL21636-CP32259-14-02-S4.json
        │     └── ...
        └── format_test/
```

전처리가 끝나면, `output_folder` 내에 parquet 파일이 생성됨 
```bash
output_folder
├── train.parquet
└── valid.parquet             
```

<br>

---

## **Training Dialogue Model** 

<br>

- `model_type`: 모델 유형      
    - `gpt2` : Pretrained KoGPT2 (`skt/kogpt2-base-v2`)
    - `bart` : Pretrained KoBART (`gogamza/kobart-base-v2`)

_저희 팀은 gpt2를 사용하였습니다._ 

<br>

`data_dir` 내에는 train.parquet, valid.parquet 파일이 있어야 함  
```bash
data_dir
├── train.parquet
└── valid.parquet             
```

훈련 커맨드
```bash
python train.py --max_epochs 10 --data_dir data/ --model_type gpt2 --max_len 256 --reply_len 64 --gpuid 0
```
batch size의 default는 16이며 만약 CUDA out of memory가 난다면 훈련 커맨드에 --batch size 8 를 추가해 batch size를 조절함

<br>

----

## **Inference** 

`input_folder` 폴더 구조

```bash
input_folder                                입력 데이터셋 경로
├── K4-02421-CL21636-CP32259-14-02-S4.json
└── ...               
```

인퍼런스가 끝나면 `output_folder` 내에 xlsx 파일 생성됨
```bash
output_folder                               제출할 xlsx 파일 저장 경로 
├── K4-02421-CL21636-CP32259-14-02-S4.xlsx  
└── ...                 
```


*하나의 GPU만 사용*  


```bash
python generate_chat.py --input_folder data/ --model_type gpt2 --output_folder result/ --max_len 256 --reply_len 64 --gpuid 0 --model_pt <model checkpoint path>
```


<br>


