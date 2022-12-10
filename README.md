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


