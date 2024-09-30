<div align='center'>

  # 🏆 LV.1 NLP 기초 프로젝트 : 문맥적 유사도 측정 (STS)

</div>

## ✏️ 대회 소개

| 특징 | 설명 |
|:---:| --- |
| 대회 주제 | 네이버 부스트캠프 AI-Tech 7기 NLP트랙의 level 1 도메인 기초 대회 |
| 대회 설명 | 두 문장이 주어졌을 때 두 문장에 대한 STS(Semantic Text Simliarity)를 추론하는 대회로 Kaggle과 Dacon과 같이 competition 형태 |
| 데이터 구성 | 데이터는 slack 대화, 네이버 영화 후기, 국민 청원 문장으로 구성. Train(9324개), Dev(550개), Test(1100개) |
| 평가 지표 | 모델의 평가지표는 피어슨 상관계수(Pearson correlation coefficient)로 측정 |


## 🎖️ Leader Board
### 🥈 Public Leader Board (2위)
<div align='center'>
<img src="https://github.com/user-attachments/assets/87e0c78b-b426-4af4-a479-1517bc54d6f1" width="1000" />
</div>

### 🥉 Private Leader Board (3위)
<div align='center'>
<img src="https://github.com/user-attachments/assets/29f19cf4-f7d1-48ed-9a71-dc75a4cb947e" width="1000" />
</div>

## 👨‍💻 15조가십오조 멤버
<div align='center'>
  
|김진재 [<img src="https://img1.daumcdn.net/thumb/R800x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FCd4TO%2FbtrUN1rc7Qa%2F3YbSSdRdD020FpAb9R88h0%2Fimg.png" width="20" style="vertical-align:middle;">](https://github.com/jin-jae)| 박규태 [<img src="https://img1.daumcdn.net/thumb/R800x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FCd4TO%2FbtrUN1rc7Qa%2F3YbSSdRdD020FpAb9R88h0%2Fimg.png" width="20" style="vertical-align:middle;">](https://github.com/doraemon500)|윤선웅 [<img src="https://img1.daumcdn.net/thumb/R800x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FCd4TO%2FbtrUN1rc7Qa%2F3YbSSdRdD020FpAb9R88h0%2Fimg.png" width="20" style="vertical-align:middle;">](https://github.com/ssunbear)|이정민 [<img src="https://img1.daumcdn.net/thumb/R800x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FCd4TO%2FbtrUN1rc7Qa%2F3YbSSdRdD020FpAb9R88h0%2Fimg.png" width="20" style="vertical-align:middle;">](https://github.com/simigami)|임한택 [<img src="https://img1.daumcdn.net/thumb/R800x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FCd4TO%2FbtrUN1rc7Qa%2F3YbSSdRdD020FpAb9R88h0%2Fimg.png" width="20" style="vertical-align:middle;">](https://github.com/LHANTAEK)|
|:-:|:-:|:-:|:-:|:-:|
|<img src='https://avatars.githubusercontent.com/u/97018331' height=125 width=125></img>|<img src='https://avatars.githubusercontent.com/u/64678476' height=125 width=125></img>|<img src='https://avatars.githubusercontent.com/u/117508164' height=125 width=125></img>|<img src='https://avatars.githubusercontent.com/u/46891822' height=125 width=125></img>|<img src='https://avatars.githubusercontent.com/u/143519383' height=125 width=125></img>|

</div>

  
## 👼 역할 분담
<div align='center'>

|팀원| 역할 |
  |---| --- |
  | 김진재 | EDA, 방법론 제안, 협업 환경 및 베이스라인 관리, 모델 탐색, 증강 기법 및 전처리 실험, 앙상블 코드 작성 및 실험 |
  | 박규태 | EDA, 모델 탐색, 데이터 증강 및 앙상블 기법에 대한 실험, Bagging 기법 코드 작성 및 실험 |
  | 윤선웅 | EDA, 협업 환경 및 베이스라인 관리, 데이터 분포 및 재분할 총괄, 모델 탐색, 데이터 증강 실험, 앙상블 코드 작성 및 실험 |
  | 이정민 | EDA, 모델 탐색, 모델에 대한 증강 기법 및 전처리 실험, KoEDA 증강 실험, K-Fold Validation 실험 |
  | 임한택 | EDA, 모델 탐색, 모델에 대한 증강 기법 및 전처리 실험,  앙상블 코드 작성 및 실험, Stacking 모델 실험 |
</div>


## 🏃 프로젝트
### 🖥️ 프로젝트 개요
|개요| 설명 |
|:----:| --- |
| 주제 | `STS(Semantic Text Similarity)` : 두 문장의 유사도 정도를 수치로 추론하는 Task |
| 목표 | 두 문장(sentence1, sentence2)이 주어졌을 때, 이 두 문장의 유사도를 0~5사이의 점수로 추론한는 AI 모델 제작 |
| 평가 지표 | 실제 값과 예측값의 피어슨 상관 계수(Pearson Correlation Coefficient) |
| 개발 환경 | `GPU` : Tesla V100 Server 4대, `IDE` : Vscode, Jupyter Notebook |
| 협업 환경 | `Notion`(진행 상황 공유), `Github`(코드 및 데이터 공유), `Slack`(실시간 소통)|

### 📅 프로젝트 타임라인
- 프로젝트는 2024-09-11 ~ 2024-09-27까지 진행되었습니다.

<div align='center'>

![Project Timeline](https://github.com/user-attachments/assets/fc70c368-0d80-4e96-ae33-4ae5769fe7be)

</div>

### 🕵️ 프로젝트 진행
- 프로젝트를 진행하며 단계별로 실험하여 적용한 내용들을 아래와 같습니다.


|프로세스| 설명 |
|:---:| --- |
| EDA | 데이터 분포 분석, Baseline 모델 예측과 실제값 차이 분석  |
| 전처리 | `동의어 교체`, `단어 순서 변경`, `랜덤 삭제` |
| 증강 | label 0 - `undersampling`, label 5 - `copied sentence`, `swapping sentence` |
| 모델 선정 | `upskyy/kf-deberta-multitask`, `team-lucid/deberta-v3-xlarge-korean`, `snunlp/KR-ELECTRA-discriminator`, `kykim/electra-kor-base`, `monologg/ko-electra-base-v3-discriminator`, `jhgan/ko-sroberta-multitask`, `FacebookAI/roberta-large-rtt`, `deliciouscat/kf-deberta-base-cross-sts`, `sorryhyun-sentence-embedding-klue-large` |
| 앙상블 | `soft voting`, `Nested Ensemble`|

### 📊 Dataset
- 데이터 증강 과정에서 라벨 분포를 균형있게 맞추고자 라벨별 증강비율을 조정하였습니다.

|버전| 설명 |크기|
|---| --- |---|
| original_train_V1 | 원본 데이터  |9324|
| augmentation_train_V2 | `SWAP`, `label 0 언더샘플링` + `label 5 오버샘플링` |28722|

### 🤖 Ensemble Model
- 최종적으로 16개의 모델을 앙상블에 사용했습니다.

| Model | val_pearson | learning_rate| batch_size | 사용 데이터 |
|-------------------------| --- |---|----- |---|
|upskyy/kf-deberta-multitask| 0.9289 |1e-5| 16 |augmentation_train_V2|
|team-lucid/deberta-v3-xlarge-korean| 0.9378 |1e-5| 16 |augmentation_train_V2|
|team-lucid/deberta-v3-xlarge-korean| 0.9377 |1e-5| 16 |original_train_V1|
|snunlp/KR-ELECTRA-discriminator|0.9325 |1e-5| 16 |original_train_V1|
|snunlp/KR-ELECTRA-discriminator|0.9313 |1e-5| 32 |original_train_V1|
|kykim/electra-kor-base|0.9255 |1e-5| 16 |original_train_V1|
|monologg/ko-electra-base-v3-discriminator|0.9252|1e-5| 16 |original_train_V1|
|kykim/electra-kor-base|0.9252|1e-5| 16 |augmentation_train_V2|
|jhgan/ko-sroberta-multitask|0.9249|1e-5| 16 |original_train_V1|
|FacebookAI/roberta-large-rtt|0.9249|1e-5| 16 |original_train_V1|
|snunlp/KR-ELECTRA-discriminator|0.9223|1e-5| 16 |augmentation_train_V2|
|sorryhyun-sentence-embedding-klue-large|0.9301|1e-5| 16 |augmentation_train_V2|
|FacebookAI/xlm-roberta-large|0.9287|1e-5| 16 |augmentation_train_V2|
|deliciouscat/kf-deberta-base-cross-sts|0.929|1e-5| 16 |augmentation_train_V2|
|team-lucid-deberta-v3-xlarge-korean|0.9399|1e-5| 16 |augmentation_train_V2|
|snunlp-KR-ELECTRA-discriminator|0.9336|1e-5| 16 |augmentation_train_V2|


## 📁 프로젝트 구조

```
📁 level1-semantictextsimilarity-nlp-15
├── README.md
├── requirements.txt
└── src
    ├── bagging.py
    ├── config.yaml
    ├── csv_ensemble
    ├── data
    ├── ensemble.py
    ├── inference.py
    ├── model
    │   └── model.py
    ├── output
    ├── run.py
    ├── train.py
    └── util
        ├── data_augmentation.py
        └── util.py
```

### src 폴더 구조 설명

- augmentation : 데이터 증강 관련 코드
- checkpoint : 체크포인트 파일(ckpt) 저장 폴더
- config : 모델 설정 관련 yaml 파일
- model : 모델 클래스가 존재하는 코드 + 모델 .pt 파일
- output : 모델 학습 결과 csv 파일
- util : 기타 유틸리티(dataset, dataloader, tokenizer) 코드
- run.py : 학습 및 추론을 실행하는 코드
- train.py : 학습을 실행하는 코드
- inference.py : 추론을 실행하는 코드
- ensemble.py : 앙상블을 실행하는 코드

### 보충 설명
1. path, 하이퍼파라미터 값과 같은 것은 전부 config.yaml에서 관리합니다.
2. config.yaml에 존재하는 모델 목록이 전부 run.py에서 for문을 돌려서 학습을 진행합니다.<br>따라서 모델을 변경할 때 yaml에 주석을 이용해주세요
3. 앙상블은 config.yaml의 ensemble_weight을 잘 조절해 주세요 
4. util의 dataset 클래스에서 다른 부분이 많을 것 같은데 많이 다르다면 논의를 해봐도 될 거 같습니다.
5. 빠진 파일들은 gitignore를 잘 확인해주세요
6. 오류나 질문 등은 카톡이나 git issue를 통해 남겨주세요

### Git 관련
1. Commit Message Convention은 다음 사이트를 참조하여 보내시면 됩니다.<br>https://github.com/joelparkerhenderson/git-commit-message?tab=readme-ov-file#top-priorities
2. Branch Convention은 추후 추가하도록 하겠습니다.

### Installation
1. pip install -r requirements.txt
2. Put train, dev, test csv files at /src/data directory
3. Put sample_submission.csv at /src/output directory
4. Set models and augmentation methods on config.yaml
5. Execute run.py
