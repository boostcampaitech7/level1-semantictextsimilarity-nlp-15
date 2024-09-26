<div align='center'>

# LV.1 NLP 기초 프로젝트 : 문맥적 유사도 측정 (STS)

</div>

<div align='center'>

## 15조가십오조

|김진재 [<img src="ETC/img/github-mark.png" width="20" style="vertical-align:middle;">](https://github.com/jin-jae)| 박규태 [<img src="ETC/img/github-mark.png" width="20" style="vertical-align:middle;">](https://github.com/doraemon500)|윤선웅 [<img src="ETC/img/github-mark.png" width="20" style="vertical-align:middle;">](https://github.com/ssunbear)|이정민 [<img src="ETC/img/github-mark.png" width="20" style="vertical-align:middle;">](https://github.com/simigami)|임한택 [<img src="ETC/img/github-mark.png" width="20" style="vertical-align:middle;">](https://github.com/LHANTAEK)|
|:-:|:-:|:-:|:-:|:-:|
|<img src='ETC/img/jin-jae.png' height=125 width=125></img>|<img src='ETC/img/doraemon500.png' height=125 width=125></img>|<img src='ETC/img/ssunbear.png' height=125 width=125></img>|<img src='ETC/img/simigami.png' height=125 width=125></img>|<img src='ETC/img/LHANTAEK.png' height=125 width=125></img>|


</div>

<div align='center'>
  
## 역할 분담

|팀원| 역할 |
|:---:| --- |
| 김진재 |  |
| 박규태 |  |
| 윤선웅 |  |
| 이정민 |  |
| 임한택 |  |

</div>

<div align='left'>
  
## src 폴더 구조 설명

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

## 보충 설명
1. path, 하이퍼파라미터 값과 같은 것은 전부 config.yaml에서 관리합니다.
2. config.yaml에 존재하는 모델 목록이 전부 run.py에서 for문을 돌려서 학습을 진행합니다.<br>따라서 모델을 변경할 때 yaml에 주석을 이용해주세요
3. 앙상블은 config.yaml의 ensemble_weight을 잘 조절해 주세요 
4. util의 dataset 클래스에서 다른 부분이 많을 것 같은데 많이 다르다면 논의를 해봐도 될 거 같습니다.
5. 빠진 파일들은 gitignore를 잘 확인해주세요
5. 오류나 질문 등은 카톡이나 git issue를 통해 남겨주세요

</div>

<div align='left'>
  
## Git 관련

1. Commit Message Convention은 다음 사이트를 참조하여 보내시면 됩니다.<br>https://github.com/joelparkerhenderson/git-commit-message?tab=readme-ov-file#top-priorities
2. Branch Convention은 추후 추가하도록 하겠습니다.

</div>



