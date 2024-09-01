
## 1. 대회 목표
"두개의 llm 생성 문장 중 인간이 선호하는 문장 분류"


## 2. 규칙

- 인풋 : prompt, model_a_response, model_b_response
- 트레인 데이터 : 각 생성 문장이 어떤 모델에서 생성되었는지 정보가 존재
- 오프라인
- 평가 F1 스코어 : log loss (확률 분포)


## 3. eda
- 모델 분포 확인
    - Check the winner distribution
    - Check the n_gram distribution
    - Check label distribution (text length or n_gram or model)
- n_gram 확인> Top Unigrams are tokenizer checks
    - Locations within the main token sequence
- 텍스트 길이 확인
- 토큰 길이 확인
- 중복 확인
- 라벨 분포 확인

## 4. 전처리
- 학습 데이터를 prompt/chosen/rejected 형태로 변환(winner_tie 제외)
- 데이터셋 증분 > 큰 성능 향상

## 5. 기획
### DPO 학습
- huggingface trl 라이브러리 dpo 트레이너 학습
### 분류기 추가
- 학습 데이터를 하나의 프롬프트 생성
- 파인튠 완료 모델에 아웃풋 3개인 분류기 추가 후 재학습

### 프롬프트 엔지니어링
- 파인튠 완료 모델에 테스트데이터 입력 시 아웃풋을 세개의 확률로 출력하도록 프롬프트 작성

### 앙상블
llm + tf-idf + sequence length classifier



## 6. 고려할 점
- 캐글 노트북 환경에서 학습이 가능한 모델 선택
- (2048 embedding size× 1000(~1536)^2 sequence length + 1,300,000,000 parameter)× 1 batch size× 4 float32 = 13G GPU memory
- 정적 양자화는 느리고 메모리를 많이 사용 > torch.quantization.quantize_dynamic은 cpu만 지원
