# Guardrails Benchmark

한국어 유해 프롬프트에 대한 AI 안전 가드레일 시스템 성능 벤치마킹 도구입니다.

## 📋 프로젝트 개요

이 프로젝트는 한국어 유해 콘텐츠 탐지를 위한 두 가지 가드레일 시스템의 성능을 비교 평가합니다:
- **AWS Bedrock Guardrail**: Amazon의 클라우드 기반 안전 필터링 서비스
- **Kanana Safeguard**: 커스텀 AI 안전 모델 (8B, Siren 8B, Prompt 2.1B 앙상블)

## 🎯 주요 기능

- 한국어 유해 프롬프트 데이터셋 자동 처리
- 두 가드레일 시스템의 병렬 성능 테스트
- 정량적 성능 비교 및 분석
- 결과 시각화 및 리포트 생성

## 📂 프로젝트 구조

```
guardrails_benchmark/
├── guardrails_bench_kanana_bedrock.py  # 메인 벤치마킹 스크립트
├── harmful_prompts_results.csv         # 벤치마크 테스트 결과
├── result.ipynb                        # 결과 분석 노트북
└── README.md                          # 프로젝트 문서
```

## 🔧 사용된 기술 스택

- **Python 3.x**
- **Datasets**: Hugging Face datasets 라이브러리
- **AWS Boto3**: Bedrock 서비스 연동
- **LangChain**: AI 모델 인터페이스
- **Pandas**: 데이터 처리 및 분석
- **Langfuse**: AI 모델 모니터링 및 추적

## 📊 데이터셋

- **소스**: `iknow-lab/wildguardmix-train-ko` 
- **언어**: 한국어
- **테스트 샘플**: 10,000개의 유해 프롬프트
- **평가 기준**: harmful/safe 분류 정확도

## 🚀 사용 방법

### 1. 환경 설정

```bash
pip install datasets pandas langfuse boto3 langchain-openai
```

### 2. 인증 정보 설정

스크립트 내에서 다음 정보를 설정하세요:
- AWS Bedrock 접근 권한
- Langfuse API 키
- Kanana Safeguard 모델 엔드포인트

### 3. 벤치마크 실행

```bash
python guardrails_bench_kanana_bedrock.py
```

### 4. 결과 분석

```bash
jupyter notebook result.ipynb
```

## 📈 벤치마크 결과

최근 테스트 결과 (10,000개 샘플):

| 가드레일 시스템 | UNSAFE 판정 | 탐지율 |
|----------------|-------------|--------|
| Bedrock Guardrail | 7,123개 | 71.2% |
| Kanana Safeguard | 8,534개 | 85.3% |