# 🚀 로컬 인프라 기반 사내 문서 분석 및 자동화 에이전트 시스템

본 프로젝트는 대기업 및 금융권의 폐쇄망 환경을 상정하여, 사내 보안을 완벽하게 지키면서 비정형 고객 민원(VOC) 및 계약서 문서를 자동으로 분류하고 요약하는 백엔드 파이프라인 시스템입니다.

## 🛠️ 개발 및 가동 인프라 환경
- **Target H/W:** NVIDIA GeForce RTX 5060 (VRAM 8GB)
- **Runtime:** Python 가상환경 + 로컬 Ollama 서빙 인프라
- **Main Core Engine:** `Qwen3.5-4B-Instruct` (하드웨어 자원 역산 최적화 완료)

## 📂 산출물 폴더 구조 (Directory Tree)
- `docs/experiment-plan.md` : 하드웨어 제약 조건 및 실험 계획서
- `docs/final-selection-report.md` : 가용 자원 검증 및 최종 모델 선정 보고서
- `src/benchmark.py` : 메모리 계산 스크립트
- `src/runModel.py` : 선정 모델 구동 파일
- `results/local_summary.csv` : 모델 성능 측정 결과 측정 파일
- `results/local_results.xlsx` : 모델 예측 결과 파일

## 🎯 주요 비즈니스 핵심 성과 (Key 아웃풋)
1. **자원 최적화(MLOps):** 메모리 모니터링 분석을 통해 물리적 한계점 내에서 최적의 훈련 규격(Batch 2 / Context 1024)을 유도하여 자원 낭비 최소화.
2. **비용 절감:** 외부 유료 API 구독 비용 없이 내 로컬 그래픽카드 자원만으로 실시간 사내 데이터 자동화 처리 기틀 마련.
3. **완벽한 보안 내재화:** 외부망과의 유출 경로를 원천 차단하여 기업 기밀 문서 유실 리스크 제로 달성.
