# [필수 설치 라이브러리]
# pip install datasets pandas requests openpyxl

import os
import requests
import json
import pandas as pd
from datasets import load_dataset

# Ollama 로컬 서버 기본 주소 세팅
OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "hf.co/google/gemma-4-E4B-it-qat-q4_0-gguf"

TOTAL_DATA_SIZE = 1000      # 1차로 추출할 총 VOC 데이터 건수
INFERENCE_BATCH_SIZE = 5    # 실제 AI 추론을 돌릴 테스트 건수
CURRENT_MODE = "train"      # 실행 MODE train,test
DATA_URLS  = {
        "train" : "https://raw.githubusercontent.com/e9t/nsmc/refs/heads/master/ratings_train.txt",
        "test" : "https://raw.githubusercontent.com/e9t/nsmc/refs/heads/master/ratings_test.txt"
        }

print("🔄 1단계: 허깅페이스에서 대규모 한국어 NSMC 말뭉치 다운로드...")
nsmc_dataset = load_dataset(
    "csv", 
    data_files=DATA_URLS[CURRENT_MODE],
    delimiter="\t",
    split=CURRENT_MODE
)

#df = pd.DataFrame(nsmc_dataset['train'])
df = pd.DataFrame(nsmc_dataset)

# 실무 규격으로 컬럼명 변경 (영화 리뷰 ➔ 고객 불만 VOC)
df = df.rename(columns={'document': 'voc_text', 'label': 'is_complaint'})

# 불만성(부정) VOC 데이터 TOTAL_DATA_SIZE 건 추출
complaint_voc = df[df['is_complaint'] == 0].head(TOTAL_DATA_SIZE).reset_index(drop=True)

# 5. [★핵심] 추출한 1,000건을 모델 학습/입력에 쓸 수 있도록 다시 허깅페이스 Dataset으로 최종 변환!
#final_model_dataset = load_dataset.from_pandas(complaint_voc)

print(f"✅ 가상 사내 불만 VOC 데이터 {len(complaint_voc)}건 확보 완료!")

print(f"\n🔄 2단계: Ollama 엔진({MODEL_NAME}) 기반 가상 VOC 추론 가동...")

results = [] # 최종 분석 결과 리포트를 저장할 배열

# 1,000건 중 먼저 상위 INFERENCE_BATCH_SIZE 건만 안전하게 반복문(배치) 테스트 실행
for idx, row in complaint_voc.head(INFERENCE_BATCH_SIZE).iterrows():
    voc_content = row['voc_text']
    print(f"\n📥 [{idx+1}번 민원 접수]: {voc_content}")
    
    # Ollama가 인식할 수 있는 표준 시스템/유저 프롬프트 메시지 구조 설계
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "너는 사내 고객만족 팀의 데이터 분석가야. 답변은 마크다운이나 불필요한 미사여구 없이 지시한 레이아웃 양식으로만 깔끔하게 대답해."
            },
            {
                "role": "user",
                "content": f"다음 사내 고객 민원을 분석해서 '핵심 불만 사항 1줄 요약'과 '상담사 대응 가이드 1줄'을 작성해줘.\n민원내용: {voc_content}"
            }
        ],
        "options": {
            "temperature": 0.1
        },
        "stream": False # 실시간 타이핑 대신 문장이 완성되면 한 번에 수신 (배치 처리 최적화)
    }
    
    try:
        # 파이썬에서 Ollama 엔진으로 POST 요청 전송 (VRAM 점유는 Ollama가 전담)
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
        
        if response.status_code == 200:
            ai_response = response.json()['message']['content'].strip()
            print(f"🤖 [Gemma-4 분석 완료]:\n{ai_response}")
            
            # 결과 적재
            results.append({
                "민원_번호": idx + 1,
                "민원_원문": voc_content,
                "AI_분석_리포트": ai_response
            })
        else:
            print(f"❌ Ollama 서버 응답 에러: {response.status_code}")
            
    except Exception as e:
        print(f"❌ 통신 오류 발생: {e}")
        print("💡 팁: 터미널에서 'ollama serve'가 켜져 있는지 확인하세요.")
        break

print("\n🔄 3단계: 분석 데이터 정제 및 사내 포트폴리오 파일 저장...")
if results:
    # 수집된 AI 리포트 리스트를 판다스 데이터프레임으로 최종 변환
    report_df = pd.DataFrame(results)
    
    #엑셀(Excel) 파일로 저장
    output_filename = "사내_VOC_AI_분석_리포트.xlsx"
    report_df.to_excel(output_filename, index=False)
    
    print(f"📊 [프로젝트 산출물 추출 성공] '{output_filename}' 파일이 프로젝트 폴더에 저장되었습니다.")
else:
    print("❌ 적재된 분석 결과가 없어 파일 저장을 건너뜁니다.")
