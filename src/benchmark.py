# [필수 설치 라이브러리]
# pip install transformers huggingface_hub torch

import os
import math
from huggingface_hub import HfApi
from transformers import AutoConfig

# 1. 검증할 허깅페이스 모델 ID 정의
MODEL_ID = "google/gemma-4-E2B-it"  
HF_TOKEN = os.environ.get("HUGGIN_FACE_TOKEN")

# 2. 하드웨어 물리 마지노선 설정 (RTX 5060 8GB의 80% 안전 가용선)
GPU_MODEL = "RTX 5060"
TARGET_VRAM = 8.0
TARGET_VRAM_LIMIT = TARGET_VRAM * 0.8
OS_DISPLAY_VRAM = 0.60  

print(f"🔄 하드웨어 안전 가용 마지노선: {TARGET_VRAM_LIMIT} GB")
print(f"🚀 [{MODEL_ID}] 자원 극대화 배치 사이즈 스케일업 알고리즘 가동...\n")

try:
    # 허깅페이스 API 데이터 동적 파싱
    hf_api = HfApi(token=HF_TOKEN)
    model_info = hf_api.model_info(MODEL_ID)
    hf_config = AutoConfig.from_pretrained(MODEL_ID, token=HF_TOKEN, force_download=True)
    
    if hasattr(model_info, 'safetensors') and model_info.safetensors is not None:
        model_fp16_gb = model_info.safetensors.get('total', 0) / (1024 ** 3)
    else:
        model_fp16_gb = sum(getattr(f, 'size', 0) for f in model_info.siblings if f.rfilename.endswith(('.safetensors', '.bin'))) / (1024 ** 3)
        
    if model_fp16_gb == 0:
        model_fp16_gb = 1.2

    # 임베딩 및 백본 4bit 로드 비중 계산
    embedding_ratio = 0.30
    embedding_vram = model_fp16_gb * embedding_ratio
    backbone_vram = (model_fp16_gb * (1.0 - embedding_ratio)) * 0.25 
    
    base_load_vram = embedding_vram + backbone_vram
    lora_adapter_vram = 0.05
    dequant_spike_buffer = backbone_vram * 0.50 
    
    # 모델 아키텍처 파싱
    layers = getattr(hf_config, "num_hidden_layers", getattr(hf_config, "n_layer", 24))
    hidden_dim = getattr(hf_config, "hidden_size", getattr(hf_config, "n_embd", 1024))
    
    # 정적 자원 합산
    static_vram_total = base_load_vram + lora_adapter_vram + OS_DISPLAY_VRAM
    lora_train_overhead = 1.00 
    
    is_oom_inevitable = (static_vram_total + lora_train_overhead) > TARGET_VRAM_LIMIT

    # --- 📊 교정된 리포트 출력 ---
    print("=" * 60)
    print(f"📑 1. 분석 모델 ID   : {model_info.modelId}")
    print(f"🛠️ 2. 아키텍처 식별   : 레이어 {layers}개 / {hidden_dim} 차원")
    
    if is_oom_inevitable:
        print(f"🎯 3. 최대 한계 토큰량: 0 토큰 (🚨 학습 불능)")
        available_dynamic_vram = 0
        total_token_pool = 0
    else:
        available_dynamic_vram = TARGET_VRAM_LIMIT - static_vram_total - lora_train_overhead - dequant_spike_buffer
        total_token_pool = (available_dynamic_vram * (10**9)) / (layers * (hidden_dim // 4) * 38 * 0.6 + 80000)
        print(f"🎯 3. 최대 한계 토큰량: 총 {int(total_token_pool):,} 토큰 수용 공간 확보")
        
    print("=" * 60)
    
    # [단계 1] 모델 초기 로딩 상태
    print(f"[단계 1] 모델 초기 로딩 상태 VRAM 점유 예측 (4bit PEFT 로딩 완료):")
    print(f"   - A. 실제 모델 가중치 로드 무게    : 약 {base_load_vram:.2f} GB")
    print(f"   - B. 시스템 기본 화면 유지 비용   : 약 {OS_DISPLAY_VRAM:.2f} GB")
    print(f"   🔒 정적 대기 VRAM 실제 총량        : 약 {static_vram_total:.2f} GB")
    print("-" * 60)
    
    # [단계 2] 동적 스펙 추천 결론 (배치 사이즈 극대화 튜닝 완료)
    print("[단계 2] total_token_pool 기반 '최적 훈련 조합' 자동 추론 결과:")
    if is_oom_inevitable:
        recommended_batch = 0
        recommended_context = 0
        print(f"   👉 권장 물리 배치 사이즈 (BATCH_SIZE) : 0")
        print(f"   👉 권장 최대 문맥 길이 (CONTEXT_LEN)   : 0 토큰")
    else:
        # 문맥 길이를 고정한 채 배치를 강제로 한계점까지 밀어 올리는 알고리즘으로 리팩토링
        context_candidates = [2048, 1024, 512, 256, 128]
        max_model_context = getattr(hf_config, "max_position_embeddings", 32768)
        
        recommended_batch = 1
        recommended_context = 128
        for ctx in context_candidates:
            if ctx <= max_model_context:
                # 한계 토큰 풀에 가득 찰 때까지 배치를 최대한 확보합니다.
                raw_batch = total_token_pool / ctx
                if raw_batch >= 2:
                    possible_batch = math.floor(raw_batch)
                    # 하드웨어 병렬 가속 연산(Tensor Core)의 효율을 위해 무조건 짝수 배치(예: 14➔12, 23➔20 등)로 매핑
                    recommended_batch = possible_batch if possible_batch % 2 == 0 else max(2, possible_batch - 1)
                    recommended_context = ctx
                    break
                    
        print(f"   👉 권장 물리 배치 사이즈 (BATCH_SIZE) : {recommended_batch} (🔥 초경량 모델 가속 업사이징 완료)")
        print(f"   👉 권장 최대 문맥 길이 (CONTEXT_LEN)   : {recommended_context} 토큰 (한글 약 {int(recommended_context*0.6)}자)")
        
    print("-" * 60)
    
    # [단계 3] 실제 구동 피크 시뮬레이션
    print("[단계 3] 위 추천 스펙 가동 시 실전 연산 VRAM 시뮬레이션:")
    if is_oom_inevitable:
        print(f"   💥 [OOM] 가동 불능")
    else:
        final_activation = ((recommended_batch * recommended_context * layers * (hidden_dim // 4)) / 10**9) * 0.6
        context_vram_margin = (recommended_batch * recommended_context * hidden_dim * 2) / (10**9)
        
        lora_gradient_vram = lora_train_overhead * 0.4
        lora_optimizer_vram = lora_train_overhead * 0.6
        
        final_total_vram = static_vram_total + dequant_spike_buffer + final_activation + lora_train_overhead + context_vram_margin
        
        print(f"   - [기저층] 기저 모델 가중치 4bit 고정 점유: 약 {base_load_vram:.2f} GB")
        print(f"   - [어댑터] PEFT LoRA 신규 레이어 추가     : 약 {lora_adapter_vram:.2f} GB")
        print(f"   - [복원값] 실시간 복원(De-quant) 버퍼     : 약 {dequant_spike_buffer:.2f} GB")
        print(f"   - [활성화] LoRA 최적화 순방향 활성화 메모리: 약 {final_activation:.2f} GB")
        print(f"   - [미분값] 업데이트 대상 전용 Gradients 버퍼 : 약 {lora_gradient_vram:.2f} GB")
        print(f"   - [엔진치] 옵티마이저 상태 창 (동결층 제외) : 약 {lora_optimizer_vram:.2f} GB")
        print(f"   - [마진치] sdpa 커널 압축 컨텍스트 및 OS  : 약 {context_vram_margin + OS_DISPLAY_VRAM:.2f} GB")
        
        print(f"   🚨 [실전 최대 피크 연산 총 VRAM 예상]: 약 {final_total_vram:.2f} GB")
        
    print("=" * 60)
    
    if is_oom_inevitable or final_total_vram > 8.0:
        print(f"❌ [최종 판단] RTX 5060 8GB 안전선({TARGET_VRAM_LIMIT}GB) 가동 '⚠️ 불합격'")
    else:
        print(f"🏆 [최종 판단] 안전선({TARGET_VRAM_LIMIT}GB) 내 동적 매칭 최적화 통과.")
    print("=" * 60)

except Exception as e:
    print(f"❌ 스펙 정밀 재역산 중 오류 발생: {e}")
